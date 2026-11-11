"""Tests for the optional batched access recording mode.

Exact "least-recently-used" behavior remains the default. Setting
``access_log_limit`` to a positive value buffers cache hits per process and
flushes them in one transaction, which trades exact recency ordering for
fewer SQLite writes on the hit path.

Tests count logical SQL statements (through the ``_sql`` wrapper) instead of
timing individual calls, and use a controllable clock to make refresh
boundaries deterministic.
"""

import shutil
import subprocess as sp
import sys
import tempfile
import threading
from unittest import mock

import pytest

import diskcache as dc


class CountingCache(dc.Cache):
    """Cache that records every logical SQL statement it executes."""

    def __init__(self, *args, **kwargs):
        self.statements = []
        super().__init__(*args, **kwargs)

    @property
    def _sql(self):
        execute = super()._sql
        statements = self.statements

        def counting(statement, *args, **kwargs):
            statements.append(statement)
            return execute(statement, *args, **kwargs)

        return counting

    def clear_statements(self):
        del self.statements[:]

    def count_access_updates(self):
        """Count UPDATE statements whose SET clause updates access_time.

        Both the exact-mode hit (``UPDATE Cache SET access_time = ...``) and
        the batched flush (same prefix plus a ``WHERE access_time < ...``
        guard) match. Row replacements that also touch access_time as part
        of a full-column update do not.
        """
        prefix = 'UPDATE Cache SET access_time ='
        return sum(
            1
            for statement in self.statements
            if statement.startswith(prefix)
        )

    def count_write_transactions(self):
        """Count ``BEGIN IMMEDIATE`` statements (SQLite writer locks)."""
        return sum(
            1
            for statement in self.statements
            if statement == 'BEGIN IMMEDIATE'
        )


@pytest.fixture
def clock():
    """Controllable clock patched into ``time.time`` for the test body."""
    state = {'now': 1000.0}

    with mock.patch('time.time', lambda: state['now']):
        yield state


def make_cache(directory, **settings):
    settings.setdefault('eviction_policy', 'least-recently-used')
    return CountingCache(directory, **settings)


def cleanup(cache):
    cache.close()
    shutil.rmtree(cache.directory, ignore_errors=True)


def access_time(cache, key):
    ((value,),) = cache._sql(
        'SELECT access_time FROM Cache WHERE key = ?', (key,)
    ).fetchall()
    return value


def test_default_is_exact():
    with dc.Cache() as cache:
        assert cache.access_log_limit == 0
        assert cache._access_log == {}


def test_exact_mode_writes_every_hit(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory)

    try:
        for key in ('a', 'b'):
            cache.set(key, key)

        cache.clear_statements()

        for _ in range(5):
            cache.get('a')

        assert cache.count_access_updates() == 5
        assert cache._access_log == {}
        assert access_time(cache, 'a') == 1000.0
    finally:
        cleanup(cache)


def test_batched_hits_skip_writer_locks(clock):
    """Batched hits read without BEGIN IMMEDIATE; the flush takes the lock."""
    directory = tempfile.mkdtemp()
    exact = make_cache(tempfile.mkdtemp())
    batched = make_cache(directory, access_log_limit=100)

    try:
        clock['now'] = 900.0

        for key in ('a', 'b'):
            exact.set(key, key)
            batched.set(key, key)

        exact.clear_statements()
        batched.clear_statements()

        clock['now'] = 1100.0

        for _ in range(10):
            exact.get('a')
            batched.get('a')

        # Exact mode takes a writer lock on every hit.
        assert exact.count_write_transactions() == 10

        # Batched mode takes none on the hit path.
        assert batched.count_write_transactions() == 0
        assert batched.count_access_updates() == 0
        assert batched._access_log == {1: 1100.0}

        # The single flush takes one lock and applies one update.
        assert batched.flush_access_log() == 1
        assert batched.count_write_transactions() == 1
        assert batched.count_access_updates() == 1
    finally:
        cleanup(exact)
        cleanup(batched)


def test_batched_mode_defers_hits(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=3)

    try:
        cache.set('a', 1)
        cache.clear_statements()

        cache.get('a')
        assert cache._access_log == {1: 1000.0}
        assert cache.count_access_updates() == 0
        assert access_time(cache, 'a') == 1000.0

        cache.flush_access_log()
        assert cache._access_log == {}
        assert cache.count_access_updates() == 1
        assert access_time(cache, 'a') == 1000.0
    finally:
        cleanup(cache)


def test_write_amplification_is_bounded(clock):
    """Batched writes never exceed exact writes and shrink with hot keys."""
    runs = 24
    keys = ('hot', 'cold')
    limit = 4

    exact = make_cache(tempfile.mkdtemp())
    batched = make_cache(tempfile.mkdtemp(), access_log_limit=limit)

    try:
        for key in keys:
            exact.set(key, key)
            batched.set(key, key)

        # Warm both caches so every measured access is a hit.
        exact.get('hot')
        exact.get('cold')
        batched.get('hot')
        batched.get('cold')
        exact.clear_statements()
        batched.clear_statements()

        for step in range(runs):
            clock['now'] = 1000.0 + step + 1
            exact.get('hot')
            batched.get('hot')

            if step % 3 == 0:
                exact.get('cold')
                batched.get('cold')

        expected_flushes = (runs + runs // 3) // limit + 1
        exact_writes = exact.count_access_updates()
        batched_writes = batched.count_access_updates()

        # Exact mode writes once per hit; batched mode writes nothing on
        # the hit path until the buffer is flushed.
        assert exact_writes == runs + runs // 3
        assert batched_writes == 0
        assert batched_writes <= expected_flushes
        # Every recorded access lands on or before its flush time.
        assert access_time(batched, 'hot') <= clock['now']
        assert access_time(batched, 'cold') <= clock['now']

        # Flushing in the same order reproduces exact recency for both keys.
        batched.flush_access_log()
        assert access_time(batched, 'hot') == access_time(exact, 'hot')
        assert access_time(batched, 'cold') == access_time(exact, 'cold')
    finally:
        cleanup(exact)
        cleanup(batched)


def test_flush_triggers(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=3)

    try:
        cache.set(0, 0)
        cache.set(1, 1)
        cache.set(2, 2)
        cache.clear_statements()

        # 1. Buffer reaches the limit: flushed inline with zero writes first.
        for key in (0, 1, 2, 3):
            clock['now'] = 1000.0 + key
            cache.set(key, key)
            cache.get(key)

        # The third distinct hit triggered one flush of three rows; the
        # fourth hit started a fresh buffer.
        assert cache._access_log == {4: 1003.0}
        assert cache.count_access_updates() == 3

        # 2. Explicit flush.
        cache.clear_statements()
        cache.get(0)
        assert cache._access_log
        assert cache.flush_access_log() == 1

        # 3. Normal close flushes what is still buffered.
        cache.get(0)
        assert cache._access_log
        directory = cache.directory
        cache.close()

        reopened = make_cache(directory, access_log_limit=3)
        assert reopened._access_log == {}
        assert access_time(reopened, 0) == clock['now']
        # Keep the directory: later sections reopen the same database.
        reopened.close()

        # 4. Switching the limit flushes pending records.
        cache = make_cache(directory, access_log_limit=3)
        clock['now'] = 2000.0
        cache.get(0)
        assert cache._access_log
        cache.reset('access_log_limit', 5)
        assert cache._access_log == {}
        assert access_time(cache, 0) == 2000.0

        clock['now'] = 2100.0
        cache.get(0)
        assert cache._access_log
        cache.reset('access_log_limit', 0)
        assert cache._access_log == {}
        assert access_time(cache, 0) == 2100.0
        cleanup(cache)

        # 5. Bulk deletion flushes pending records for surviving keys.
        cache = make_cache(directory, access_log_limit=3)
        cache.set(0, 0)
        cache.get(0)
        assert cache._access_log
        cache.clear()
        assert cache._access_log == {}
        cleanup(cache)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_late_access_does_not_move_time_backwards(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)

    try:
        cache.set('a', 1)
        cache.get('a')  # Buffered at t=1000.
        assert cache._access_log == {1: 1000.0}

        # A newer write, from this or another process, lands first.
        cache._sql(
            'UPDATE Cache SET access_time = ? WHERE rowid = ?',
            (2000.0, 1),
        )
        assert cache.flush_access_log() == 0
        assert access_time(cache, 'a') == 2000.0
    finally:
        cleanup(cache)


def test_deleted_key_is_not_resurrected(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)

    try:
        cache.set('a', 1)
        cache.get('a')

        # Delete through a raw statement so the buffer keeps the stale record.
        cache._sql('DELETE FROM Cache WHERE rowid = ?', (1,))
        assert cache._access_log == {1: 1000.0}

        assert cache.flush_access_log() == 0
        assert len(cache) == 0
        assert cache._sql('SELECT COUNT(*) FROM Cache').fetchall() == [(0,)]
    finally:
        cleanup(cache)


def test_repeated_accesses_collapse(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)

    try:
        cache.set('a', 1)

        for offset in range(5):
            clock['now'] = 1000.0 + offset
            cache.get('a')

        assert len(cache._access_log) == 1
        assert cache._access_log == {1: 1004.0}

        assert cache.flush_access_log() == 1
        assert access_time(cache, 'a') == 1004.0
    finally:
        cleanup(cache)


def test_overwrite_discards_stale_record(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)

    try:
        cache.set('a', 'old')
        cache.get('a')  # Buffered at t=1000.

        clock['now'] = 1300.0
        cache.set('a', 'new')
        assert cache._access_log == {}

        cache.flush_access_log()
        assert access_time(cache, 'a') == 1300.0
    finally:
        cleanup(cache)


def test_eviction_uses_flushed_order(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)
    cache.reset('size_limit', int(10.1e6))
    cache.reset('cull_limit', 5)

    try:
        million = b'x' * int(1e6)

        for value in range(10):
            cache.set(value, million)

        clock['now'] = 1100.0

        for value in (0, 1, 7, 8, 9):
            cache.get(value)

        cache.flush_access_log()
        clock['now'] = 1200.0
        cache.set(10, million)

        assert sorted(cache) == [0, 1, 7, 8, 9, 10]
        assert len(cache.check()) == 0
    finally:
        cleanup(cache)


def test_cross_process_visibility(clock):
    directory = tempfile.mkdtemp()
    parent = make_cache(directory, access_log_limit=100)

    try:
        parent.set('a', 'value')
        parent.close()

        code = (
            'import diskcache as dc\n'
            'from unittest import mock\n'
            'with mock.patch("time.time", lambda: 5000.0):\n'
            '    cache = dc.Cache(%r, eviction_policy="least-recently-used",'
            ' access_log_limit=100)\n'
            '    cache.get("a")\n'
            '    cache.flush_access_log()\n'
            '    cache.close()\n'
        ) % directory

        sp.run([sys.executable, '-c', code], check=True)

        child = make_cache(directory, access_log_limit=100)
        assert access_time(child, 'a') == 5000.0
        assert child.get('a') == 'value'
        child.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_abnormal_exit_keeps_last_flushed_time(clock):
    directory = tempfile.mkdtemp()

    try:
        parent = make_cache(directory, access_log_limit=100)
        parent.set('a', 'value')
        parent.close()

        code = (
            'import os\n'
            'import diskcache as dc\n'
            'from unittest import mock\n'
            'clock = {"now": 2500.0}\n'
            'with mock.patch("time.time", lambda: clock["now"]):\n'
            '    cache = dc.Cache(%r, eviction_policy="least-recently-used",'
            ' access_log_limit=100)\n'
            '    cache.get("a")\n'
            '    assert cache._access_log\n'
            '    os._exit(0)\n'
        ) % directory

        sp.run([sys.executable, '-c', code], check=True)

        survivor = make_cache(directory, access_log_limit=100)
        # The pending record was lost with the process; the database keeps
        # the last flushed time and the key stays usable.
        assert access_time(survivor, 'a') == 1000.0
        assert survivor.get('a') == 'value'
        survivor.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_incr_defer_access_but_writes_value(clock):
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=100)

    try:
        clock['now'] = 1000.0
        assert cache.incr('n') == 1
        clock['now'] = 1100.0
        assert cache.incr('n') == 2
        assert cache.decr('n') == 1

        # The value is written immediately while access is buffered.
        assert cache._access_log == {1: 1100.0}
        assert cache._sql(
            'SELECT value FROM Cache WHERE key = ?', ('n',)
        ).fetchall() == [(1,)]

        assert cache.flush_access_log() == 1
        assert access_time(cache, 'n') == 1100.0
    finally:
        cleanup(cache)


def test_other_policies_ignore_access_log(clock):
    directory = tempfile.mkdtemp()

    try:
        stored = CountingCache(
            directory,
            eviction_policy='least-recently-stored',
            access_log_limit=100,
        )
        stored.set('a', 1)
        stored.get('a')
        assert stored._access_log == {}
        assert stored.count_access_updates() == 0
        stored.close()

        used = CountingCache(
            directory,
            eviction_policy='least-recently-used',
            access_log_limit=100,
        )
        used.set('b', 2)
        clock['now'] = 1100.0
        used.get('b')
        assert used._access_log == {2: 1100.0}
        assert used.flush_access_log() == 1
        used.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_zero_limit_is_exact_and_invalid_limits_rejected(clock):
    directory = tempfile.mkdtemp()

    try:
        cache = CountingCache(
            directory,
            eviction_policy='least-recently-used',
            access_log_limit=0,
        )
        cache.set('a', 1)
        cache.clear_statements()
        cache.get('a')
        assert cache.count_access_updates() == 1
        assert cache._access_log == {}
        cache.close()

        with pytest.raises(ValueError):
            CountingCache(directory, access_log_limit=-1)

        with pytest.raises(ValueError):
            CountingCache(directory, access_log_limit=1.5)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_concurrent_readers(clock):
    """Concurrent readers share one buffer without errors or lost keys."""
    directory = tempfile.mkdtemp()
    cache = make_cache(directory, access_log_limit=16)

    try:
        for key in range(8):
            cache.set(key, key)

        errors = []

        def reader():
            try:
                for _ in range(200):
                    for key in range(8):
                        clock['now'] += 1.0
                        assert cache.get(key) == key
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=reader) for _ in range(4)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors
        assert len(cache) == 8

        cache.flush_access_log()
        assert cache._access_log == {}

        for key in range(8):
            assert cache.get(key) == key

        assert len(cache.check()) == 0
    finally:
        cleanup(cache)


def test_fanout_flush_and_close(clock):
    directory = tempfile.mkdtemp()
    cache = dc.FanoutCache(
        directory,
        eviction_policy='least-recently-used',
        access_log_limit=100,
    )

    try:
        cache.set('a', 1)
        clock['now'] = 1500.0
        assert cache.get('a') == 1
        assert any(shard._access_log for shard in cache._shards)

        applied = cache.flush_access_log()
        assert applied == 1
        assert not any(shard._access_log for shard in cache._shards)

        clock['now'] = 1600.0
        assert cache.get('a') == 1
        cache.close()

        # The key is hashed to one shard; check every shard directory.
        found = 0

        for shard in cache._shards:
            reopened = dc.Cache(
                shard.directory,
                eviction_policy='least-recently-used',
            )
            rows = reopened._sql(
                'SELECT access_time FROM Cache WHERE access_time = 1600.0'
            ).fetchall()
            found += len(rows)
            reopened.close()

        assert found == 1
    finally:
        shutil.rmtree(directory, ignore_errors=True)
