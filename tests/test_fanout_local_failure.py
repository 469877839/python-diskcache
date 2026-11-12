"""Regression tests for FanoutCache partial-failure behavior.

These tests hold a write transaction on ONE shard in a separate process and
verify how FanoutCache answers callers while the remaining shards keep
serving. Synchronization uses process events (barriers) plus short SQLite
timeouts; no probabilistic collisions are involved.

Determinism notes:

* ``spawn`` start method is required. With ``fork`` the child inherits the
  parent's already-open SQLite connection and the write lock is not reliably
  visible across processes.
* The parent must construct the ``FanoutCache`` *before* the holder acquires
  the lock. ``Cache.__init__`` writes to the Settings table and retries
  "database is locked" for 60 seconds; a lock held during construction turns
  every test into a 60-second stall.
* Keys are chosen by inspecting ``FanoutCache._hash`` so that exactly one key
  lands on the locked shard.

"""

import multiprocessing as mp
import os.path as op
import shutil
import sqlite3
import threading
import time

import pytest

import diskcache as dc

SHARDS = 4
LOCKED = 1
TIMEOUT = 0.05
HOLD_SECONDS = 600

pytestmark = pytest.mark.skipif(
    'fork' not in mp.get_all_start_methods(),
    reason='spawn start method required for cross-process lock tests',
)


def _hold_lock(shard_dir, ready, go):
    """Hold a write transaction on a single shard until released."""
    cache = dc.Cache(shard_dir, timeout=TIMEOUT)
    with cache.transact():
        ready.set()
        go.wait(HOLD_SECONDS)
    cache.close()


def _key_for_shard(cache, shard, start=0):
    key = start
    while cache._hash(key) % cache._count != shard:
        key += 1
    return key


def _call_with_bound(func, bound=1.0):
    """Run `func` in a daemon thread and report whether it finished."""
    result = {}

    def worker():
        try:
            result['value'] = func()
        except BaseException as error:  # noqa: BLE001 - report any failure
            result['error'] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(bound)
    return (not thread.is_alive()), result


class LockedShard:
    """FanoutCache with one shard write-locked by another process."""

    def __init__(self, directory):
        mp.set_start_method('spawn', force=True)
        self.directory = directory
        self.cache = dc.FanoutCache(
            directory, shards=SHARDS, timeout=TIMEOUT
        )
        self.keys = [_key_for_shard(self.cache, num, num * 1000)
                     for num in range(SHARDS)]
        self.locked_key = self.keys[LOCKED]
        self.healthy_key = self.keys[0]
        for num, key in enumerate(self.keys):
            assert self.cache.set(key, 'value-%d' % num, tag='color')
        self.ready = mp.Event()
        self.go = mp.Event()
        shard_dir = op.join(directory, '%03d' % LOCKED)
        self.process = mp.Process(
            target=_hold_lock, args=(shard_dir, self.ready, self.go)
        )
        self.process.start()
        assert self.ready.wait(10), 'holder did not acquire lock'

    def release(self):
        self.go.set()
        self.process.join(10)

    def stop(self):
        self.release()
        self.cache.close()
        shutil.rmtree(self.directory, ignore_errors=True)


@pytest.fixture
def locked(tmp_path):
    holder = LockedShard(str(tmp_path))
    try:
        yield holder
    finally:
        holder.stop()


@pytest.fixture
def stat_locked(tmp_path):
    mp.set_start_method('spawn', force=True)
    directory = str(tmp_path)
    cache = dc.FanoutCache(
        directory, shards=SHARDS, timeout=TIMEOUT, statistics=True
    )
    keys = [_key_for_shard(cache, num, num * 1000) for num in range(SHARDS)]
    for num, key in enumerate(keys):
        assert cache.set(key, 'value-%d' % num)
    ready = mp.Event()
    go = mp.Event()
    shard_dir = op.join(directory, '%03d' % LOCKED)
    process = mp.Process(target=_hold_lock, args=(shard_dir, ready, go))
    process.start()
    assert ready.wait(10), 'holder did not acquire lock'
    try:
        yield cache, keys
    finally:
        go.set()
        process.join(10)
        cache.close()
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Single-key operations on the write-locked shard.
# ---------------------------------------------------------------------------


def test_set_returns_false_without_writing(locked):
    cache = locked.cache
    key = locked.locked_key

    assert cache.set(key, 'overwritten') is False

    rows = cache._shards[LOCKED]._sql(
        'SELECT value FROM Cache'
    ).fetchall()
    assert rows == [('value-%d' % LOCKED,)]


def test_set_retry_true_blocks_until_release(locked):
    cache = locked.cache
    key = locked.locked_key

    finished, _ = _call_with_bound(lambda: cache.set(key, 'x', retry=True))
    assert finished is False

    locked.release()
    finished, result = _call_with_bound(lambda: cache.set(key, 'x', retry=True))
    assert finished is True and result['value'] is True


def test_setitem_blocks_until_release(locked):
    cache = locked.cache
    key = locked.locked_key

    finished, _ = _call_with_bound(lambda: cache.__setitem__(key, 'x'))
    assert finished is False

    locked.release()
    finished, result = _call_with_bound(lambda: cache.__setitem__(key, 'x'))
    assert finished is True
    assert 'error' not in result


def test_get_fast_path_returns_real_value(locked):
    cache = locked.cache
    assert cache.get(locked.locked_key, default='DEF') == 'value-%d' % LOCKED


def test_get_slow_path_returns_default_and_masks_failure(stat_locked):
    cache, keys = stat_locked
    key = keys[LOCKED]

    # Statistics on forces a write transaction inside Cache.get; the Timeout
    # is swallowed and the caller cannot distinguish this from a missing key.
    assert cache.get(key, default='MISSING') == 'MISSING'

    # A genuinely missing key on a healthy shard returns the identical value.
    missing = _key_for_shard(cache, 0, 900000)
    assert missing not in cache
    assert cache.get(missing, default='MISSING') == 'MISSING'


def test_get_timeout_bypasses_default_tuple_wrapping(stat_locked):
    # Cache.get wraps the default as (default, None) when tag/expire_time is
    # requested, but FanoutCache.get returns the raw default because the
    # Timeout is caught before that wrapping happens. This documents the
    # divergence; callers unpacking tuples break under partial failure.
    cache, keys = stat_locked
    key = keys[LOCKED]
    assert cache.get(key, default='MISSING', tag=True) == 'MISSING'
    assert cache.get(key, default='MISSING', expire_time=True) == 'MISSING'
    assert cache.get(
        key, default='MISSING', expire_time=True, tag=True
    ) == 'MISSING'


def test_contains_and_iteration_survive_lock(locked):
    cache = locked.cache
    assert locked.locked_key in cache
    assert set(cache) == set(locked.keys)
    assert len(cache) == SHARDS


def test_delete_returns_false_and_keeps_key(locked):
    cache = locked.cache
    key = locked.locked_key

    assert cache.delete(key) is False
    assert key in cache


def test_delete_retry_true_completes_after_release(locked):
    cache = locked.cache
    key = locked.locked_key

    thread_result = {}
    thread = threading.Thread(
        target=lambda: thread_result.update(value=cache.delete(key, retry=True)),
        daemon=True,
    )
    thread.start()
    thread.join(1.0)
    assert thread.is_alive()

    locked.release()
    thread.join(5)
    assert not thread.is_alive()
    assert thread_result['value'] is True
    assert key not in cache


def test_delitem_blocks_until_release(locked):
    cache = locked.cache
    key = locked.locked_key

    thread = threading.Thread(target=lambda: cache.__delitem__(key), daemon=True)
    thread.start()
    thread.join(1.0)
    assert thread.is_alive()
    assert key in cache

    locked.release()
    thread.join(5)
    assert not thread.is_alive()
    assert key not in cache


def test_touch_add_return_false(locked):
    cache = locked.cache
    key = locked.locked_key
    fresh = _key_for_shard(cache, LOCKED, 500000)

    assert cache.touch(key, expire=60) is False
    assert cache.add(fresh, 'x') is False
    assert fresh not in cache


def test_incr_decr_return_none_without_partial_write(locked):
    cache = locked.cache
    fresh = _key_for_shard(cache, LOCKED, 600000)

    assert cache.incr(fresh, default=5) is None
    assert cache.decr(fresh, default=5) is None
    assert fresh not in cache


def test_pop_returns_default_and_masks_failure(locked):
    cache = locked.cache
    key = locked.locked_key

    # The key exists; the default is returned only because the shard is
    # locked. The caller cannot tell this apart from a missing key.
    assert cache.pop(key, default='DEF') == 'DEF'
    assert key in cache


def test_healthy_shard_unaffected(locked):
    cache = locked.cache
    key = locked.healthy_key

    assert cache.set(key, 'new') is True
    assert cache.get(key) == 'new'
    assert cache.delete(key) is True
    assert key not in cache


# ---------------------------------------------------------------------------
# Cross-shard operations: partial completion and retry loops.
# ---------------------------------------------------------------------------


def test_evict_partially_completes_and_returns_count(locked):
    # `retry=False` (the default) counts what each shard removed before its
    # Timeout and moves on, matching the documented "fails silently" contract.
    assert locked.cache.evict('color') == SHARDS - 1

    for num in range(SHARDS):
        count = locked.cache._shards[num]._sql(
            'SELECT count(*) FROM Cache'
        ).fetchone()[0]
        assert count == (1 if num == LOCKED else 0)


def test_clear_partially_completes_and_returns_count(locked):
    assert locked.cache.clear() == SHARDS - 1
    assert len(locked.cache) == 1
    assert locked.locked_key in locked.cache


def test_remove_hangs_when_retry_true(locked):
    # `retry=True` keeps retrying the locked shard; partial counts from the
    # preceding shards are held internally and never returned.
    finished, _ = _call_with_bound(
        lambda: locked.cache.evict('color', retry=True), bound=1.5
    )
    assert finished is False


def test_remove_retry_true_resumes_and_reports_total_after_release(locked):
    cache = locked.cache

    result = {}
    thread = threading.Thread(
        target=lambda: result.update(value=cache.evict('color', retry=True)),
        daemon=True,
    )
    thread.start()
    thread.join(1.0)
    assert thread.is_alive()

    locked.release()
    thread.join(10)
    assert not thread.is_alive()
    # Every shard completed individually; the returned count is the sum.
    assert result['value'] == SHARDS
    assert len(cache) == 0


def test_check_raises_timeout_on_locked_shard(locked):
    with pytest.raises(dc.Timeout):
        locked.cache.check()


def test_check_fix_raises_operational_error(locked):
    with pytest.raises(sqlite3.OperationalError):
        locked.cache.check(fix=True)


def test_stats_blocks_then_raises_operational_error(locked):
    # Cache.reset retries "database is locked" for 60 seconds via
    # _sql_retry and then raises the raw sqlite3 error, not Timeout. The
    # FanoutCache wrapper does not catch it.
    from unittest import mock

    finished, _ = _call_with_bound(locked.cache.stats, bound=1.0)
    assert finished is False

    # Deterministically verify the exception type after the retry loop by
    # shortening the 60-second budget to zero.
    def immediate_retry(self):  # type: ignore[no-untyped-def]
        def execute(statement, *args, **kwargs):
            raise sqlite3.OperationalError('database is locked')

        return execute

    with mock.patch.object(dc.Cache, '_sql_retry', property(immediate_retry)):
        with pytest.raises(sqlite3.OperationalError):
            locked.cache.stats()


def test_reset_write_hangs_on_locked_shard(locked):
    finished, _ = _call_with_bound(
        lambda: locked.cache.reset('cull_limit', 5), bound=1.0
    )
    assert finished is False


def test_tag_index_ddl_raises_operational_error(locked):
    with pytest.raises(sqlite3.OperationalError):
        locked.cache.create_tag_index()


def test_read_only_meta_apis_still_answer(locked):
    cache = locked.cache
    assert isinstance(cache.volume(), int)
    assert len(cache) == SHARDS
    assert len(list(iter(cache))) == SHARDS
    assert len(list(reversed(cache))) == SHARDS


# ---------------------------------------------------------------------------
# Transaction boundaries.
# ---------------------------------------------------------------------------


def test_transact_blocks_and_poisons_preceding_shards(locked):
    cache = locked.cache
    entered = threading.Event()

    def worker():
        with cache.transact():
            entered.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(1.0)
    assert not entered.is_set()

    # Shard 0's write lock was acquired before the blocked shard 1.
    with pytest.raises(dc.Timeout):
        cache._shards[0].set('probe', 1, retry=False)
    # Later shards were never touched.
    assert cache._shards[2].set('probe', 1, retry=False) is True

    locked.release()
    thread.join(5)
    assert entered.is_set()


def test_transact_nested_retry_false_rejected(locked):
    with pytest.raises(AssertionError):
        with locked.cache.transact(retry=False):
            pass

