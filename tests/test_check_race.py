"""Regression tests for Cache.check racing concurrent value-file writes.

Cache.set writes the value file before publishing the database row, so
Cache.check(fix=True) can observe files that are not yet referenced by any
row. These tests pin down that check must not delete files still inside
the publish window while continuing to reclaim genuine leftovers.
"""

import multiprocessing as mp
import os
import os.path as op
import shutil
import tempfile
from unittest import mock

import pytest

import diskcache as dc
from diskcache.core import TEMP_SUFFIX

pytestmark = pytest.mark.filterwarnings('ignore', category=dc.EmptyDirWarning)

BLOB = b'x' * 2**20  # Larger than disk_min_file_size; stored as a file.


def value_files(directory):
    """List non-database files in the cache directory."""
    result = []
    for dirpath, _, files in os.walk(directory):
        for name in files:
            full_path = op.join(dirpath, name)
            if 'cache.db' not in full_path:
                result.append(full_path)
    return result


def _paused_writer(directory, written, proceed):
    """Write a value file, then pause before publishing the row."""
    cache = dc.Cache(directory)
    store = cache._disk._store

    def paused_store(value, read, key=dc.UNKNOWN):
        # The real store has written and renamed the value file and holds
        # its lock, but the row is not yet committed: the publish window.
        result = store(value, read, key=key)
        written.set()
        proceed.wait(30)
        return result

    cache._disk._store = paused_store
    assert cache.set('key', BLOB)
    cache.close()


def _crashed_writer(directory, started):
    """Simulate a writer killed while its temp file is locked."""
    cache = dc.Cache(directory)
    full_path = op.join(directory, 'aa', 'bb', 'crash.val')
    os.makedirs(op.dirname(full_path), exist_ok=True)
    temp_path = full_path + TEMP_SUFFIX
    writer = open(temp_path, 'xb')
    assert dc.core._try_lock(writer)
    writer.write(b'partial')
    writer.flush()
    started.set()
    # Exit without closing: the kernel releases the lock, leaving an
    # orphaned temp file behind.
    os._exit(1)


def test_check_preserves_concurrent_value():
    """check(fix=True) must not delete a value file mid-publish."""
    directory = tempfile.mkdtemp()
    try:
        context = mp.get_context('spawn')
        written = context.Event()
        proceed = context.Event()
        writer = context.Process(
            target=_paused_writer, args=(directory, written, proceed)
        )
        writer.start()
        assert written.wait(30)

        cache = dc.Cache(directory)
        in_flight = [
            path for path in value_files(directory) if path.endswith('.val')
        ]
        assert len(in_flight) == 1
        assert writer.is_alive()

        cache.check(fix=True)

        # The checker saw an unknown file but the writer still holds its
        # lock, so the file must survive the fixing run.
        assert op.exists(in_flight[0])
        cache.close()

        proceed.set()
        writer.join(30)
        assert writer.exitcode == 0

        cache = dc.Cache(directory)
        assert cache['key'] == BLOB
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_check_reclaims_orphans():
    """check(fix=True) still removes genuine leftover files."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        cache['keep'] = BLOB
        keep = set(value_files(directory))

        # Orphaned final-named file (e.g. from an old cache directory).
        orphan_final = op.join(directory, 'aa', 'bb', 'orphan.val')
        os.makedirs(op.dirname(orphan_final), exist_ok=True)
        with open(orphan_final, 'wb') as writer:
            writer.write(b'orphan')

        # Orphaned temp file (crashed writer, failed overwrite, rollback).
        orphan_temp = op.join(
            directory, 'cc', 'dd', 'orphan.val' + TEMP_SUFFIX
        )
        os.makedirs(op.dirname(orphan_temp), exist_ok=True)
        with open(orphan_temp, 'wb') as writer:
            writer.write(b'orphan')

        # Unknown file with arbitrary name (legacy cache contents).
        legacy = op.join(directory, 'legacy.bin')
        with open(legacy, 'wb') as writer:
            writer.write(b'legacy')

        cache.check(fix=True)
        assert set(value_files(directory)) == keep
        assert cache['keep'] == BLOB

        # Empty directories are removed one level per check; repeat until
        # the cache is clean.
        for _ in range(3):
            if not cache.check(fix=True):
                break
        assert cache.check() == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_check_skips_locked_unknown_file():
    """An unknown file locked by a live process is never reclaimed."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        locked = op.join(directory, 'aa', 'bb', 'locked.val')
        os.makedirs(op.dirname(locked), exist_ok=True)
        writer = open(locked, 'wb')
        writer.write(b'locked')
        writer.flush()
        assert dc.core._try_lock(writer)

        cache.check(fix=True)
        assert op.exists(locked)

        writer.close()
        cache.check(fix=True)
        assert not op.exists(locked)
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_check_reclaims_crash_leftover():
    """A crashed writer's temp file is reclaimed by check(fix=True)."""
    directory = tempfile.mkdtemp()
    try:
        context = mp.get_context('spawn')
        started = context.Event()
        crasher = context.Process(
            target=_crashed_writer, args=(directory, started)
        )
        crasher.start()
        assert started.wait(30)
        crasher.join(30)
        assert crasher.exitcode == 1

        cache = dc.Cache(directory)
        leftovers = value_files(directory)
        assert any(path.endswith(TEMP_SUFFIX) for path in leftovers)
        cache.check(fix=True)
        assert value_files(directory) == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_check_interrupted_cache_usable():
    """A check aborted mid-transaction leaves the cache fully usable."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        cache['before'] = BLOB

        real_walk = os.walk
        calls = []

        def flaky_walk(*args, **kwargs):
            if calls:
                raise RuntimeError('check interrupted')
            calls.append(True)
            return real_walk(*args, **kwargs)


        with mock.patch('diskcache.core.os.walk', flaky_walk):
            with pytest.raises(RuntimeError):
                cache.check(fix=True)

        # The aborted check rolled its transaction back; reads and writes
        # continue to work and a fresh check is clean.
        assert cache['before'] == BLOB
        cache['after'] = BLOB
        assert cache['after'] == BLOB
        assert cache.check() == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_rollback_removes_new_file():
    """A rolled-back transaction leaves no value file behind."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        with pytest.raises(RuntimeError):
            with cache.transact():
                cache['txn-key'] = BLOB
                raise RuntimeError('boom')
        assert value_files(directory) == []
        cache['after'] = BLOB
        assert cache['after'] == BLOB
        assert cache.check() == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_timeout_removes_new_file():
    """A timed-out publish leaves no value file behind."""
    directory = tempfile.mkdtemp()
    try:
        blocker = dc.Cache(directory)
        cache = dc.Cache(directory, timeout=0.01)
        with blocker.transact():
            with pytest.raises(dc.Timeout):
                cache.set('blocked', BLOB, retry=False)
        assert value_files(directory) == []
        cache.set('ok', BLOB, retry=True)
        assert cache['ok'] == BLOB
        assert cache.check() == []
        cache.close()
        blocker.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_failed_write_leaves_no_leftover():
    """Repeated open failures while writing must not leave a temp file."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        with mock.patch('diskcache.core.open', side_effect=OSError):
            with pytest.raises(OSError):
                cache['big'] = BLOB
        assert value_files(directory) == []
        cache['retry-ok'] = BLOB
        assert cache['retry-ok'] == BLOB
        assert cache.check() == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_check_without_lock_support_is_conservative():
    """Without an advisory-lock primitive check must never reclaim files."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        cache['keep'] = BLOB
        orphan = op.join(directory, 'aa', 'orphan.val')
        os.makedirs(op.dirname(orphan), exist_ok=True)
        with open(orphan, 'wb') as writer:
            writer.write(b'x')

        supported = dc.core.LOCK_SUPPORTED
        dc.core.LOCK_SUPPORTED = False
        try:
            cache.check(fix=True)
        finally:
            dc.core.LOCK_SUPPORTED = supported

        # No lock primitive means the orphan cannot be proven abandoned, so
        # the checker leaves it in place rather than risk a live value.
        assert op.exists(orphan)
        assert cache['keep'] == BLOB
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_nested_transaction_files():
    """Nested writes survive commit and are cleaned up on rollback."""
    directory = tempfile.mkdtemp()
    try:
        cache = dc.Cache(directory)
        with cache.transact():
            cache['a'] = BLOB
            cache['b'] = BLOB + b'b'
        assert cache['a'] == BLOB and cache['b'] == BLOB + b'b'
        assert cache.check() == []

        with pytest.raises(RuntimeError):
            with cache.transact():
                cache['c'] = BLOB
                cache['d'] = BLOB + b'd'
                raise RuntimeError('rollback')
        assert cache.get('c') is None and cache.get('d') is None
        assert value_files(directory) and cache.check() == []
        cache.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)
