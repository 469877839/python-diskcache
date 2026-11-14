"""Test diskcache multi-tag associations."""

import os
import os.path as op
import shutil
import sqlite3
import threading
import time

import pytest

import diskcache as dc

pytestmark = pytest.mark.filterwarnings('ignore', category=dc.EmptyDirWarning)


@pytest.fixture
def cache():
    with dc.Cache() as cache:
        yield cache
    shutil.rmtree(cache.directory, ignore_errors=True)


def tag_rows(cache):
    sql = cache._sql
    return sql('SELECT rowid, tag FROM Tag ORDER BY rowid, tag').fetchall()


def test_set_get_multi_tags(cache):
    assert cache.set('a', 1, tags=['src:ods', 'batch:1', 'schema:v1'])
    assert cache.set('b', 2, tags=['src:ods', 'batch:2'])
    assert cache.set('c', 3, tags=['src:api'])

    assert sorted(cache.keys_by_tags(['src:ods'])) == ['a', 'b']
    assert sorted(cache.keys_by_tags(['batch:1', 'src:api'])) == ['a', 'c']
    assert cache.keys_by_tags(['src:ods', 'batch:1'], all=True) == ['a']
    assert cache.keys_by_tags(['src:ods', 'src:api'], all=True) == []
    assert cache.keys_by_tags(['missing']) == []


def test_empty_and_duplicate_tags(cache):
    assert cache.set('a', 1, tags=[])
    assert tag_rows(cache) == []
    assert cache.keys_by_tags([]) == []
    assert cache.keys_by_tags((), all=True) == []
    assert cache.evict_by_tags([]) == 0
    assert cache.evict_by_tags((), all=True) == 0

    assert cache.set('b', 2, tags=['x', 'x', 'y', 'y'])
    assert sorted(tag for _, tag in tag_rows(cache)) == ['x', 'y']
    assert cache.keys_by_tags(['x', 'x']) == ['b']
    assert cache.keys_by_tags(['x', 'y'], all=True) == ['b']


def test_legacy_tag_compat(cache):
    # Legacy single tag keeps its meaning and is visible to multi-tag
    # queries and deletes.
    assert cache.set('a', 1, tag='legacy')
    assert cache.get('a', tag=True) == (1, 'legacy')
    assert cache.keys_by_tags(['legacy']) == ['a']

    # Items tagged only through the multi-tag interface are matched by
    # the legacy evict as well.
    assert cache.set('b', 2, tags=['legacy'])
    assert cache.evict('legacy') == 2
    assert len(cache) == 0
    assert tag_rows(cache) == []

    # Legacy tag combines with multi tags on one item.
    assert cache.set('c', 3, tag='single', tags=['multi'])
    assert cache.keys_by_tags(['single', 'multi'], all=True) == ['c']


def test_add_and_push_tags(cache):
    assert cache.add('a', 1, tags=['queued'])
    assert not cache.add('a', 2, tags=['other'])
    assert cache.keys_by_tags(['queued']) == ['a']
    assert cache.keys_by_tags(['other']) == []

    key = cache.push('value', tags=['queued'])
    assert set(cache.keys_by_tags(['queued'])) == {'a', key}


def test_expired_items(cache):
    assert cache.set('short', 1, expire=0.01, tags=['t'])
    assert cache.set('long', 2, tags=['t'])
    time.sleep(0.02)

    # Queries exclude expired items.
    assert cache.keys_by_tags(['t']) == ['long']

    # Deletes remove expired items too, consistent with Cache.evict.
    assert cache.evict_by_tags(['t']) == 2
    assert len(cache) == 0


def test_overwrite_replaces_tags(cache):
    assert cache.set('k', 1, tags=['old', 'shared'])
    assert cache.set('k', 2, tags=['new', 'shared'])

    assert cache.keys_by_tags(['old']) == []
    assert cache.keys_by_tags(['new']) == ['k']
    assert cache.keys_by_tags(['shared']) == ['k']
    assert sorted(tag for _, tag in tag_rows(cache)) == ['new', 'shared']

    # Overwrite with no tags clears all associations.
    assert cache.set('k', 3)
    assert tag_rows(cache) == []
    assert cache.keys_by_tags(['shared']) == []


def test_upgrade_existing_directory(tmp_path):
    directory = str(tmp_path / 'cache')

    # Build a pre-feature cache directory: no Tag table, legacy tags
    # stored only in the Cache.tag column.
    with dc.Cache(directory) as cache:
        for num in range(5):
            assert cache.set(num, num, tag='legacy-%d' % (num % 2))
        cache._sql('DROP TRIGGER Tag_delete')
        cache._sql('DROP TABLE Tag')

    db_path = op.join(directory, 'cache.db')
    with sqlite3.connect(db_path) as con:
        tables = con.execute(
            'SELECT name FROM sqlite_master WHERE type = "table"'
        ).fetchall()
        assert ('Tag',) not in tables

    # Re-opening upgrades the directory and migrates legacy tags.
    with dc.Cache(directory) as cache:
        assert sorted(cache.keys_by_tags(['legacy-0'])) == [0, 2, 4]
        assert sorted(cache.keys_by_tags(['legacy-1'])) == [1, 3]
        assert cache.evict_by_tags(['legacy-0']) == 3
        assert sorted(cache) == [1, 3]
        assert cache.check() == []

    shutil.rmtree(directory, ignore_errors=True)


def test_reopen_preserves_tags(tmp_path):
    directory = str(tmp_path / 'cache')

    with dc.Cache(directory) as cache:
        cache.set('a', 1, tags=['src:ods', 'batch:1'])
        cache.set('b', 2, tags=['src:ods'])

    with dc.Cache(directory) as cache:
        assert sorted(cache.keys_by_tags(['src:ods'])) == ['a', 'b']
        assert cache.keys_by_tags(['src:ods', 'batch:1'], all=True) == ['a']
        assert cache.evict_by_tags(['batch:1']) == 1
        assert cache.keys_by_tags(['src:ods']) == ['b']

    shutil.rmtree(directory, ignore_errors=True)


def test_concurrent_overwrite(tmp_path):
    directory = str(tmp_path / 'cache')
    barrier = threading.Barrier(8)

    with dc.Cache(directory) as cache:

        def writer(num):
            barrier.wait()
            for index in range(50):
                cache.set(
                    'key-%d' % (index % 10),
                    (num, index),
                    tags=['worker-%d' % num, 'round-%d' % (index % 5)],
                )

        threads = [
            threading.Thread(target=writer, args=(num,)) for num in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Every association must reference a live Cache row and the
        # associations of each row must be exactly one worker tag and
        # one round tag (never a mix left over from concurrent writes).
        assert cache.check() == []
        rows = tag_rows(cache)
        by_rowid = {}
        for rowid, tag in rows:
            by_rowid.setdefault(rowid, []).append(tag)
        for rowid, tags in by_rowid.items():
            assert len(tags) == 2
            workers = [tag for tag in tags if tag.startswith('worker-')]
            rounds = [tag for tag in tags if tag.startswith('round-')]
            assert len(workers) == 1
            assert len(rounds) == 1

    shutil.rmtree(directory, ignore_errors=True)


def test_fanout_any_all_and_evict(tmp_path):
    directory = str(tmp_path / 'fanout')
    cache = dc.FanoutCache(directory, shards=4)
    try:
        for num in range(20):
            tags = ['batch:%d' % (num % 4), 'src:ods']
            if num % 2:
                tags.append('schema:v2')
            assert cache.set(num, num, tags=tags)

        assert len(cache.keys_by_tags(['src:ods'])) == 20
        assert len(cache.keys_by_tags(['batch:0'])) == 5
        assert (
            len(cache.keys_by_tags(['src:ods', 'schema:v2'], all=True)) == 10
        )
        assert sorted(
            cache.keys_by_tags(['batch:0', 'schema:v2'], all=True)
        ) == [num for num in range(20) if num % 4 == 0 and num % 2]

        # Legacy single-tag interface still works on shards.
        assert cache.set('legacy', 0, tag='old')
        assert cache.keys_by_tags(['old']) == ['legacy']
        assert cache.evict('old') == 1

        assert cache.evict_by_tags(['schema:v2']) == 10
        assert len(cache.keys_by_tags(['src:ods'])) == 10
        assert cache.evict_by_tags(['src:ods', 'batch:2'], all=True) == 5
        assert cache.check() == []
    finally:
        cache.close()
        shutil.rmtree(directory, ignore_errors=True)


def test_large_value_delete_leaves_no_trace(tmp_path):
    directory = str(tmp_path / 'cache')
    big = b'x' * (2**16)  # Exceeds default disk_min_file_size (32kb).

    with dc.Cache(directory, disk_min_file_size=1024) as cache:
        for num in range(10):
            assert cache.set(num, big, tags=['bulk'])
        assert cache.set('keep', big, tags=['other'])

        filenames_before = {
            filename
            for (filename,) in cache._sql(
                'SELECT filename FROM Cache WHERE filename IS NOT NULL'
            )
        }
        assert len(filenames_before) == 11

        # Deleting by tag must not deserialize values.
        fetch_calls = []
        original_fetch = cache._disk.fetch

        def counting_fetch(mode, filename, value, read):
            fetch_calls.append(filename)
            return original_fetch(mode, filename, value, read)

        cache._disk.fetch = counting_fetch
        try:
            assert cache.evict_by_tags(['bulk']) == 10
        finally:
            cache._disk.fetch = original_fetch
        assert fetch_calls == []

        # No leftover association rows, no dangling file references,
        # and no orphaned files on disk.
        assert len(tag_rows(cache)) == 1
        assert tag_rows(cache)[0][1] == 'other'
        assert cache.check() == []

        kept = {
            filename
            for (filename,) in cache._sql(
                'SELECT filename FROM Cache WHERE filename IS NOT NULL'
            )
        }
        for dirpath, _, files in os.walk(directory):
            for name in files:
                if name.startswith('cache.db'):
                    continue
                relpath = op.relpath(op.join(dirpath, name), directory)
                assert relpath in kept

        assert cache.get('keep') == big

    shutil.rmtree(directory, ignore_errors=True)
