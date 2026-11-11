"""Test multi-tag associations in diskcache."""

import os
import os.path as op
import shutil
import sqlite3
import threading
import time

import pytest

import diskcache as dc


class _Picklable:
    def __init__(self, marker):
        self.marker = marker

    def __eq__(self, other):
        return isinstance(other, _Picklable) and self.marker == other.marker


@pytest.fixture
def cache(tmp_path):
    cache_obj = dc.Cache(str(tmp_path / 'cache'))
    yield cache_obj
    cache_obj.close()
    shutil.rmtree(cache_obj.directory, ignore_errors=True)


def _assert_no_orphans(cache_obj):
    rows = cache_obj._con.execute(
        'SELECT COUNT(*) FROM CacheTag AS t'
        ' LEFT JOIN Cache AS c ON c.rowid = t.rowid'
        ' WHERE c.rowid IS NULL'
    ).fetchone()
    assert rows[0] == 0


def test_get_tags_and_legacy_tag(cache):
    assert cache.set('k', 1, tag='primary', tags=['source', 'batch'])
    assert cache.get_tags('k') == ['primary', 'batch', 'source']
    assert cache.get('k', tag=True) == (1, 'primary')
    assert cache.get_tags('missing') is None
    assert cache.get_tags('missing', default=[]) == []


def test_query_union_and_intersection(cache):
    cache.set('k1', 1, tags=['source-a', 'batch-1', 'schema-v2'])
    cache.set('k2', 2, tags=['source-a', 'batch-2'])
    cache.set('k3', 3, tags=['source-b', 'batch-1'])
    cache.set('k4', 4, tags=['source-a', 'batch-1'])

    assert cache.query_tags(['source-a']) == ['k1', 'k2', 'k4']
    assert cache.query_tags(['source-a', 'source-b']) == ['k1', 'k2', 'k3',
                                                         'k4']
    assert cache.query_tags(['source-a', 'batch-1'], match='all') == [
        'k1',
        'k4',
    ]
    assert cache.query_tags(['source-a', 'batch-2'], match='all') == ['k2']
    assert cache.query_tags(['source-c'], match='all') == []
    assert cache.query_tags(['source-a'], mode='union') == ['k1', 'k2', 'k4']
    assert cache.query_tags(['source-a', 'batch-1'], mode='intersection') == [
        'k1',
        'k4',
    ]


def test_empty_and_duplicate_tags(cache):
    cache.set('k', 1, tags=['a', 'a', 'b', 'b', 1, 1.0])

    assert cache.query_tags([]) == []
    assert cache.query_tags([], match='all') == []
    assert cache.evict_tags([]) == 0
    assert cache.evict_tags([], match='all') == 0

    # 1 and 1.0 compare equal in SQLite and are stored once.
    assert cache.get_tags('k') == [1, 'a', 'b']
    assert cache.query_tags(['a', 'a', 'b'], match='all') == ['k']
    assert cache.query_tags(['a', 1.0], match='all') == ['k']


@pytest.mark.parametrize('match', ['any', 'all'])
def test_expired_items(cache, match):
    # Disable lazy culling so the expired row survives until the explicit
    # tag eviction exercises the no-expiry-fallback deletion path.
    cache.reset('cull_limit', 0)
    cache.set('expired', 1, expire=1e-9, tags=['g1', 'g2'])
    cache.set('live', 2, tags=['g1'])
    time.sleep(0.01)

    assert cache.query_tags(['g1'], match=match) == ['live']
    assert cache.query_tags(['g2'], match=match) == []

    # Eviction removes expired associations as well.
    assert cache.evict_tags(['g2'], match=match) == 1
    assert len(cache) == 1
    assert cache.check() == []
    _assert_no_orphans(cache)


def test_overwrite_replaces_tags(cache):
    cache.set('k', 'v1', tag='old', tags=['a', 'b'])
    cache.set('k', 'v2', tags=['c'])

    assert cache.get_tags('k') == ['c']
    assert cache.query_tags(['a']) == []
    assert cache.query_tags(['old']) == []
    assert cache.query_tags(['c']) == ['k']
    assert cache.get('k', tag=True) == ('v2', None)

    rows = cache._con.execute('SELECT tag FROM CacheTag').fetchall()
    assert rows == [('c',)]


def test_add_and_push_tags(cache):
    assert cache.add('added', 1, tags=['a'])
    assert not cache.add('added', 2, tags=['b'])
    assert cache.get_tags('added') == ['a']

    key = cache.push('value', tags=['queue'])
    assert cache.get_tags(key) == ['queue']

    key2 = cache.push('value2', tag='primary', tags=['queue', 'primary'])
    assert cache.get_tags(key2) == ['primary', 'queue']


def test_legacy_evict_still_works(cache):
    for value in range(90):
        cache.set(value, value, tag=('red', 'blue', 'yellow')[value % 3])

    assert cache.evict('red') == 30
    assert len(cache) == 60
    assert cache.evict_tags(['red']) == 0
    assert cache.check() == []
    _assert_no_orphans(cache)


def test_evict_dispatch(cache):
    cache.set('a', 1, tags=['x', 'y'])
    cache.set('b', 2, tags=['x'])
    cache.set('c', 3, tags=['y'])

    assert cache.evict(tags=['x', 'y'], match='all') == 1
    assert 'a' not in cache
    assert cache.evict(tags=['y'], match='any') == 1
    assert 'c' not in cache
    assert len(cache) == 1


def test_invalid_match(cache):
    cache.set('k', 1, tags=['a'])
    with pytest.raises(ValueError):
        cache.query_tags(['a'], match='bad')
    with pytest.raises(ValueError):
        cache.evict_tags(['a'], match='bad')


def test_evict_mode_alias(cache):
    cache.set('a', 1, tags=['x', 'y'])
    cache.set('b', 2, tags=['x'])

    assert cache.evict(tags=['x', 'y'], mode='and') == 1
    assert cache.evict_tags(['x'], mode='union') == 1
    assert len(cache) == 0


def test_limit(cache):
    for num in range(10):
        cache.set(num, num, tags=['all'])

    assert cache.query_tags(['all'], limit=3) == [0, 1, 2]
    assert cache.query_tags(['all'], limit=0) == []
    with pytest.raises(ValueError):
        cache.query_tags(['all'], limit=-1)


def test_delete_paths_remove_associations(cache):
    cache.set('del', 1, tags=['x'])
    cache.delete('del')

    cache.set('pop', 1, tags=['x'])
    del cache['pop']

    cache.push('queue', tags=['x'])
    cache.pull()

    cache.push('queue2', tags=['x'])
    cache.peek()

    cache.set('peekitem', 1, tags=['x'])
    cache.peekitem()

    cache.set('expire', 1, expire=1e-9, tags=['x'])
    time.sleep(0.01)
    cache.expire()

    cache.set('evict', 1, tags=['x'])
    cache.evict_tags(['x'])

    cache.set('legacy', 1, tag='x')
    cache.evict('x')

    cache.set('clear', 1, tags=['x'])
    cache.clear()

    rows = cache._con.execute('SELECT COUNT(*) FROM CacheTag').fetchone()
    assert rows[0] == 0
    assert cache.check() == []


def test_evict_does_not_deserialize_values(cache, monkeypatch):
    cache.set('k1', _Picklable(1), tags=['bulk'])
    cache.set('k2', _Picklable(2), tags=['bulk', 'other'])

    def fail_fetch(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError('values must not be deserialized on eviction')

    monkeypatch.setattr(cache._disk, 'fetch', fail_fetch)

    assert cache.evict_tags(['bulk']) == 2
    assert cache._con.execute('SELECT COUNT(*) FROM Cache'
                             ).fetchone()[0] == 0
    _assert_no_orphans(cache)


def test_large_file_cleanup(cache):
    cache.reset('cull_limit', 0)
    cache.reset('disk_min_file_size', 1024)
    big = b'x' * 4096

    cache.set('big1', big, tags=['bulk', 's1'])
    cache.set('big2', big, tags=['bulk', 's2'])

    def value_files():
        paths = []
        for dirpath, _, filenames in os.walk(cache.directory):
            paths.extend(
                op.join(dirpath, name)
                for name in filenames
                if name.endswith('.val')
            )
        return paths

    assert len(value_files()) == 2
    assert cache.evict_tags(['bulk']) == 2
    assert value_files() == []
    assert cache._con.execute('SELECT COUNT(*) FROM CacheTag'
                             ).fetchone()[0] == 0
    assert cache.check() == []
    _assert_no_orphans(cache)


def test_reopen_preserves_associations(tmp_path):
    directory = str(tmp_path / 'cache')
    cache = dc.Cache(directory)
    cache.set('k1', 1, tag='primary', tags=['source', 'batch'])
    cache.set('k2', 2, tags=['source'])
    cache.close()

    cache = dc.Cache(directory)
    assert cache.get_tags('k1') == ['primary', 'batch', 'source']
    assert cache.query_tags(['source']) == ['k1', 'k2']
    assert cache.query_tags(['source', 'batch'], match='all') == ['k1']
    assert cache.check() == []
    cache.close()


def test_upgrade_from_legacy_directory(tmp_path):
    directory = str(tmp_path / 'cache')
    cache = dc.Cache(directory)

    for num in range(10):
        cache.set(num, num, tag='even' if num % 2 == 0 else 'odd')

    cache.close()

    # Emulate a cache directory created before CacheTag existed.
    con = sqlite3.connect(op.join(directory, 'cache.db'))
    con.execute('DROP TRIGGER IF EXISTS CacheTag_cache_delete')
    con.execute('DROP INDEX IF EXISTS CacheTag_tag_rowid')
    con.execute('DROP TABLE IF EXISTS CacheTag')
    con.execute('DELETE FROM Settings WHERE key = ?',
                (dc.core.TAG_SCHEMA_SETTING,))
    con.commit()
    con.close()

    cache = dc.Cache(directory)
    assert cache.query_tags(['even']) == [0, 2, 4, 6, 8]
    assert cache.query_tags(['odd'], match='all') == [1, 3, 5, 7, 9]
    assert cache.get_tags(0) == ['even']
    assert cache._con.execute('SELECT COUNT(*) FROM CacheTag'
                             ).fetchone()[0] == 10

    # Reopening must not duplicate associations or rerun the migration.
    cache.close()
    cache = dc.Cache(directory)
    assert cache._con.execute('SELECT COUNT(*) FROM CacheTag'
                             ).fetchone()[0] == 10
    assert cache.check() == []
    cache.close()


def test_concurrent_overwrite(cache):
    def worker(thread_id):
        for num in range(50):
            cache.set(
                'shared',
                (thread_id, num),
                tags=['thread-%d' % thread_id],
                retry=True,
            )

    threads = [
        threading.Thread(target=worker, args=(thread_id,))
        for thread_id in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tags = cache.get_tags('shared')
    assert len(tags) == 1
    assert tags[0].startswith('thread-')
    assert cache.check() == []
    _assert_no_orphans(cache)


def test_check_detects_and_fixes_orphans(cache):
    cache.set('k', 1, tags=['a'])
    rowid = cache._con.execute('SELECT rowid FROM Cache').fetchone()[0]
    cache._con.execute('INSERT INTO CacheTag (rowid, tag) VALUES (?, ?)',
                       (rowid + 999, 'orphan'))

    warnings = cache.check()
    assert any('orphan tag association' in str(warning.message)
               for warning in warnings)

    warnings = cache.check(fix=True)
    assert any('orphan tag association' in str(warning.message)
               for warning in warnings)
    _assert_no_orphans(cache)


@pytest.fixture
def fanout(tmp_path):
    cache_obj = dc.FanoutCache(str(tmp_path / 'fanout'), shards=4)
    yield cache_obj
    cache_obj.close()
    shutil.rmtree(cache_obj.directory, ignore_errors=True)


def test_fanout_union_and_intersection(fanout):
    for num in range(60):
        fanout.set(num, num, tags=['source-%d' % (num % 3),
                                   'batch-%d' % (num % 5)])

    expected_any = [num for num in range(60) if num % 3 == 0]
    expected_all = [
        num for num in range(60) if num % 3 == 0 and num % 5 == 0
    ]

    assert sorted(fanout.query_tags(['source-0'])) == expected_any
    assert sorted(fanout.query_tags(['source-0', 'batch-0'], match='all')) \
        == expected_all
    assert fanout.query_tags(['missing']) == []
    assert fanout.evict_tags([]) == 0


def test_fanout_evict_result_boundaries(fanout):
    for num in range(60):
        fanout.set(num, num, tags=['source-%d' % (num % 3)])

    result = fanout.evict_tags(['source-0', 'source-1'])
    assert isinstance(result, dc.TagEvictResult)
    assert result == 40
    assert int(result) == 40
    assert result.total == 40
    assert sum(result.shards) == 40
    assert len(result.shards) == 4
    assert result.atomic is False
    assert result.shard_atomic is True
    assert result.completed_shards == 4
    assert result.failed_shards == ()
    assert len(fanout) == 20
    assert fanout.check() == []


def test_fanout_atomic_transaction(fanout):
    for num in range(20):
        fanout.set(num, num, tags=['source'])

    result = fanout.evict_tags(['source'], transact=True)
    assert result == 20
    assert result.atomic is True
    assert result.shard_atomic is True
    assert result.completed_shards == 4
    assert len(fanout) == 0


def test_fanout_legacy_evict(fanout):
    for num in range(30):
        fanout.set(num, num, tag=('red', 'blue', 'green')[num % 3])

    assert fanout.evict('red') == 10
    assert len(fanout) == 20
    assert fanout.evict_tags(['red']) == 0
    assert fanout.check() == []


def test_fanout_get_tags_and_overwrite(fanout):
    fanout.set('k', 1, tag='primary', tags=['a', 'b'])
    assert set(fanout.get_tags('k')) == {'primary', 'a', 'b'}
    fanout.set('k', 2, tags=['c'])
    assert fanout.get_tags('k') == ['c']
    assert fanout.query_tags(['a']) == []
    assert fanout.query_tags(['c']) == ['k']
    assert fanout.get_tags('missing') is None
