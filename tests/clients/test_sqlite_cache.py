"""Tests for SQLiteCache - serialization, core behavior, concurrency, migration."""

import os
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import orjson
import pydantic
import pytest

from dspy.clients.sqlite_cache import (
    SQLiteCache,
    _deserialize,
    _serialize,
    has_legacy_diskcache,
    migrate_diskcache,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


class PydanticModel(pydantic.BaseModel):
    name: str
    value: int
    tags: list[str] = []
    metadata: dict[str, str] = {}


class Outer(pydantic.BaseModel):
    class Inner(pydantic.BaseModel):
        value: int

    inner: Inner
    label: str


class NestedPydanticModel(pydantic.BaseModel):
    inner: PydanticModel
    count: int


@dataclass
class SimpleDataclass:
    x: int
    y: str


@dataclass
class NestedDataclass:
    label: str
    inner: SimpleDataclass


# ── Serialization ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "hello",
        42,
        3.14,
        True,
        False,
        None,
        [1, "two", 3.0, None, True],
        {"a": 1, "b": [2, 3], "c": {"nested": True}},
        {},
        [],
        {"big_int": 2**53, "neg": -999999, "zero": 0},
        {"emoji": "Hello World", "chinese": "你好世界", "mixed": "café résumé naïve"},
        {"key\nwith\nnewlines": "value\twith\ttabs", "quotes": 'he said "hello"'},
    ],
    ids=lambda v: repr(v)[:40],
)
def test_serialize_roundtrip_plain(value):
    assert _deserialize(_serialize(value)) == value


def test_serialize_roundtrip_pydantic():
    m = PydanticModel(name="test", value=42, tags=["a", "b"], metadata={"k": "v"})
    result = _deserialize(_serialize(m))
    assert isinstance(result, PydanticModel)
    assert result == m


def test_serialize_roundtrip_nested_pydantic():
    m = NestedPydanticModel(inner=PydanticModel(name="inner", value=1), count=5)
    result = _deserialize(_serialize(m))
    assert isinstance(result, NestedPydanticModel)
    assert isinstance(result.inner, PydanticModel)
    assert result.inner.name == "inner"


def test_serialize_roundtrip_nested_class():
    m = Outer(inner=Outer.Inner(value=42), label="test")
    result = _deserialize(_serialize(m))
    assert isinstance(result, Outer)
    assert isinstance(result.inner, Outer.Inner)
    assert result.inner.value == 42
    assert result.label == "test"


def test_serialize_roundtrip_dataclass():
    d = SimpleDataclass(x=10, y="hello")
    result = _deserialize(_serialize(d))
    assert isinstance(result, SimpleDataclass)
    assert result.x == 10 and result.y == "hello"


def test_serialize_roundtrip_nested_dataclass():
    d = NestedDataclass(label="outer", inner=SimpleDataclass(x=10, y="hello"))
    result = _deserialize(_serialize(d))
    assert isinstance(result, NestedDataclass)
    assert isinstance(result.inner, SimpleDataclass)
    assert result == d


def test_serialize_non_serializable_raises():
    class Custom:
        pass

    with pytest.raises(TypeError):
        _serialize(Custom())

    with pytest.raises(TypeError):
        _serialize({"obj": Custom()})


# ── Litellm ModelResponse ────────────────────────────────────────────────────


def test_model_response_roundtrip():
    from litellm import ModelResponse

    r = ModelResponse(
        id="chatcmpl-test123",
        choices=[{"message": {"content": "Hello world"}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )
    result = _deserialize(_serialize(r))
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello world"
    assert result.id == "chatcmpl-test123"
    result.usage = {}
    result.cache_hit = True
    assert result.cache_hit is True


def test_model_response_with_tool_calls_roundtrip():
    from litellm import ModelResponse

    r = ModelResponse(
        id="chatcmpl-tools",
        choices=[{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                }],
            },
            "index": 0,
            "finish_reason": "tool_calls",
        }],
    )
    result = _deserialize(_serialize(r))
    assert isinstance(result, ModelResponse)
    assert result.choices[0].finish_reason == "tool_calls"


def test_embedding_response_roundtrip():
    from litellm import EmbeddingResponse

    r = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        model="text-embedding-3-small",
        usage={"prompt_tokens": 5, "total_tokens": 5},
    )
    result = _deserialize(_serialize(r))
    assert isinstance(result, EmbeddingResponse)
    assert result.data[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3])


# ── SQLiteCache core behavior ────────────────────────────────────────────────


class TestSQLiteCacheBasic:
    def test_set_get_contains(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        assert "missing" not in cache
        cache["key1"] = "value1"
        assert "key1" in cache
        assert cache["key1"] == "value1"

    def test_key_error_on_missing(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        with pytest.raises(KeyError):
            _ = cache["nonexistent"]

    def test_overwrite_key(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache["key"] = "first"
        cache["key"] = "second"
        assert cache["key"] == "second"

    def test_is_empty(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        assert cache.is_empty()
        cache["key"] = "value"
        assert not cache.is_empty()

    def test_many_keys(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        for i in range(1000):
            cache[f"key_{i}"] = f"value_{i}"
        for i in range(1000):
            assert cache[f"key_{i}"] == f"value_{i}"

    def test_pydantic_model_stored_and_retrieved(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        m = PydanticModel(name="cached", value=99, tags=["x"])
        cache["model"] = m
        result = cache["model"]
        assert isinstance(result, PydanticModel)
        assert result.name == "cached"

    def test_non_serializable_value_raises_on_set(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)

        class Custom:
            pass

        with pytest.raises(TypeError):
            cache["bad"] = Custom()


# ── Persistence ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_data_survives_close_and_reopen(self, tmp_path):
        cache1 = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache1["persistent"] = {"data": [1, 2, 3]}
        cache1.close()

        cache2 = SQLiteCache(directory=str(tmp_path), size_limit=None)
        assert cache2["persistent"] == {"data": [1, 2, 3]}

    def test_db_file_created(self, tmp_path):
        SQLiteCache(directory=str(tmp_path), size_limit=None)
        assert (tmp_path / "dspy_cache.db").exists()

    def test_creates_nested_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        SQLiteCache(directory=str(nested), size_limit=None)
        assert (nested / "dspy_cache.db").exists()


# ── LRU Eviction ─────────────────────────────────────────────────────────────


class TestEviction:
    def test_eviction_removes_oldest(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=500)
        cache["old"] = "x" * 200
        time.sleep(0.01)
        cache["new"] = "y" * 200
        time.sleep(0.01)
        cache["newest"] = "z" * 200
        assert "old" not in cache
        assert "new" in cache
        assert "newest" in cache

    def test_eviction_respects_access_order(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=500)
        cache["a"] = "x" * 200
        time.sleep(0.01)
        cache["b"] = "y" * 200
        time.sleep(0.01)
        _ = cache["a"]
        time.sleep(0.01)
        cache["c"] = "z" * 200
        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_no_eviction_when_under_limit(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=100_000)
        for i in range(100):
            cache[f"key_{i}"] = f"val_{i}"
        for i in range(100):
            assert f"key_{i}" in cache

    @pytest.mark.parametrize("limit", [0, None])
    def test_no_eviction_with_disabled_limit(self, tmp_path, limit):
        cache = SQLiteCache(directory=str(tmp_path / str(limit)), size_limit=limit)
        for i in range(50):
            cache[f"key_{i}"] = "x" * 1000
        for i in range(50):
            assert f"key_{i}" in cache


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_reads_and_writes(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        for i in range(100):
            cache[f"key_{i}"] = f"value_{i}"

        errors = []

        def read(i):
            try:
                assert cache[f"key_{i}"] == f"value_{i}"
            except Exception as e:
                errors.append(e)

        def write(i):
            try:
                cache[f"new_key_{i}"] = f"new_value_{i}"
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = []
            for i in range(100):
                futures.append(pool.submit(read, i))
                futures.append(pool.submit(write, i))
            for f in as_completed(futures):
                f.result()

        assert not errors
        for i in range(100):
            assert cache[f"new_key_{i}"] == f"new_value_{i}"

    def test_concurrent_eviction(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=5000)
        errors = []

        def write(i):
            try:
                cache[f"key_{i}"] = "x" * 100
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(write, i) for i in range(200)]
            for f in as_completed(futures):
                f.result()

        assert not errors
        total = cache._get_conn().execute("SELECT COALESCE(SUM(size), 0) FROM cache").fetchone()[0]
        assert total < 5000 * 3

# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_concurrent_close_safety(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache["key"] = "value"
        cache.close()
        with pytest.raises(sqlite3.ProgrammingError):
            cache["key"]

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
    def test_get_conn_resets_lock_after_fork(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache["key"] = "value"
        cache._lock.acquire()
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                cache._get_conn()
                acquired = cache._lock.acquire(timeout=0.2)
                if acquired:
                    cache._lock.release()
                os.write(write_fd, b"1" if acquired else b"0")
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        os.waitpid(pid, 0)
        result = os.read(read_fd, 1)
        os.close(read_fd)
        cache._lock.release()
        assert result == b"1"

    def test_multiple_caches_same_directory(self, tmp_path):
        cache1 = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache2 = SQLiteCache(directory=str(tmp_path), size_limit=None)
        cache1["from_1"] = "value_1"
        assert cache2["from_1"] == "value_1"

    def test_value_with_envelope_like_keys(self, tmp_path):
        """Values that look like our internal envelope don't confuse deserialization."""
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        tricky = {"_pydantic": "fake.Module.FakeClass", "_data": {"foo": "bar"}}
        cache["tricky"] = tricky
        assert cache["tricky"] == tricky


# ── Migration helpers ────────────────────────────────────────────────────────


def _create_diskcache_shard(shard_dir: str, entries: list[tuple[str, object, float]]):
    """Create a fake diskcache shard with the given entries (key, value, access_time)."""
    os.makedirs(shard_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute(
        "CREATE TABLE Cache ("
        "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
        "  store_time REAL, expire_time REAL, access_time REAL,"
        "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
        "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
    )
    for key, value, access_time in entries:
        blob = pickle.dumps(value)
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, ?, ?)",
            (key, blob, access_time, len(blob)),
        )
    conn.commit()
    conn.close()


def _create_diskcache_shard_with_file(shard_dir: str, key: str, value: object, access_time: float):
    """Create a shard entry where the pickle data is stored in a .val file (NULL value in SQLite)."""
    os.makedirs(shard_dir, exist_ok=True)
    filename = "00/00/test.val"
    filepath = os.path.join(shard_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(value, f)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS Cache ("
        "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
        "  store_time REAL, expire_time REAL, access_time REAL,"
        "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
        "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
    )
    conn.execute(
        "INSERT INTO Cache (key, raw, value, mode, filename, access_time, size) VALUES (?, 1, NULL, 4, ?, ?, ?)",
        (key, filename, access_time, os.path.getsize(filepath)),
    )
    conn.commit()
    conn.close()


# ── Migration tests ──────────────────────────────────────────────────────────


class TestMigration:
    def test_has_legacy_diskcache_detection(self, tmp_path):
        assert not has_legacy_diskcache(str(tmp_path))
        _create_diskcache_shard(str(tmp_path / "000"), [("k", "v", time.time())])
        assert has_legacy_diskcache(str(tmp_path))

    def test_migrate_basic_entries(self, tmp_path):
        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("key1", {"hello": "world"}, 1000.0), ("key2", 42, 2000.0)],
        )
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 2
        assert errors == 0
        assert target["key1"] == {"hello": "world"}
        assert target["key2"] == 42

    def test_migrate_model_response(self, tmp_path):
        from litellm import ModelResponse

        response = ModelResponse(
            id="chatcmpl-test",
            choices=[{"message": {"content": "hello"}, "index": 0, "finish_reason": "stop"}],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        _create_diskcache_shard(str(tmp_path / "000"), [("mr_key", response, time.time())])
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        result = target["mr_key"]
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "hello"

    def test_migrate_text_mode_file_entries(self, tmp_path):
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir, exist_ok=True)
        filename = "00/00/text_entry.val"
        filepath = os.path.join(shard_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="UTF-8") as f:
            f.write("hello from text mode")
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute(
            "CREATE TABLE Cache ("
            "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
            "  store_time REAL, expire_time REAL, access_time REAL,"
            "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
            "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, filename, access_time, size) VALUES (?, 1, NULL, 3, ?, ?, ?)",
            ("text_key", filename, time.time(), os.path.getsize(filepath)),
        )
        conn.commit()
        conn.close()
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0
        assert target["text_key"] == "hello from text mode"

    def test_migrate_binary_mode_file_entries(self, tmp_path):
        """Binary mode stores raw bytes which aren't JSON-serializable, so they count as errors."""
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir, exist_ok=True)
        filename = "00/00/binary_entry.val"
        filepath = os.path.join(shard_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(b"\x00\x01\x02binary data")
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute(
            "CREATE TABLE Cache ("
            "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
            "  store_time REAL, expire_time REAL, access_time REAL,"
            "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
            "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, filename, access_time, size) VALUES (?, 1, NULL, 2, ?, ?, ?)",
            ("bin_key", filename, time.time(), os.path.getsize(filepath)),
        )
        conn.commit()
        conn.close()
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 0
        assert errors == 1

    def test_migrate_file_backed_entries(self, tmp_path):
        _create_diskcache_shard_with_file(
            str(tmp_path / "000"), "file_key", {"large": "data"}, time.time()
        )
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert target["file_key"] == {"large": "data"}

    def test_migrate_preserves_access_time(self, tmp_path):
        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("old", "old_val", 1000.0), ("new", "new_val", 2000.0)],
        )
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrate_diskcache(str(tmp_path), target)
        conn = sqlite3.connect(str(tmp_path / "dspy_cache.db"))
        rows = conn.execute("SELECT key, last_access FROM cache ORDER BY last_access").fetchall()
        conn.close()
        assert rows[0] == ("old", 1000.0)
        assert rows[1] == ("new", 2000.0)

    def test_migrate_preserves_zero_access_time(self, tmp_path):
        """access_time=0.0 is valid (epoch) and must not be replaced with current time."""
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute(
            "CREATE TABLE Cache ("
            "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
            "  store_time REAL, expire_time REAL, access_time REAL,"
            "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
            "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, 0.0, 50)",
            ("epoch_key", pickle.dumps("epoch_value")),
        )
        conn.commit()
        conn.close()
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrate_diskcache(str(tmp_path), target)
        db_conn = sqlite3.connect(str(tmp_path / "dspy_cache.db"))
        row = db_conn.execute("SELECT last_access FROM cache WHERE key = 'epoch_key'").fetchone()
        db_conn.close()
        assert row[0] == 0.0

    def test_migrate_handles_permission_errors(self, tmp_path):
        """Files that can't be read should be counted as errors, not crash migration."""
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir, exist_ok=True)
        filename = "00/00/secret.val"
        filepath = os.path.join(shard_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("secret")
        os.chmod(filepath, 0o000)
        try:
            conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
            conn.execute(
                "CREATE TABLE Cache ("
                "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
                "  store_time REAL, expire_time REAL, access_time REAL,"
                "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
                "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
            )
            conn.execute(
                "INSERT INTO Cache (key, raw, value, mode, filename, access_time, size)"
                " VALUES (?, 1, NULL, 3, ?, ?, 100)",
                ("perm_key", filename, time.time()),
            )
            conn.commit()
            conn.close()
            target = SQLiteCache(directory=str(tmp_path), size_limit=None)
            migrated, errors = migrate_diskcache(str(tmp_path), target)
            assert migrated == 0
            assert errors == 1
        finally:
            os.chmod(filepath, 0o644)

    def test_migrate_skips_corrupt_entries(self, tmp_path):
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute(
            "CREATE TABLE Cache ("
            "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
            "  store_time REAL, expire_time REAL, access_time REAL,"
            "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
            "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, ?, 50)",
            ("good", pickle.dumps("hello"), time.time()),
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, ?, 50)",
            ("bad", b"\x80\x04\x95CORRUPT", time.time()),
        )
        conn.commit()
        conn.close()

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 1
        assert target["good"] == "hello"

    def test_migrate_handles_missing_shards_and_wrong_schema(self, tmp_path):
        _create_diskcache_shard(str(tmp_path / "000"), [("k1", "v1", time.time())])
        # Shard with wrong schema
        shard_dir = str(tmp_path / "001")
        os.makedirs(shard_dir)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()
        # Shards 002-015 don't exist at all

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0

    def test_migrate_multiple_shards(self, tmp_path):
        for shard_id in range(3):
            _create_diskcache_shard(
                str(tmp_path / f"{shard_id:03d}"),
                [(f"shard{shard_id}_key", f"shard{shard_id}_val", time.time())],
            )
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 3
        for i in range(3):
            assert target[f"shard{i}_key"] == f"shard{i}_val"

    def test_cache_init_triggers_auto_migration(self, tmp_path, monkeypatch):
        from dspy.clients.cache import Cache

        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("auto_key", {"auto": True}, time.time())],
        )

        monkeypatch.setenv("DSPY_MIGRATE_CACHE", "1")
        cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=False,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )
        assert cache.disk_cache["auto_key"] == {"auto": True}

    def test_cache_init_skips_migration_when_db_non_empty(self, tmp_path, monkeypatch):
        """Migration is skipped on second start since new DB already has data."""
        from dspy.clients.cache import Cache

        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("migrated_key", {"migrated": True}, time.time())],
        )
        monkeypatch.setenv("DSPY_MIGRATE_CACHE", "1")
        cache1 = Cache(
            enable_disk_cache=True,
            enable_memory_cache=False,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )
        assert cache1.disk_cache["migrated_key"] == {"migrated": True}
        request = {"prompt": "post_migration"}
        cache1.put(request, "new_value")
        # Second init skips migration (DB non-empty), both old and new data present
        cache2 = Cache(
            enable_disk_cache=True,
            enable_memory_cache=False,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )
        assert cache2.disk_cache["migrated_key"] == {"migrated": True}
        assert cache2.get(request) == "new_value"

    def test_cache_init_migration_e2e_model_response(self, tmp_path, monkeypatch):
        """Full e2e: migrate a ModelResponse from legacy shard, retrieve as cache hit."""
        from litellm import ModelResponse

        from dspy.clients.cache import Cache

        response = ModelResponse(
            id="chatcmpl-old",
            choices=[{"message": {"content": "old cached"}, "index": 0, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        request = {"model": "openai/gpt-5-nano", "prompt": "what is 2+2"}
        from hashlib import sha256

        key = sha256(orjson.dumps(request, option=orjson.OPT_SORT_KEYS)).hexdigest()

        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute(
            "CREATE TABLE Cache ("
            "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
            "  store_time REAL, expire_time REAL, access_time REAL,"
            "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
            "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
        )
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, ?, 500)",
            (key, pickle.dumps(response), time.time()),
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("DSPY_MIGRATE_CACHE", "1")
        cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )
        result = cache.get(request)
        assert result is not None
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "old cached"
        assert result.cache_hit is True
        assert result.usage == {}

    def test_cache_init_skips_migration_without_env_var(self, tmp_path, monkeypatch):
        from dspy.clients.cache import Cache

        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("auto_key", {"auto": True}, time.time())],
        )

        monkeypatch.delenv("DSPY_MIGRATE_CACHE", raising=False)
        cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=False,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )
        assert cache.disk_cache.is_empty()


# ── Pydantic extra-field handling ────────────────────────────────────────────


def test_serialize_roundtrip_pydantic_extra_fields():
    from litellm.types.utils import ChatCompletionMessageToolCall

    tc = ChatCompletionMessageToolCall(
        function={"name": "test", "arguments": "{}"},
        id="call_123",
        type="function",
    )
    result = _deserialize(_serialize(tc))
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.id == "call_123"
    assert result.type == "function"
    assert result.function.name == "test"
