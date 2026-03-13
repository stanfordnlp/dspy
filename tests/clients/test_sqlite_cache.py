"""Tests for SQLiteCache - serialization, core behavior, concurrency, migration."""

import asyncio
import os
import pickle
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from unittest.mock import patch

import orjson
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.cache import Cache, request_cache
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


class AliasPydanticModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=False)
    actual_name: int = pydantic.Field(alias="wire_name")

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


def test_serialize_roundtrip_pydantic_with_aliases():
    m = AliasPydanticModel(wire_name=3)
    result = _deserialize(_serialize(m))
    assert isinstance(result, AliasPydanticModel)
    assert result == m


def test_serialize_roundtrip_nested_class():
    m = Outer(inner=Outer.Inner(value=42), label="test")
    result = _deserialize(_serialize(m))
    assert isinstance(result, Outer)
    assert isinstance(result.inner, Outer.Inner)
    assert result.inner.value == 42
    assert result.label == "test"


def test_serialize_non_serializable_raises():
    @dataclass
    class DataclassValue:
        x: int

    class Custom:
        pass

    with pytest.raises(TypeError):
        _serialize(Custom())

    with pytest.raises(TypeError):
        _serialize({"obj": Custom()})

    with pytest.raises(TypeError):
        _serialize((1, 2, 3))

    with pytest.raises(TypeError):
        _serialize(DataclassValue(x=1))


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

    def test_get_returns_default_on_missing(self, tmp_path):
        cache = SQLiteCache(directory=str(tmp_path), size_limit=None)
        assert cache.get("missing", "default") == "default"

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

        @dataclass
        class DataclassValue:
            x: int

        class Custom:
            pass

        with pytest.raises(TypeError):
            cache["bad"] = Custom()

        with pytest.raises(TypeError):
            cache["tuple"] = (1, 2, 3)

        with pytest.raises(TypeError):
            cache["dataclass"] = DataclassValue(x=1)


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


def _make_legacy_model_response():
    from litellm import ModelResponse

    return ModelResponse(
        id="chatcmpl-legacy-rich",
        model="openai/gpt-4o-mini",
        choices=[{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Chicago"}',
                    },
                }],
            },
        }],
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )


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

        response = _make_legacy_model_response()
        _create_diskcache_shard(str(tmp_path / "000"), [("mr_key", response, time.time())])
        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0
        result = target["mr_key"]
        assert isinstance(result, ModelResponse)
        assert result.id == response.id
        assert result.model == response.model
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls[0].id == "call_123"
        assert result.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        assert result.choices[0].message.tool_calls[0].function.arguments == '{"city":"Chicago"}'
        assert result.usage.prompt_tokens == 11
        assert result.usage.total_tokens == 18

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

        response = _make_legacy_model_response()
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
        migrated = cache.disk_cache[key]
        assert isinstance(migrated, ModelResponse)
        assert migrated.id == response.id
        assert migrated.model == response.model
        assert migrated.choices[0].finish_reason == "tool_calls"
        assert migrated.choices[0].message.tool_calls[0].id == "call_123"
        assert migrated.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        assert migrated.choices[0].message.tool_calls[0].function.arguments == '{"city":"Chicago"}'
        assert migrated.usage.prompt_tokens == 11
        assert migrated.usage.total_tokens == 18

        result = cache.get(request)
        assert result is not None
        assert isinstance(result, ModelResponse)
        assert result.id == response.id
        assert result.model == response.model
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls[0].id == "call_123"
        assert result.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        assert result.choices[0].message.tool_calls[0].function.arguments == '{"city":"Chicago"}'
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


# ── Migration pattern helpers ────────────────────────────────────────────────


def _create_legacy_shard(directory: str, shard_num: int, entries: list[tuple[str, object, int]]):
    """Create a fake diskcache shard with pickled entries.

    Each entry is ``(key, value, mode)`` where mode=4 means PICKLE.
    """
    shard_dir = os.path.join(directory, f"{shard_num:03d}")
    os.makedirs(shard_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS Cache ("
        "  rowid INTEGER PRIMARY KEY, key BLOB, raw INTEGER,"
        "  store_time REAL, expire_time REAL, access_time REAL,"
        "  access_count INTEGER DEFAULT 0, tag BLOB, size INTEGER DEFAULT 0,"
        "  mode INTEGER DEFAULT 0, filename TEXT, value BLOB)"
    )
    for key, value, mode in entries:
        blob = pickle.dumps(value) if mode == 4 else value
        conn.execute(
            "INSERT INTO Cache (key, raw, value, access_time, mode, size) VALUES (?, 1, ?, ?, ?, ?)",
            (key, blob, time.time(), mode, len(blob) if isinstance(blob, bytes) else 0),
        )
    conn.commit()
    conn.close()


def _make_model_response() -> ModelResponse:
    """Create a realistic ModelResponse matching what DSPy stores."""
    return ModelResponse(
        id="chatcmpl-abc123",
        model="openai/gpt-4o-mini",
        choices=[{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
            },
        }],
        usage={"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    )


def _make_model_response_with_tool_calls() -> ModelResponse:
    """Create a ModelResponse with tool_calls, matching DSPy function-calling patterns."""
    return ModelResponse(
        id="chatcmpl-tools-456",
        model="openai/gpt-4o-mini",
        choices=[{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "San Francisco", "unit": "celsius"}',
                        },
                    },
                    {
                        "id": "call_def",
                        "type": "function",
                        "function": {
                            "name": "get_population",
                            "arguments": '{"city": "San Francisco"}',
                        },
                    },
                ],
            },
        }],
        usage={"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
    )


def _make_embedding_response() -> EmbeddingResponse:
    """Create an EmbeddingResponse matching what DSPy stores for embeddings."""
    return EmbeddingResponse(
        data=[
            {"embedding": [0.123, -0.456, 0.789, 0.0, -1.0], "index": 0, "object": "embedding"},
            {"embedding": [0.111, 0.222, 0.333, 0.444, 0.555], "index": 1, "object": "embedding"},
        ],
        model="text-embedding-3-small",
        usage={"prompt_tokens": 12, "total_tokens": 12},
    )


# ── Migration pattern tests ─────────────────────────────────────────────────


class TestMigrationPatterns:
    """Migration tests for real DSPy cache patterns using litellm response objects."""

    def test_pickled_model_response_migrates(self, tmp_path):
        """VAL-MIG-001: Pickled ModelResponse from legacy shard migrates to new format
        and reads back correctly."""
        response = _make_model_response()
        _create_legacy_shard(str(tmp_path), 0, [("mr_key_001", response, 4)])

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)

        assert migrated == 1
        assert errors == 0

        result = target["mr_key_001"]
        assert isinstance(result, ModelResponse)
        assert result.id == "chatcmpl-abc123"
        assert result.model == "openai/gpt-4o-mini"
        assert result.choices[0].message.content == "The answer is 42."
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 23

    def test_model_response_with_tool_calls_migrates(self, tmp_path):
        """VAL-MIG-002: Pickled ModelResponse with tool_calls preserves tool call data
        through migration."""
        response = _make_model_response_with_tool_calls()
        _create_legacy_shard(str(tmp_path), 0, [("tc_key_001", response, 4)])

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)

        assert migrated == 1
        assert errors == 0

        result = target["tc_key_001"]
        assert isinstance(result, ModelResponse)
        assert result.id == "chatcmpl-tools-456"
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.content is None

        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 2

        assert tool_calls[0].id == "call_abc"
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city": "San Francisco", "unit": "celsius"}'

        assert tool_calls[1].id == "call_def"
        assert tool_calls[1].type == "function"
        assert tool_calls[1].function.name == "get_population"
        assert tool_calls[1].function.arguments == '{"city": "San Francisco"}'

    def test_entries_from_multiple_shards_migrate(self, tmp_path):
        """VAL-MIG-003: Entries spread across 3+ shards all migrate to the target DB.
        No entries are lost."""
        responses = {}
        for shard_id in [0, 5, 12]:
            key = f"shard_{shard_id:03d}_response"
            resp = ModelResponse(
                id=f"chatcmpl-shard{shard_id}",
                model="openai/gpt-4o-mini",
                choices=[{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"Response from shard {shard_id}"},
                }],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            responses[key] = resp
            _create_legacy_shard(str(tmp_path), shard_id, [(key, resp, 4)])

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)

        assert migrated == 3
        assert errors == 0

        for key, original in responses.items():
            result = target[key]
            assert isinstance(result, ModelResponse)
            assert result.id == original.id
            assert result.choices[0].message.content == original.choices[0].message.content

    def test_migrated_entry_cache_hit(self, tmp_path, monkeypatch):
        """VAL-MIG-004: After migration, Cache.get() with the same key returns
        the migrated value (cache hit via the full Cache layer)."""
        response = _make_model_response()
        request = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the meaning of life?"}],
            "_fn_identifier": "dspy.clients.lm.litellm_completion",
        }

        # Compute the cache key the same way Cache does
        from hashlib import sha256

        key = sha256(orjson.dumps(request, option=orjson.OPT_SORT_KEYS)).hexdigest()

        _create_legacy_shard(str(tmp_path), 0, [(key, response, 4)])

        monkeypatch.setenv("DSPY_MIGRATE_CACHE", "1")
        cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=1024 * 1024,
            memory_max_entries=100,
        )

        # Use Cache.get() with the original request dict
        result = cache.get(request)
        assert result is not None
        assert isinstance(result, ModelResponse)
        assert result.id == "chatcmpl-abc123"
        assert result.choices[0].message.content == "The answer is 42."
        assert result.cache_hit is True
        assert result.usage == {}

    def test_embedding_response_migrates(self, tmp_path):
        """VAL-MIG-005: Pickled EmbeddingResponse migrates correctly and embedding
        data is preserved."""
        response = _make_embedding_response()
        _create_legacy_shard(str(tmp_path), 0, [("emb_key_001", response, 4)])

        target = SQLiteCache(directory=str(tmp_path), size_limit=None)
        migrated, errors = migrate_diskcache(str(tmp_path), target)

        assert migrated == 1
        assert errors == 0

        result = target["emb_key_001"]
        assert isinstance(result, EmbeddingResponse)
        assert result.model == "text-embedding-3-small"
        assert len(result.data) == 2

        assert result.data[0]["embedding"] == pytest.approx([0.123, -0.456, 0.789, 0.0, -1.0])
        assert result.data[0]["index"] == 0
        assert result.data[0]["object"] == "embedding"

        assert result.data[1]["embedding"] == pytest.approx([0.111, 0.222, 0.333, 0.444, 0.555])
        assert result.data[1]["index"] == 1

        assert result.usage.prompt_tokens == 12
        assert result.usage.total_tokens == 12


# ── Cache tier concurrency fixture ───────────────────────────────────────────


@pytest.fixture
def tier_cache(tmp_path):
    """Cache with both memory and disk tiers enabled."""
    return Cache(
        enable_memory_cache=True,
        enable_disk_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=int(1e9),
        memory_max_entries=100_000,
    )


# ── Cache tier concurrency tests ─────────────────────────────────────────────


class TestCacheTierConcurrency:
    """Concurrent cache access patterns that DSPy users naturally hit."""

    @pytest.mark.asyncio
    async def test_async_concurrent_cache_access(self, tmp_path):
        """20+ asyncio tasks via asyncio.gather concurrently call a @request_cache-decorated
        async function with both shared and distinct keys. All return correct values, no corruption.

        Simulates parallel aforward() calls hitting the cache layer.
        """
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=int(1e9),
            memory_max_entries=100_000,
        )

        call_count = 0
        call_lock = asyncio.Lock()

        @request_cache()
        async def cached_function(prompt, model="openai/gpt-5-nano"):
            nonlocal call_count
            async with call_lock:
                call_count += 1
            # Simulate async work
            await asyncio.sleep(0.001)
            return f"result_for_{prompt}"

        with patch("dspy.cache", cache):
            # Create 30 tasks: 10 shared keys (3 calls each) + some distinct keys
            tasks = []

            # 10 distinct prompts, each called 3 times = 30 tasks total
            for i in range(10):
                for _ in range(3):
                    tasks.append(cached_function(prompt=f"prompt_{i}", model="openai/gpt-5-nano"))

            results = await asyncio.gather(*tasks)

        # All 30 results should be correct
        assert len(results) == 30
        for i in range(10):
            for j in range(3):
                idx = i * 3 + j
                assert results[idx] == f"result_for_prompt_{i}", (
                    f"Task {idx} returned {results[idx]!r}, expected 'result_for_prompt_{i}'"
                )

        # The function should have been called at most 10 times (one per distinct key),
        # but may be called more due to concurrent cache misses (no cross-task locking).
        # The important thing is correctness of values, not deduplication.
        assert call_count >= 10
        assert call_count <= 30

    @pytest.mark.asyncio
    async def test_async_concurrent_cache_access_distinct_keys(self, tmp_path):
        """20+ asyncio tasks with all distinct keys. Verifies no corruption when
        many tasks populate cache simultaneously."""
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=int(1e9),
            memory_max_entries=100_000,
        )

        @request_cache()
        async def cached_function(prompt, model="openai/gpt-5-nano"):
            await asyncio.sleep(0.001)
            return f"result_for_{prompt}"

        with patch("dspy.cache", cache):
            tasks = [cached_function(prompt=f"unique_{i}", model="openai/gpt-5-nano") for i in range(25)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 25
        for i in range(25):
            assert results[i] == f"result_for_unique_{i}"

    def test_threaded_cache_access_both_tiers(self, tier_cache):
        """ThreadPoolExecutor with 8+ workers do Cache.get()/Cache.put() with both
        memory and disk enabled. All threads get correct values, no deadlocks.

        Simulates DSPy's Parallelizer usage pattern.
        """
        num_workers = 16
        num_entries = 100
        errors = []

        # Pre-populate half the entries
        for i in range(0, num_entries, 2):
            request = {"prompt": f"prompt_{i}", "model": "openai/gpt-5-nano"}
            tier_cache.put(request, f"value_{i}")

        def worker(i):
            try:
                request = {"prompt": f"prompt_{i}", "model": "openai/gpt-5-nano"}
                if i % 2 == 0:
                    # Even indices: read pre-populated entries
                    result = tier_cache.get(request)
                    assert result == f"value_{i}", f"Thread {i} got {result!r}, expected 'value_{i}'"
                else:
                    # Odd indices: write new entries
                    tier_cache.put(request, f"value_{i}")
                    result = tier_cache.get(request)
                    assert result == f"value_{i}", f"Thread {i} got {result!r} after put"
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_entries)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

        # Verify all entries are readable after concurrent access
        for i in range(num_entries):
            request = {"prompt": f"prompt_{i}", "model": "openai/gpt-5-nano"}
            result = tier_cache.get(request)
            assert result == f"value_{i}", f"Final check: key {i} got {result!r}"

    def test_threaded_cache_access_heavy_contention(self, tier_cache):
        """8+ threads all reading and writing the same small set of keys simultaneously.
        Tests for deadlocks under high contention on the same keys."""
        num_workers = 12
        iterations_per_worker = 50
        errors = []

        def worker(worker_id):
            try:
                for j in range(iterations_per_worker):
                    key_idx = j % 5  # Only 5 keys to maximize contention
                    request = {"prompt": f"shared_{key_idx}", "model": "test"}
                    tier_cache.put(request, f"worker_{worker_id}_iter_{j}")
                    result = tier_cache.get(request)
                    # Value should be a string matching the pattern from some worker
                    assert result is not None, f"Worker {worker_id} got None for key {key_idx}"
                    assert isinstance(result, str), f"Worker {worker_id} got non-string: {result!r}"
                    assert result.startswith("worker_"), f"Worker {worker_id} got corrupted value: {result!r}"
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

    def test_concurrent_overlapping_key_writes(self, tier_cache):
        """Multiple threads write different values to the same key via Cache.put().
        After completion, Cache.get() returns one consistent value (not corrupted).
        """
        num_writers = 16
        request = {"prompt": "contested_key", "model": "openai/gpt-5-nano"}
        errors = []
        written_values = [f"value_from_thread_{i}" for i in range(num_writers)]

        def writer(i):
            try:
                tier_cache.put(request, written_values[i])
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(num_writers)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

        # After all writes, the value should be one of the written values (not corrupted)
        final_value = tier_cache.get(request)
        assert final_value in written_values, (
            f"Final value {final_value!r} is not one of the written values"
        )

        # Multiple reads should all return the same consistent value
        results = [tier_cache.get(request) for _ in range(10)]
        assert all(r == final_value for r in results), (
            f"Inconsistent reads: {set(results)}"
        )

    def test_concurrent_overlapping_key_writes_disk_consistency(self, tier_cache):
        """After concurrent writes to the same key, clearing memory cache and reading from
        disk should return a consistent value that matches what memory returns."""
        num_writers = 12
        request = {"prompt": "disk_contested", "model": "test"}
        written_values = [f"disk_value_{i}" for i in range(num_writers)]

        def writer(i):
            tier_cache.put(request, written_values[i])

        with ThreadPoolExecutor(max_workers=num_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(num_writers)]
            for f in as_completed(futures):
                f.result()

        # Read from memory
        memory_value = tier_cache.get(request)
        assert memory_value in written_values

        # Clear memory, read from disk
        tier_cache.reset_memory_cache()
        disk_value = tier_cache.get(request)
        assert disk_value in written_values

        # Both should be consistent (disk may differ from memory if last write to
        # each tier came from different threads, but each should be a valid value)

    def test_concurrent_disk_to_memory_promotion(self, tier_cache):
        """Store key in Cache (both tiers), reset memory cache, then 8+ threads
        simultaneously call Cache.get() for same key. All get correct value from
        disk promotion."""
        request = {"prompt": "promote_me", "model": "openai/gpt-5-nano"}
        expected_value = {"content": "promoted_response", "tokens": 42}

        # Store in both tiers
        tier_cache.put(request, expected_value)

        # Verify it's in both tiers
        key = tier_cache.cache_key(request)
        assert key in tier_cache.memory_cache
        assert key in tier_cache.disk_cache

        # Clear memory cache so all threads must promote from disk
        tier_cache.reset_memory_cache()
        assert key not in tier_cache.memory_cache

        num_readers = 16
        results = [None] * num_readers
        errors = []

        def reader(i):
            try:
                result = tier_cache.get(request)
                results[i] = result
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_readers) as executor:
            futures = [executor.submit(reader, i) for i in range(num_readers)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

        # All threads should have gotten the correct value
        for i, result in enumerate(results):
            assert result == expected_value, (
                f"Thread {i} got {result!r}, expected {expected_value!r}"
            )

        # After promotion, the key should be back in memory cache
        assert key in tier_cache.memory_cache

    def test_concurrent_disk_to_memory_promotion_model_response(self, tmp_path):
        """Concurrent disk-to-memory promotion with a real ModelResponse object.
        Verifies that deepcopy in Cache.get() handles concurrent access correctly
        and all responses have cache_hit=True, usage={}."""
        cache = Cache(
            enable_memory_cache=True,
            enable_disk_cache=True,
            disk_cache_dir=str(tmp_path),
            disk_size_limit_bytes=int(1e9),
            memory_max_entries=100_000,
        )

        response = ModelResponse(
            id="chatcmpl-concurrent",
            choices=[{"message": {"content": "concurrent response"}, "index": 0, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        request = {"prompt": "concurrent_model_response", "model": "openai/gpt-5-nano"}
        cache.put(request, response)
        cache.reset_memory_cache()

        num_readers = 12
        results = [None] * num_readers
        errors = []

        def reader(i):
            try:
                results[i] = cache.get(request)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_readers) as executor:
            futures = [executor.submit(reader, i) for i in range(num_readers)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

        for i, result in enumerate(results):
            assert isinstance(result, ModelResponse), f"Thread {i} got type {type(result)}"
            assert result.choices[0].message.content == "concurrent response"
            assert result.cache_hit is True
            assert result.usage == {}

        # Verify all results are independent copies (mutating one doesn't affect others)
        results[0].choices[0].message.content = "mutated"
        assert results[1].choices[0].message.content == "concurrent response"

    def test_concurrent_put_and_get_interleaved(self, tier_cache):
        """Threads interleave put() and get() calls on the same set of keys.
        Verifies no deadlocks or crashes under mixed read/write load."""
        num_workers = 10
        num_keys = 20
        errors = []
        barrier = threading.Barrier(num_workers)

        def worker(worker_id):
            try:
                barrier.wait(timeout=5)  # Start all threads simultaneously
                for key_idx in range(num_keys):
                    request = {"prompt": f"interleaved_{key_idx}", "model": "test"}
                    if worker_id % 2 == 0:
                        tier_cache.put(request, f"w{worker_id}_k{key_idx}")
                    else:
                        tier_cache.get(request)  # May return None or a value
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Errors from threads: {errors}"

        # All keys should have been written by at least one thread
        for key_idx in range(num_keys):
            request = {"prompt": f"interleaved_{key_idx}", "model": "test"}
            result = tier_cache.get(request)
            assert result is not None, f"Key {key_idx} was never written"
