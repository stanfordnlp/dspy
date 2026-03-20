"""Tests for DSPyDisk serialization, diskcache integration, and legacy migration."""

import asyncio
import os
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from unittest.mock import patch

import diskcache
import orjson
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.cache import Cache, request_cache
from dspy.clients.cache_migration import has_legacy_diskcache, migrate_diskcache
from dspy.clients.disk import DSPyDisk, _decode_value, _encode_value

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


def _serialize(value):
    """Helper: serialize using the same logic as DSPyDisk.store()."""
    blob = orjson.dumps({"_data": _encode_value(value)})
    return blob


def _deserialize(blob):
    """Helper: deserialize using the same logic as DSPyDisk.fetch()."""
    envelope = orjson.loads(blob)
    return _decode_value(envelope["_data"])


def _make_fanout_cache(directory, size_limit=None):
    """Create a FanoutCache with DSPyDisk for testing."""
    effective_limit = size_limit if size_limit is not None else 2**40
    return diskcache.FanoutCache(
        directory=directory,
        shards=16,
        disk=DSPyDisk,
        size_limit=effective_limit,
        eviction_policy="least-recently-used",
        timeout=60,
    )


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


@pytest.mark.parametrize(
    "model",
    [
        PydanticModel(name="test", value=42, tags=["a", "b"], metadata={"k": "v"}),
        NestedPydanticModel(inner=PydanticModel(name="inner", value=1), count=5),
        AliasPydanticModel(wire_name=3),
        Outer(inner=Outer.Inner(value=42), label="test"),
    ],
    ids=["flat", "nested", "alias", "nested_class"],
)
def test_serialize_roundtrip_pydantic(model):
    result = _deserialize(_serialize(model))
    assert isinstance(result, type(model))
    assert result == model


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


@pytest.mark.parametrize(
    "response, check",
    [
        (
            ModelResponse(
                id="chatcmpl-test123",
                choices=[{"message": {"content": "Hello world"}, "index": 0, "finish_reason": "stop"}],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            lambda r: r.choices[0].message.content == "Hello world" and r.id == "chatcmpl-test123",
        ),
        (
            ModelResponse(
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
            ),
            lambda r: r.choices[0].finish_reason == "tool_calls",
        ),
        (
            EmbeddingResponse(
                data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
                model="text-embedding-3-small",
                usage={"prompt_tokens": 5, "total_tokens": 5},
            ),
            lambda r: r.data[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3]),
        ),
    ],
    ids=["model_response", "tool_calls", "embedding"],
)
def test_litellm_response_roundtrip(response, check):
    result = _deserialize(_serialize(response))
    assert isinstance(result, type(response))
    assert check(result)


# ── FanoutCache + DSPyDisk core behavior ─────────────────────────────────────


class TestDSPyDiskCacheBasic:
    def test_set_get_contains(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        assert "missing" not in cache
        cache["key1"] = "value1"
        assert "key1" in cache
        assert cache["key1"] == "value1"

    def test_get_returns_default_on_missing(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        assert cache.get("missing", "default") == "default"

    def test_key_error_on_missing(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        with pytest.raises(KeyError):
            _ = cache["nonexistent"]

    def test_overwrite_key(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        cache["key"] = "first"
        cache["key"] = "second"
        assert cache["key"] == "second"

    def test_many_keys(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        for i in range(1000):
            cache[f"key_{i}"] = f"value_{i}"
        for i in range(1000):
            assert cache[f"key_{i}"] == f"value_{i}"

    def test_pydantic_model_stored_and_retrieved(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        m = PydanticModel(name="cached", value=99, tags=["x"])
        cache["model"] = m
        result = cache["model"]
        assert isinstance(result, PydanticModel)
        assert result.name == "cached"

    def test_non_serializable_value_raises_on_set(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))

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

    @pytest.mark.parametrize(
        "value",
        [
            "x" * (4 * 1024 * 1024),
            {"key_%d" % i: "val_%d" % i * 1000 for i in range(5000)},
            ModelResponse(
                id="chatcmpl-large",
                choices=[{"message": {"content": "word " * 200_000}, "index": 0, "finish_reason": "stop"}],
                usage={"prompt_tokens": 100, "completion_tokens": 50000, "total_tokens": 50100},
            ),
            EmbeddingResponse(
                data=[{"embedding": [float(i) / 10000 for i in range(3072)], "index": 0, "object": "embedding"}],
                model="text-embedding-3-large",
                usage={"prompt_tokens": 10, "total_tokens": 10},
            ),
        ],
        ids=["4mb_string", "large_dict", "model_response", "embedding_3072d"],
    )
    def test_large_value_roundtrip(self, tmp_path, value):
        cache = _make_fanout_cache(str(tmp_path))
        cache["key"] = value
        result = cache["key"]
        assert isinstance(result, type(value))
        if isinstance(value, EmbeddingResponse):
            assert result.data[0]["embedding"] == pytest.approx(value.data[0]["embedding"])
        elif isinstance(value, ModelResponse):
            assert result.choices[0].message.content == value.choices[0].message.content
        else:
            assert result == value

    def test_large_value_overwrite_and_persistence(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
        cache["key"] = "x" * (2 * 1024 * 1024)
        cache["key"] = "y" * (2 * 1024 * 1024)
        assert cache["key"] == "y" * (2 * 1024 * 1024)

        large = {"data": list(range(100_000))}
        cache["persist"] = large
        cache.close()
        cache2 = _make_fanout_cache(str(tmp_path))
        assert cache2["persist"] == large


# ── Persistence ──────────────────────────────────────────────────────────────


class TestPersistence:
    def test_data_survives_close_and_reopen(self, tmp_path):
        cache1 = _make_fanout_cache(str(tmp_path))
        cache1["persistent"] = {"data": [1, 2, 3]}
        cache1.close()

        cache2 = _make_fanout_cache(str(tmp_path))
        assert cache2["persistent"] == {"data": [1, 2, 3]}

    def test_directory_copy_produces_functional_cache(self, tmp_path):
        import shutil

        src = tmp_path / "original"
        cache1 = _make_fanout_cache(str(src))
        cache1["small"] = "hello"
        cache1["large"] = "x" * 500_000
        cache1["nested"] = {"a": [1, 2, 3], "b": {"c": True}}
        cache1.close()

        dst = tmp_path / "copy"
        shutil.copytree(str(src), str(dst))

        cache2 = _make_fanout_cache(str(dst))
        assert cache2["small"] == "hello"
        assert cache2["large"] == "x" * 500_000
        assert cache2["nested"] == {"a": [1, 2, 3], "b": {"c": True}}

        cache2["new_key"] = "only_in_copy"
        cache_orig = _make_fanout_cache(str(src))
        assert "new_key" not in cache_orig


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_reads_and_writes(self, tmp_path):
        cache = _make_fanout_cache(str(tmp_path))
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


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_value_with_envelope_like_keys(self, tmp_path):
        """Values that look like our internal envelope don't confuse deserialization."""
        cache = _make_fanout_cache(str(tmp_path))
        tricky = {"_pydantic": "fake.Module.FakeClass", "_data": {"foo": "bar"}}
        cache["tricky"] = tricky
        assert cache["tricky"] == tricky


# ── Migration helpers ────────────────────────────────────────────────────────


def _create_diskcache_shard(shard_dir: str, entries: list[tuple[str, object, float]]):
    """Create a fake legacy diskcache shard with the given entries (key, value, access_time)."""
    os.makedirs(shard_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS Settings (key TEXT NOT NULL UNIQUE, value)"
    )
    conn.execute("INSERT OR REPLACE INTO Settings VALUES ('eviction_policy', 'least-recently-stored')")
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
    """Create a shard entry where the pickle data is stored in a .val file."""
    os.makedirs(shard_dir, exist_ok=True)
    filename = "00/00/test.val"
    filepath = os.path.join(shard_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(value, f)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS Settings (key TEXT NOT NULL UNIQUE, value)"
    )
    conn.execute("INSERT OR REPLACE INTO Settings VALUES ('eviction_policy', 'least-recently-stored')")
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

    def test_new_cache_not_detected_as_legacy(self, tmp_path):
        """A FanoutCache created with DSPyDisk should NOT be detected as legacy."""
        cache = _make_fanout_cache(str(tmp_path))
        cache["test"] = "value"
        cache.close()
        assert not has_legacy_diskcache(str(tmp_path))

    def test_migrate_basic_entries(self, tmp_path):
        _create_diskcache_shard(
            str(tmp_path / "000"),
            [("key1", {"hello": "world"}, 1000.0), ("key2", 42, 2000.0)],
        )
        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 2
        assert errors == 0
        assert target["key1"] == {"hello": "world"}
        assert target["key2"] == 42

    def test_migrate_file_backed_entries(self, tmp_path):
        _create_diskcache_shard_with_file(
            str(tmp_path / "000"), "file_key", {"large": "data"}, time.time()
        )
        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert target["file_key"] == {"large": "data"}

    def test_migrate_skips_corrupt_entries(self, tmp_path):
        shard_dir = str(tmp_path / "000")
        os.makedirs(shard_dir)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute("CREATE TABLE IF NOT EXISTS Settings (key TEXT NOT NULL UNIQUE, value)")
        conn.execute("INSERT OR REPLACE INTO Settings VALUES ('eviction_policy', 'least-recently-stored')")
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

        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 1
        assert target["good"] == "hello"

    def test_migrate_handles_missing_shards_and_wrong_schema(self, tmp_path):
        _create_diskcache_shard(str(tmp_path / "000"), [("k1", "v1", time.time())])
        shard_dir = str(tmp_path / "001")
        os.makedirs(shard_dir)
        conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()

        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0

    def test_cache_init_migration_e2e_model_response(self, tmp_path, monkeypatch):
        """Full e2e: migrate a ModelResponse from legacy shard, retrieve as cache hit."""
        response = _make_legacy_model_response()
        request = {"model": "openai/gpt-5-nano", "prompt": "what is 2+2"}
        from hashlib import sha256
        key = sha256(orjson.dumps(request, option=orjson.OPT_SORT_KEYS)).hexdigest()

        _create_diskcache_shard(
            str(tmp_path / "000"),
            [(key, response, time.time())],
        )

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
        assert result.id == response.id
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        assert result.cache_hit is True
        assert result.usage == {}

    def test_cache_init_skips_migration_without_env_var(self, tmp_path, monkeypatch):
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
        assert len(cache.disk_cache) == 0


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
    shard_dir = os.path.join(directory, f"{shard_num:03d}")
    os.makedirs(shard_dir, exist_ok=True)
    conn = sqlite3.connect(os.path.join(shard_dir, "cache.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS Settings (key TEXT NOT NULL UNIQUE, value)")
    conn.execute("INSERT OR REPLACE INTO Settings VALUES ('eviction_policy', 'least-recently-stored')")
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
    return ModelResponse(
        id="chatcmpl-abc123",
        model="openai/gpt-4o-mini",
        choices=[{"index": 0, "finish_reason": "stop",
                  "message": {"role": "assistant", "content": "The answer is 42."}}],
        usage={"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    )


def _make_embedding_response() -> EmbeddingResponse:
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
    def test_pickled_model_response_migrates(self, tmp_path):
        response = _make_model_response()
        _create_legacy_shard(str(tmp_path), 0, [("mr_key_001", response, 4)])
        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0
        result = target["mr_key_001"]
        assert isinstance(result, ModelResponse)
        assert result.id == "chatcmpl-abc123"
        assert result.choices[0].message.content == "The answer is 42."

    def test_entries_from_multiple_shards_migrate(self, tmp_path):
        responses = {}
        for shard_id in [0, 5, 12]:
            key = f"shard_{shard_id:03d}_response"
            resp = ModelResponse(
                id=f"chatcmpl-shard{shard_id}",
                model="openai/gpt-4o-mini",
                choices=[{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": f"Response from shard {shard_id}"}}],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            responses[key] = resp
            _create_legacy_shard(str(tmp_path), shard_id, [(key, resp, 4)])

        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 3
        assert errors == 0
        for key, original in responses.items():
            result = target[key]
            assert isinstance(result, ModelResponse)
            assert result.id == original.id

    def test_embedding_response_migrates(self, tmp_path):
        response = _make_embedding_response()
        _create_legacy_shard(str(tmp_path), 0, [("emb_key_001", response, 4)])
        target = _make_fanout_cache(str(tmp_path / "new_cache"))
        migrated, errors = migrate_diskcache(str(tmp_path), target)
        assert migrated == 1
        assert errors == 0
        result = target["emb_key_001"]
        assert isinstance(result, EmbeddingResponse)
        assert result.data[0]["embedding"] == pytest.approx([0.123, -0.456, 0.789, 0.0, -1.0])


# ── Cache tier concurrency ───────────────────────────────────────────────────


@pytest.fixture
def tier_cache(tmp_path):
    return Cache(
        enable_memory_cache=True,
        enable_disk_cache=True,
        disk_cache_dir=str(tmp_path),
        disk_size_limit_bytes=int(1e9),
        memory_max_entries=100_000,
    )


class TestCacheTierConcurrency:
    @pytest.mark.asyncio
    async def test_async_concurrent_cache_access(self, tmp_path):
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
            await asyncio.sleep(0.001)
            return f"result_for_{prompt}"

        with patch("dspy.cache", cache):
            tasks = []
            for i in range(10):
                for _ in range(3):
                    tasks.append(cached_function(prompt=f"prompt_{i}", model="openai/gpt-5-nano"))
            results = await asyncio.gather(*tasks)

        assert len(results) == 30
        for i in range(10):
            for j in range(3):
                assert results[i * 3 + j] == f"result_for_prompt_{i}"
        assert 10 <= call_count <= 30

    def test_threaded_cache_access_heavy_contention(self, tier_cache):
        num_workers = 12
        iterations_per_worker = 50
        errors = []

        def worker(worker_id):
            try:
                for j in range(iterations_per_worker):
                    key_idx = j % 5
                    request = {"prompt": f"shared_{key_idx}", "model": "test"}
                    tier_cache.put(request, f"worker_{worker_id}_iter_{j}")
                    result = tier_cache.get(request)
                    assert result is not None
                    assert isinstance(result, str)
                    assert result.startswith("worker_")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, w) for w in range(num_workers)]
            for f in as_completed(futures):
                f.result()
        assert not errors

    def test_concurrent_disk_to_memory_promotion_model_response(self, tmp_path):
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

        assert not errors
        for i, result in enumerate(results):
            assert isinstance(result, ModelResponse)
            assert result.choices[0].message.content == "concurrent response"
            assert result.cache_hit is True
            assert result.usage == {}

        results[0].choices[0].message.content = "mutated"
        assert results[1].choices[0].message.content == "concurrent response"
