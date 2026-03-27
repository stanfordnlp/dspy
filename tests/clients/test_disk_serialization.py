"""Tests for OrjsonDisk serialization and legacy cache migration."""

import os
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256

import diskcache
import orjson
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.cache import Cache
from dspy.clients.cache_migration import migrate_diskcache
from dspy.clients.disk_serialization import OrjsonDisk, _decode_value, _encode_value

# ── Helpers ──────────────────────────────────────────────────────────────────


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
    return orjson.dumps({"_data": _encode_value(value)})


def _deserialize(blob):
    return _decode_value(orjson.loads(blob)["_data"])


def _make_fanout_cache(directory):
    return diskcache.FanoutCache(
        directory=directory, shards=16, disk=OrjsonDisk,
        size_limit=2**40, eviction_policy="least-recently-stored", timeout=60,
    )


def _create_diskcache_shard(shard_dir, entries):
    os.makedirs(shard_dir, exist_ok=True)
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
    for key, value, access_time in entries:
        blob = pickle.dumps(value)
        conn.execute(
            "INSERT INTO Cache (key, raw, value, mode, access_time, size) VALUES (?, 1, ?, 4, ?, ?)",
            (key, blob, access_time, len(blob)),
        )
    conn.commit()
    conn.close()


def _create_diskcache_shard_with_file(shard_dir, key, value, access_time):
    os.makedirs(shard_dir, exist_ok=True)
    filename = "00/00/test.val"
    filepath = os.path.join(shard_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(value, f)
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
    conn.execute(
        "INSERT INTO Cache (key, raw, value, mode, filename, access_time, size) VALUES (?, 1, NULL, 4, ?, ?, ?)",
        (key, filename, access_time, os.path.getsize(filepath)),
    )
    conn.commit()
    conn.close()


# ── Serialization roundtrips ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [
        "hello", 42, 3.14, True, False, None,
        [1, "two", 3.0, None, True],
        {"a": 1, "b": [2, 3], "c": {"nested": True}},
        {}, [],
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
    class Custom:
        pass

    with pytest.raises(TypeError):
        _serialize(Custom())

    with pytest.raises(TypeError):
        _serialize((1, 2, 3))

    @dataclass
    class DC:
        x: int

    with pytest.raises(TypeError):
        _serialize(DC(x=1))


@pytest.mark.parametrize(
    "response, check",
    [
        (
            ModelResponse(
                id="chatcmpl-test123",
                choices=[{"message": {"content": "Hello world"}, "index": 0, "finish_reason": "stop"}],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            lambda r: r.choices[0].message.content == "Hello world",
        ),
        (
            ModelResponse(
                id="chatcmpl-tools",
                choices=[{"message": {"content": None, "tool_calls": [{"id": "call_123", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "SF"}'}}]},
                    "index": 0, "finish_reason": "tool_calls"}],
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


def test_openai_chat_completion_roundtrip():
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    cc = ChatCompletion(
        id="chatcmpl-test",
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[Choice(
            index=0,
            finish_reason="stop",
            message=ChatCompletionMessage(role="assistant", content="Hello world"),
        )],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    result = _deserialize(_serialize(cc))
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "Hello world"
    assert result.usage.total_tokens == 15


def test_serialize_roundtrip_pydantic_extra_fields():
    from litellm.types.utils import ChatCompletionMessageToolCall

    tc = ChatCompletionMessageToolCall(
        function={"name": "test", "arguments": "{}"}, id="call_123", type="function",
    )
    result = _deserialize(_serialize(tc))
    assert isinstance(result, ChatCompletionMessageToolCall)
    assert result.function.name == "test"


def test_envelope_like_keys_not_confused(tmp_path):
    cache = _make_fanout_cache(str(tmp_path))
    tricky = {"_pydantic": "fake.Module.FakeClass", "_data": {"foo": "bar"}}
    cache["tricky"] = tricky
    assert cache["tricky"] == tricky


# ── Migration ────────────────────────────────────────────────────────────────


class TestMigration:
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

    def test_cache_init_migration_e2e(self, tmp_path, monkeypatch):
        response = ModelResponse(
            id="chatcmpl-legacy",
            choices=[{"message": {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "lookup_weather", "arguments": '{"city":"Chicago"}'}}]},
                "index": 0, "finish_reason": "tool_calls"}],
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )
        request = {"model": "openai/gpt-5-nano", "prompt": "what is 2+2"}
        key = sha256(orjson.dumps(request, option=orjson.OPT_SORT_KEYS)).hexdigest()
        _create_diskcache_shard(str(tmp_path / "000"), [(key, response, time.time())])

        monkeypatch.setenv("DSPY_MIGRATE_CACHE", "1")
        cache = Cache(
            enable_disk_cache=True, enable_memory_cache=True,
            disk_cache_dir=str(tmp_path), disk_size_limit_bytes=1024 * 1024, memory_max_entries=100,
        )
        result = cache.get(request)
        assert isinstance(result, ModelResponse)
        assert result.id == response.id
        assert result.choices[0].message.tool_calls[0].function.name == "lookup_weather"
        assert result.cache_hit is True
        assert result.usage == {}

    def test_cache_init_skips_migration_without_env_var(self, tmp_path, monkeypatch):
        _create_diskcache_shard(str(tmp_path / "000"), [("k", {"v": 1}, time.time())])
        monkeypatch.delenv("DSPY_MIGRATE_CACHE", raising=False)
        cache = Cache(
            enable_disk_cache=True, enable_memory_cache=False,
            disk_cache_dir=str(tmp_path), disk_size_limit_bytes=1024 * 1024, memory_max_entries=100,
        )
        assert len(cache.disk_cache) == 0


# ── Concurrent disk-to-memory promotion ──────────────────────────────────────


def test_concurrent_disk_to_memory_promotion(tmp_path):
    cache = Cache(
        enable_memory_cache=True, enable_disk_cache=True,
        disk_cache_dir=str(tmp_path), disk_size_limit_bytes=int(1e9), memory_max_entries=100_000,
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
    for result in results:
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "concurrent response"
        assert result.cache_hit is True

    # deepcopy isolation: mutating one result doesn't affect another
    results[0].choices[0].message.content = "mutated"
    assert results[1].choices[0].message.content == "concurrent response"
