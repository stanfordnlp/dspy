"""Tests for OrjsonDisk serialization."""

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import diskcache
import orjson
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.cache import Cache
from dspy.clients.disk_serialization import (
    _ENVELOPE_KEY,
    DEFAULT_ALLOWED_NAMESPACES,
    DeserializationError,
    OrjsonDisk,
    _decode_value,
    _encode_value,
)

_TEST_ALLOWED_NAMESPACES = (*DEFAULT_ALLOWED_NAMESPACES, "tests", "test_disk_serialization")

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


def _serialize(value, allowed_namespaces=_TEST_ALLOWED_NAMESPACES):
    return orjson.dumps({_ENVELOPE_KEY: _encode_value(value, allowed_namespaces)})


def _deserialize(blob, allowed_namespaces=_TEST_ALLOWED_NAMESPACES):
    return _decode_value(orjson.loads(blob)[_ENVELOPE_KEY], allowed_namespaces)


def _make_fanout_cache(directory, **kwargs):
    return diskcache.FanoutCache(
        directory=directory, shards=16, disk=OrjsonDisk,
        size_limit=2**40, eviction_policy="least-recently-stored", timeout=60,
        **kwargs,
    )


def _find_cache_row(directory, key):
    for shard_id in range(16):
        shard_db = os.path.join(directory, f"{shard_id:03d}", "cache.db")
        if not os.path.exists(shard_db):
            continue

        conn = sqlite3.connect(shard_db)
        try:
            row = conn.execute(
                "SELECT mode, filename FROM Cache WHERE key = ? AND raw = 1",
                (key,),
            ).fetchone()
        finally:
            conn.close()

        if row is not None:
            return row

    raise AssertionError(f"Could not find cache row for key {key!r}")


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


def test_large_orjson_entries_use_diskcache_file_storage(tmp_path):
    cache = _make_fanout_cache(str(tmp_path), disk_min_file_size=1)
    value = {"payload": "x" * 8192}

    cache["large"] = value

    mode, filename = _find_cache_row(str(tmp_path), "large")
    assert mode == 2
    assert filename is not None
    assert cache["large"] == value


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
    """User dicts containing our sentinel keys must roundtrip as plain dicts,
    not be misinterpreted as pydantic/ndarray envelopes."""
    cache = _make_fanout_cache(str(tmp_path))

    # Dict with real sentinel keys but extra keys -- not an exact envelope match
    tricky_extra = {
        "__dspy_cache_type__": "pydantic",
        "__dspy_cache_module__": "os",
        "__dspy_cache_qualname__": "system",
        "__dspy_cache_data__": {},
        "extra_key": "this makes it not an envelope",
    }
    cache["tricky_extra"] = tricky_extra
    assert cache["tricky_extra"] == tricky_extra

    # Dict with only some sentinel keys
    tricky_partial = {"__dspy_cache_type__": "pydantic", "user_data": 42}
    cache["tricky_partial"] = tricky_partial
    assert cache["tricky_partial"] == tricky_partial


def test_poisoned_module_entry_rejected():
    """A crafted envelope pointing to a disallowed namespace is rejected before import."""
    poisoned = orjson.dumps({
        _ENVELOPE_KEY: {
            "__dspy_cache_type__": "pydantic",
            "__dspy_cache_module__": "os",
            "__dspy_cache_qualname__": "getcwd",
            "__dspy_cache_data__": {},
        }
    })
    with pytest.raises(DeserializationError, match="registered cache type registry"):
        _deserialize(poisoned, allowed_namespaces=DEFAULT_ALLOWED_NAMESPACES)


def test_allowed_namespace_non_basemodel_rejected():
    """An allowed namespace but non-BaseModel class is rejected after import."""
    poisoned = orjson.dumps({
        _ENVELOPE_KEY: {
            "__dspy_cache_type__": "pydantic",
            "__dspy_cache_module__": "pydantic",
            "__dspy_cache_qualname__": "ConfigDict",
            "__dspy_cache_data__": {},
        }
    })
    with pytest.raises(DeserializationError, match="not a pydantic BaseModel subclass"):
        _deserialize(poisoned, allowed_namespaces=("pydantic",))


def test_malformed_pydantic_metadata_rejected():
    """Malformed pydantic envelope metadata is normalized as a deserialization error."""
    poisoned = orjson.dumps({
        _ENVELOPE_KEY: {
            "__dspy_cache_type__": "pydantic",
            "__dspy_cache_module__": ["not", "a", "string"],
            "__dspy_cache_qualname__": "PydanticModel",
            "__dspy_cache_data__": {},
        }
    })
    with pytest.raises(DeserializationError, match="metadata must be strings"):
        _deserialize(poisoned, allowed_namespaces=_TEST_ALLOWED_NAMESPACES)


# ── Concurrent disk-to-memory promotion ──────────────────────────────────────


def test_concurrent_disk_to_memory_promotion(tmp_path):
    cache = Cache(
        enable_memory_cache=True, enable_disk_cache=True,
        disk_cache_dir=str(tmp_path), disk_size_limit_bytes=int(1e9), memory_max_entries=100_000,
        use_pickle=False,
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

    # deepcopy isolation: mutating one result doesn't affect another
    results[0].choices[0].message.content = "mutated"
    assert results[1].choices[0].message.content == "concurrent response"
