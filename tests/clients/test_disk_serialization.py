"""Tests for safe cache serialization."""

import os
import sqlite3
from dataclasses import dataclass

import diskcache
import numpy as np
import orjson
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.disk_serialization import (
    DeserializationError,
    NoPickleDisk,
    SafeTypeRegistry,
    create_default_registry,
    decode_value,
    encode_value,
)


class RegisteredPydanticModel(pydantic.BaseModel):
    name: str
    value: int


class NullablePydanticModel(pydantic.BaseModel):
    required_nullable: int | None
    optional_nullable: int | None = None


@dataclass
class RegisteredDataclass:
    name: str
    value: int


@pytest.fixture()
def registry():
    reg = create_default_registry()
    reg.register(RegisteredPydanticModel)
    reg.register(NullablePydanticModel)
    reg.register(RegisteredDataclass)
    return reg


def _make_fanout_cache(directory, **kwargs):
    return diskcache.FanoutCache(
        directory=directory,
        shards=16,
        disk=NoPickleDisk,
        size_limit=2**40,
        eviction_policy="least-recently-stored",
        timeout=60,
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


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1, "b": [2, 3], "c": {"nested": True}},
        RegisteredPydanticModel(name="test", value=42),
        RegisteredDataclass(name="test", value=42),
    ],
    ids=["json", "pydantic", "dataclass"],
)
def test_safe_roundtrip(value, registry):
    assert decode_value(encode_value(value, registry), registry) == value


def test_pydantic_roundtrip_preserves_unset_fields(registry):
    model = NullablePydanticModel(required_nullable=None)

    result = decode_value(encode_value(model, registry), registry)

    assert result == model
    assert result.model_fields_set == {"required_nullable"}


def test_ndarray_roundtrip_preserves_dtype_and_shape(registry):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = decode_value(encode_value(value, registry), registry)

    assert isinstance(result, np.ndarray)
    assert result.dtype == value.dtype
    assert result.shape == value.shape
    np.testing.assert_array_equal(result, value)


def test_large_entries_use_diskcache_file_storage(tmp_path, registry):
    cache = _make_fanout_cache(str(tmp_path), disk_min_file_size=1)
    encoded = encode_value({"payload": "x" * 8192}, registry)

    cache["large"] = encoded

    mode, filename = _find_cache_row(str(tmp_path), "large")
    assert mode == 2
    assert filename is not None
    assert decode_value(cache["large"], registry) == {"payload": "x" * 8192}


def test_litellm_response_roundtrip(registry):
    response = ModelResponse(
        id="chatcmpl-test123",
        choices=[{"message": {"content": "Hello world"}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    result = decode_value(encode_value(response, registry), registry)

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello world"


def test_embedding_response_roundtrip(registry):
    response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        model="text-embedding-3-small",
        usage={"prompt_tokens": 5, "total_tokens": 5},
    )

    result = decode_value(encode_value(response, registry), registry)

    assert isinstance(result, EmbeddingResponse)
    assert result.data[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3])


def test_unregistered_custom_types_raise(registry):
    @dataclass
    class UnregisteredDataclass:
        value: int

    with pytest.raises(TypeError):
        encode_value((1, 2, 3), registry)

    with pytest.raises(TypeError):
        encode_value(UnregisteredDataclass(value=1), registry)


def test_unknown_cached_type_is_rejected(registry):
    payload = b"json:" + orjson.dumps({"type": "tests.Unknown", "data": {}})

    with pytest.raises(DeserializationError, match="Unsupported cached type"):
        decode_value(payload, registry)


def test_registries_are_isolated():
    r1 = SafeTypeRegistry()
    r2 = SafeTypeRegistry()
    r1.register(RegisteredPydanticModel)

    assert r1.get_tag(RegisteredPydanticModel) is not None
    assert r2.get_tag(RegisteredPydanticModel) is None
