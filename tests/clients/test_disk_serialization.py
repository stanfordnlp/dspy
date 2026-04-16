"""Tests for safe cache serialization."""

import os
import sqlite3
from dataclasses import dataclass, field

import diskcache
import msgspec.msgpack
import numpy as np
import pydantic
import pytest
from litellm.types.utils import EmbeddingResponse, ModelResponse

from dspy.clients.disk_serialization import (
    _TAG_PYDANTIC,
    DeserializationError,
    decode,
    default_allowed_types,
    encode,
    make_safe_disk,
)


class AllowedPydanticModel(pydantic.BaseModel):
    name: str
    value: int


class NullablePydanticModel(pydantic.BaseModel):
    required_nullable: int | None
    optional_nullable: int | None = None


class RootListModel(pydantic.RootModel[list[int]]):
    pass


class RootScalarModel(pydantic.RootModel[str]):
    pass


@dataclass
class AllowedDataclass:
    name: str
    value: int


@dataclass
class DataclassWithNonInitField:
    name: str
    value: int
    computed: str = field(init=False, default="default")


@pytest.fixture()
def allowed():
    types = default_allowed_types()
    types.add((AllowedPydanticModel.__module__, AllowedPydanticModel.__name__))
    types.add((NullablePydanticModel.__module__, NullablePydanticModel.__name__))
    types.add((AllowedDataclass.__module__, AllowedDataclass.__name__))
    types.add((RootListModel.__module__, RootListModel.__name__))
    types.add((RootScalarModel.__module__, RootScalarModel.__name__))
    types.add((DataclassWithNonInitField.__module__, DataclassWithNonInitField.__name__))
    return types


def _make_fanout_cache(directory, allowed, **kwargs):
    return diskcache.FanoutCache(
        directory=directory,
        shards=16,
        disk=make_safe_disk(allowed),
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
        AllowedPydanticModel(name="test", value=42),
        AllowedDataclass(name="test", value=42),
    ],
    ids=["json", "pydantic", "dataclass"],
)
def test_safe_roundtrip(value, allowed):
    assert decode(encode(value), allowed=allowed) == value


def test_pydantic_roundtrip_preserves_nullable_fields(allowed):
    model = NullablePydanticModel(required_nullable=None)
    result = decode(encode(model), allowed=allowed)
    assert result == model


def test_ndarray_roundtrip_preserves_dtype_and_shape(allowed):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = decode(encode(value), allowed=allowed)

    assert isinstance(result, np.ndarray)
    assert result.dtype == value.dtype
    assert result.shape == value.shape
    np.testing.assert_array_equal(result, value)


def test_large_entries_use_diskcache_file_storage(tmp_path, allowed):
    cache = _make_fanout_cache(str(tmp_path), allowed, disk_min_file_size=1)

    cache["large"] = {"payload": "x" * 8192}

    mode, filename = _find_cache_row(str(tmp_path), "large")
    assert mode == 2
    assert filename is not None
    assert cache["large"] == {"payload": "x" * 8192}


def test_litellm_response_roundtrip(allowed):
    response = ModelResponse(
        id="chatcmpl-test123",
        choices=[{"message": {"content": "Hello world"}, "index": 0, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    result = decode(encode(response), allowed=allowed)

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello world"


def test_embedding_response_roundtrip(allowed):
    response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        model="text-embedding-3-small",
        usage={"prompt_tokens": 5, "total_tokens": 5},
    )

    result = decode(encode(response), allowed=allowed)

    assert isinstance(result, EmbeddingResponse)
    assert result.data[0]["embedding"] == pytest.approx([0.1, 0.2, 0.3])


def test_unlisted_pydantic_model_blocked_on_decode():
    class Secret(pydantic.BaseModel):
        value: int

    encoded = encode(Secret(value=1))
    with pytest.raises(DeserializationError, match="not in the safe_types allowlist"):
        decode(encoded, allowed=set())


def test_tuple_roundtrips_as_list(allowed):
    result = decode(encode((1, 2, 3)), allowed=allowed)
    assert result == [1, 2, 3]


def test_unknown_routing_byte_is_rejected(allowed):
    with pytest.raises(DeserializationError, match="Unknown routing byte"):
        decode(b"\xff" + b"body", allowed=allowed)


def test_allowlists_are_isolated():
    a1 = {(AllowedPydanticModel.__module__, AllowedPydanticModel.__name__)}
    a2: set[tuple[str, str]] = set()

    encoded = encode(AllowedPydanticModel(name="test", value=1))
    assert decode(encoded, allowed=a1) == AllowedPydanticModel(name="test", value=1)
    with pytest.raises(DeserializationError, match="not in the safe_types allowlist"):
        decode(encoded, allowed=a2)


def test_schema_change_detected_on_decode(allowed):
    encoded = encode(AllowedPydanticModel(name="test", value=42))
    payload = msgspec.msgpack.decode(encoded[1:])
    payload["schema"] = "0000000000000000"
    tampered = _TAG_PYDANTIC + msgspec.msgpack.encode(payload)
    with pytest.raises(DeserializationError, match="has changed"):
        decode(tampered, allowed=allowed)


@pytest.mark.parametrize("value", [
    AllowedPydanticModel(name="test", value=1),
    AllowedDataclass(name="test", value=1),
], ids=["pydantic", "dataclass"])
def test_encode_checks_allowlist(value):
    empty_allowlist: set[tuple[str, str]] = set()
    with pytest.raises(TypeError, match="not in the safe_types allowlist"):
        encode(value, allowed=empty_allowlist)
    encode(value)


def test_plain_value_uses_msgspec_tag():
    encoded = encode({"key": "value"})
    assert encoded[0:1] == b"\x01"


def test_root_model_list_roundtrip(allowed):
    model = RootListModel([1, 2, 3])
    result = decode(encode(model), allowed=allowed)
    assert isinstance(result, RootListModel)
    assert result.root == [1, 2, 3]


def test_root_model_scalar_roundtrip(allowed):
    model = RootScalarModel("hello")
    result = decode(encode(model), allowed=allowed)
    assert isinstance(result, RootScalarModel)
    assert result.root == "hello"


def test_dataclass_with_non_init_field_roundtrip(allowed):
    obj = DataclassWithNonInitField(name="test", value=42)
    obj.computed = "custom"
    result = decode(encode(obj), allowed=allowed)
    assert isinstance(result, DataclassWithNonInitField)
    assert result.name == "test"
    assert result.value == 42
    assert result.computed == "default"


@pytest.mark.parametrize("payload", [
    b"",
    b"\xff" + b"junk",
    _TAG_PYDANTIC + msgspec.msgpack.encode({"cls": "X"}),
    _TAG_PYDANTIC + msgspec.msgpack.encode({"module": "X"}),
    _TAG_PYDANTIC + msgspec.msgpack.encode({"module": "X", "cls": "Y", "data": {}}),
    b"\x01" + b"\xc1",
], ids=["empty", "unknown-tag", "pydantic-missing-module", "pydantic-missing-cls", "pydantic-unlisted-type", "bad-msgpack"])
def test_corrupt_payload_raises_deserialization_error(payload, allowed):
    with pytest.raises(DeserializationError):
        decode(payload, allowed=allowed)


def test_restricted_unpickler_rejects_non_numpy(allowed):
    import pickle as _pickle

    payload = b"\x80" + _pickle.dumps(AllowedPydanticModel(name="test", value=1))
    with pytest.raises(DeserializationError, match="Refusing to unpickle"):
        decode(payload, allowed=allowed)
