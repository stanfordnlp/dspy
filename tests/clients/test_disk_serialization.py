"""Tests for restricted pickle deserialization."""

import io
import pickle
from dataclasses import dataclass

import diskcache
import numpy as np
import pydantic
import pytest
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from dspy.clients.disk_serialization import (
    DeserializationError,
    _restricted_load,
    restricted_disk,
)


class AllowedPydanticModel(pydantic.BaseModel):
    name: str
    value: int


@dataclass
class AllowedDataclass:
    name: str
    value: int


class _UnlistedModel(pydantic.BaseModel):
    value: int


class _Outer:
    @dataclass
    class Inner:
        value: int


def _allowed_for(*types):
    return frozenset((cls.__module__, cls.__qualname__) for cls in types)


def _roundtrip(value, allowed=None):
    """Pickle then restricted-unpickle."""
    return _restricted_load(io.BytesIO(pickle.dumps(value)), allowed or set())


# -- roundtrip tests (through diskcache) --

def _make_cache(directory, allowed):
    return diskcache.FanoutCache(
        directory=directory,
        shards=4,
        disk=restricted_disk(allowed),
        timeout=10,
    )


@pytest.mark.parametrize("value", [
    {"a": 1, "b": [2, 3]},
    [1, "two", 3.0],
    "hello",
    42,
    (1, 2, 3),
], ids=["dict", "list", "str", "int", "tuple"])
def test_plain_values_roundtrip(tmp_path, value):
    cache = _make_cache(str(tmp_path), set())
    cache["k"] = value
    assert cache["k"] == value


def test_pydantic_roundtrip(tmp_path):
    allowed = _allowed_for(AllowedPydanticModel)
    cache = _make_cache(str(tmp_path), allowed)
    cache["k"] = AllowedPydanticModel(name="test", value=42)
    assert cache["k"] == AllowedPydanticModel(name="test", value=42)


def test_dataclass_roundtrip(tmp_path):
    allowed = _allowed_for(AllowedDataclass)
    cache = _make_cache(str(tmp_path), allowed)
    cache["k"] = AllowedDataclass(name="test", value=42)
    assert cache["k"] == AllowedDataclass(name="test", value=42)


def test_ndarray_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    cache["k"] = value
    result = cache["k"]
    assert isinstance(result, np.ndarray)
    assert result.dtype == value.dtype
    np.testing.assert_array_equal(result, value)


def test_litellm_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = ModelResponse(
        id="chatcmpl-test",
        choices=[{"message": {"content": "Hello"}, "index": 0, "finish_reason": "stop"}],
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello"


def test_text_completion_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = TextCompletionResponse(
        id="cmpl-test", choices=[{"text": "Hello", "index": 0, "finish_reason": "stop"}], model="m",
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, TextCompletionResponse)
    assert result.choices[0]["text"] == "Hello"


def test_embedding_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        model="m", usage={"prompt_tokens": 1, "total_tokens": 1},
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, EmbeddingResponse)
    assert result.data[0]["embedding"] == pytest.approx([0.1, 0.2])


def test_responses_api_roundtrip(tmp_path):
    """ResponsesAPIResponse with nested openai.types.responses.* classes."""
    cache = _make_cache(str(tmp_path), set())
    response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="test",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                status="completed",
                role="assistant",
                content=[ResponseOutputText(type="output_text", text="hello", annotations=[])],
            ),
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        user=None,
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "hello"


def test_tool_call_response_roundtrip(tmp_path):
    """ModelResponse with tool calls uses nested litellm types."""
    cache = _make_cache(str(tmp_path), set())
    response = ModelResponse(
        id="chatcmpl-tool",
        choices=[{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                }],
            },
            "index": 0,
            "finish_reason": "tool_calls",
        }],
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.tool_calls[0].function.name == "get_weather"


# -- allowlist enforcement --

def test_unlisted_type_blocked_on_read(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    cache["k"] = _UnlistedModel(value=1)
    with pytest.raises(DeserializationError, match="not in the safe_types allowlist"):
        cache["k"]


def test_allowlists_are_isolated(tmp_path):
    allowed = _allowed_for(AllowedPydanticModel)
    cache = _make_cache(str(tmp_path / "a"), allowed)
    cache["k"] = AllowedPydanticModel(name="test", value=1)
    assert cache["k"] == AllowedPydanticModel(name="test", value=1)

    empty_cache = _make_cache(str(tmp_path / "b"), set())
    empty_cache["k"] = AllowedPydanticModel(name="test", value=1)
    with pytest.raises(DeserializationError):
        empty_cache["k"]


def test_numpy_ctypeslib_blocked():
    """Only specific numpy reconstruction functions are allowed, not arbitrary submodules."""
    payload = (
        b"\x80\x05"                    # PROTO 5
        b"\x8c\x0fnumpy.ctypeslib"     # SHORT_BINUNICODE (15 bytes)
        b"\x8c\x0cload_library"        # SHORT_BINUNICODE (12 bytes)
        b"\x93"                         # STACK_GLOBAL
        b"\x8c\x04evil"                 # SHORT_BINUNICODE "evil"
        b"\x8c\x04/tmp"                 # SHORT_BINUNICODE "/tmp"
        b"\x86"                         # TUPLE2
        b"R"                            # REDUCE
        b"."                            # STOP
    )
    with pytest.raises(DeserializationError):
        _restricted_load(io.BytesIO(payload), set())


# -- _restricted_load unit tests --

def test_restricted_load_rejects_unknown_type():
    data = pickle.dumps(_UnlistedModel(value=1))
    with pytest.raises(DeserializationError, match="not in the safe_types allowlist"):
        _restricted_load(io.BytesIO(data), set())


def test_corrupt_pickle_raises_deserialization_error():
    """Real pickle corruption must raise DeserializationError, not UnpicklingError."""
    data = pickle.dumps({"key": "value"})
    corrupted = data[:5] + b"nope" + data[9:]
    with pytest.raises(DeserializationError, match="Corrupt cache entry"):
        _restricted_load(io.BytesIO(corrupted), set())


def test_nested_class_safe_types(tmp_path):
    """safe_types must work for nested classes (pickle uses __qualname__)."""
    allowed = _allowed_for(_Outer.Inner)
    cache = _make_cache(str(tmp_path), allowed)
    cache["k"] = _Outer.Inner(value=42)
    assert cache["k"] == _Outer.Inner(value=42)


def test_restricted_load_allows_builtin_types():
    response = ModelResponse(
        id="t", choices=[{"message": {"content": "hi"}, "index": 0, "finish_reason": "stop"}],
    )
    result = _roundtrip(response)
    assert isinstance(result, ModelResponse)


def test_all_cached_lm_types_roundtrip():
    """Every LM return type that DSPy caches must roundtrip through the restricted unpickler."""
    test_values = [
        ModelResponse(
            id="t", choices=[{"message": {"content": "hi"}, "index": 0, "finish_reason": "stop"}],
        ),
        TextCompletionResponse(
            id="t", choices=[{"text": "hi", "index": 0, "finish_reason": "stop"}], model="m",
        ),
        EmbeddingResponse(
            data=[{"embedding": [0.1], "index": 0, "object": "embedding"}],
            model="m", usage={"prompt_tokens": 1, "total_tokens": 1},
        ),
    ]
    for value in test_values:
        result = _roundtrip(value)
        assert type(result) is type(value), f"Failed roundtrip for {type(value).__name__}"
