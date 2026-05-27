"""Tests for restricted pickle deserialization."""

import io
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import diskcache
import numpy as np
import pydantic
import pytest
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from dspy.clients.disk_serialization import (
    DeserializationError,
    _restricted_load,
    restricted_disk,
)
from dspy.clients.openai_format import completion_to_lm_response, responses_to_lm_response
from dspy.core.types import LMRequest


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


def _run_subprocess_cache_case(case, directory):
    subprocess.run(
        [sys.executable, "-m", "tests.clients.test_disk_serialization", case, str(directory)],
        check=True,
    )


def _write_tool_call_cache(directory):
    cache = _make_cache(directory, set())
    cache["k"] = ModelResponse(
        id="chatcmpl-tool",
        model="dummy",
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
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    cache.close()


def _assert_cached_tool_call_normalizes(directory):
    response = _make_cache(directory, set())["k"]
    tool_call = response.choices[0].message.tool_calls[0]
    assert type(type(tool_call).__pydantic_serializer__).__name__ == "MockValSer"
    assert type(type(tool_call.function).__pydantic_serializer__).__name__ == "MockValSer"

    lm_response = completion_to_lm_response(response, LMRequest(model="dummy", messages=[]))
    part = lm_response.outputs[0].tool_calls[0]
    assert part.name == "get_weather"
    assert part.args == {"city": "SF"}
    assert part.provider_data["function"]["name"] == "get_weather"
    assert lm_response.usage.total_tokens == 3


def _write_responses_function_call_cache(directory):
    cache = _make_cache(directory, set())
    cache["k"] = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="dummy",
        object="response",
        output=[ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            name="get_weather",
            arguments='{"city":"SF"}',
            status="completed",
        )],
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
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        user=None,
    )
    cache.close()


def _assert_cached_responses_function_call_normalizes(directory):
    response = _make_cache(directory, set())["k"]
    function_call = response.output[0]
    assert type(type(function_call).__pydantic_serializer__).__name__ == "MockValSer"

    lm_response = responses_to_lm_response(response, LMRequest(model="dummy", messages=[]))
    part = lm_response.outputs[0].tool_calls[0]
    assert part.name == "get_weather"
    assert part.args == {"city": "SF"}
    assert part.provider_data["call_id"] == "call_1"
    assert lm_response.usage.total_tokens == 3


def _write_normalization_caches(directory):
    directory = Path(directory)
    _write_tool_call_cache(str(directory / "tool_call"))
    _write_responses_function_call_cache(str(directory / "responses_function_call"))


def _assert_normalization_caches(directory):
    directory = Path(directory)
    _assert_cached_tool_call_normalizes(str(directory / "tool_call"))
    _assert_cached_responses_function_call_normalizes(str(directory / "responses_function_call"))


_SUBPROCESS_CASES = {
    "write-normalization-caches": _write_normalization_caches,
    "assert-normalization-caches": _assert_normalization_caches,
}


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


def test_cached_lm_responses_normalize_after_fresh_process_roundtrip(tmp_path):
    _run_subprocess_cache_case("write-normalization-caches", tmp_path)
    _run_subprocess_cache_case("assert-normalization-caches", tmp_path)


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


if __name__ == "__main__":
    _SUBPROCESS_CASES[sys.argv[1]](sys.argv[2])
