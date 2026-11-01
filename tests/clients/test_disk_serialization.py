"""Tests for restricted pickle deserialization."""

import pickle
import subprocess
import sys
from dataclasses import dataclass

import diskcache_community
import numpy as np
import pydantic
import pytest
from diskcache_community.core import MODE_PICKLE
from openai.types import Completion as TextCompletionResponse
from openai.types import CreateEmbeddingResponse as EmbeddingResponse
from openai.types.chat import ChatCompletion as ModelResponse
from openai.types.responses import Response as ResponsesAPIResponse
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai.types.responses import ResponseUsage as ResponseAPIUsage

from dspy.clients.disk_serialization import restricted_disk
from dspy.clients.lm15_boundary import response_value
from dspy.lm15 import Message, Request


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


def _restricted_load(directory, data, allowed=()):
    return restricted_disk(allowed)(str(directory)).fetch(MODE_PICKLE, None, data, False)


# -- roundtrip tests (through diskcache) --

def _make_cache(directory, allowed):
    return diskcache_community.FanoutCache(
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
        created=0,
        object="chat.completion",
        choices=[{
            "message": {
                "role": "assistant",
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
    assert tool_call.function.name == "get_weather"

    lm_response = response_value(response, "chat", Request(model="dummy", messages=(Message.user("-"),)))
    part = lm_response.tool_calls[0]
    assert part.name == "get_weather"
    assert part.input == {"city": "SF"}
    assert lm_response.provider_data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
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
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=2, total_tokens=3,
                               input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                               output_tokens_details={"reasoning_tokens": 0}),
        user=None,
    )
    cache.close()


def _assert_cached_responses_function_call_normalizes(directory):
    response = _make_cache(directory, set())["k"]
    function_call = response.output[0]
    assert function_call.name == "get_weather"

    lm_response = response_value(response, "responses", Request(model="dummy", messages=(Message.user("-"),)))
    part = lm_response.tool_calls[0]
    assert part.name == "get_weather"
    assert part.input == {"city": "SF"}
    assert lm_response.provider_data["output"][0]["call_id"] == "call_1"
    assert lm_response.usage.total_tokens == 3


_SUBPROCESS_CASES = {
    "write-tool-call-cache": _write_tool_call_cache,
    "assert-tool-call-cache-normalizes": _assert_cached_tool_call_normalizes,
    "write-responses-function-call-cache": _write_responses_function_call_cache,
    "assert-responses-function-call-cache-normalizes": _assert_cached_responses_function_call_normalizes,
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
    cache = _make_cache(str(tmp_path), [AllowedPydanticModel])
    cache["k"] = AllowedPydanticModel(name="test", value=42)
    assert cache["k"] == AllowedPydanticModel(name="test", value=42)


def test_dataclass_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), [AllowedDataclass])
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


def test_numpy_cannot_be_registered_as_safe_type():
    with pytest.raises(ValueError, match="with_types refuses"):
        restricted_disk([np.ndarray])


def test_openai_chat_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = ModelResponse(
        id="chatcmpl-test", model="m", created=0, object="chat.completion",
        choices=[{"message": {"role": "assistant", "content": "Hello"}, "index": 0, "finish_reason": "stop"}],
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello"


def test_text_completion_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = TextCompletionResponse(
        id="cmpl-test", created=0, object="text_completion",
        choices=[{"text": "Hello", "index": 0, "finish_reason": "stop", "logprobs": None}], model="m",
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, TextCompletionResponse)
    assert result.choices[0].text == "Hello"


def test_embedding_response_roundtrip(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    response = EmbeddingResponse(
        data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
        model="m", object="list", usage={"prompt_tokens": 1, "total_tokens": 1},
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, EmbeddingResponse)
    assert result.data[0].embedding == pytest.approx([0.1, 0.2])


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
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2,
                               input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                               output_tokens_details={"reasoning_tokens": 0}),
        user=None,
    )
    cache["k"] = response
    result = cache["k"]
    assert isinstance(result, ResponsesAPIResponse)
    assert result.output[0].content[0].text == "hello"


def test_tool_call_response_roundtrip(tmp_path):
    """ChatCompletion with tool calls uses nested OpenAI types."""
    cache = _make_cache(str(tmp_path), set())
    response = ModelResponse(
        id="chatcmpl-tool", model="m", created=0, object="chat.completion",
        choices=[{
            "message": {
                "role": "assistant",
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


def test_cached_tool_call_normalizes_after_fresh_process_roundtrip(tmp_path):
    _run_subprocess_cache_case("write-tool-call-cache", tmp_path)
    _run_subprocess_cache_case("assert-tool-call-cache-normalizes", tmp_path)


def test_cached_responses_function_call_normalizes_after_fresh_process_roundtrip(tmp_path):
    _run_subprocess_cache_case("write-responses-function-call-cache", tmp_path)
    _run_subprocess_cache_case("assert-responses-function-call-cache-normalizes", tmp_path)


# -- allowlist enforcement --

def test_unlisted_type_refused_on_write(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    with pytest.raises(TypeError, match="SafeDisk cannot store"):
        cache["k"] = _UnlistedModel(value=1)


def test_allowlists_are_isolated(tmp_path):
    cache = _make_cache(str(tmp_path / "a"), [AllowedPydanticModel])
    cache["k"] = AllowedPydanticModel(name="test", value=1)
    assert cache["k"] == AllowedPydanticModel(name="test", value=1)

    empty_cache = _make_cache(str(tmp_path / "b"), set())
    with pytest.raises(TypeError, match="SafeDisk cannot store"):
        empty_cache["k"] = AllowedPydanticModel(name="test", value=1)


def test_numpy_ctypeslib_blocked(tmp_path):
    """numpy functions must not resolve from a pickled entry."""
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
    with pytest.raises(pickle.UnpicklingError, match="SafeDisk refuses"):
        _restricted_load(tmp_path, payload)


# -- _restricted_load unit tests --

def test_restricted_load_rejects_unknown_type(tmp_path):
    data = pickle.dumps(_UnlistedModel(value=1))
    with pytest.raises(pickle.UnpicklingError, match="SafeDisk refuses"):
        _restricted_load(tmp_path, data)


def test_nested_class_safe_types(tmp_path):
    """safe_types must work for nested classes (pickle uses __qualname__)."""
    cache = _make_cache(str(tmp_path), [_Outer.Inner])
    cache["k"] = _Outer.Inner(value=42)
    assert cache["k"] == _Outer.Inner(value=42)


def test_all_cached_lm_types_roundtrip(tmp_path):
    """Every LM return type that DSPy caches must roundtrip through the restricted unpickler."""
    test_values = [
        ModelResponse(
            id="t", model="m", created=0, object="chat.completion",
            choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
        ),
        TextCompletionResponse(
            id="t", created=0, object="text_completion",
            choices=[{"text": "hi", "index": 0, "finish_reason": "stop", "logprobs": None}], model="m",
        ),
        EmbeddingResponse(
            data=[{"embedding": [0.1], "index": 0, "object": "embedding"}],
            model="m", object="list", usage={"prompt_tokens": 1, "total_tokens": 1},
        ),
    ]
    for value in test_values:
        result = _restricted_load(tmp_path, pickle.dumps(value))
        assert type(result) is type(value), f"Failed roundtrip for {type(value).__name__}"


if __name__ == "__main__":
    _SUBPROCESS_CASES[sys.argv[1]](sys.argv[2])
