"""Tests for restricted pickle deserialization."""

import importlib
import io
import pickle
import sqlite3
import subprocess
import sys
from dataclasses import dataclass

import diskcache
import numpy as np
import pydantic
import pytest
from openai.types import Completion as TextCompletionResponse
from openai.types import CreateEmbeddingResponse as EmbeddingResponse
from openai.types.chat import ChatCompletion as ModelResponse
from openai.types.responses import Response as ResponsesAPIResponse
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from openai.types.responses import ResponseUsage as ResponseAPIUsage

from dspy.clients.cache import Cache
from dspy.clients.disk_serialization import (
    DeserializationError,
    _restricted_load,
    restricted_disk,
)
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


def test_file_backed_ndarray_roundtrip(tmp_path):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()), disk_min_file_size=1)
    value = np.arange(1200, dtype=np.float64).reshape(30, 40)
    cache["array"] = value
    cache.close()

    reopened = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()))
    np.testing.assert_array_equal(reopened["array"], value)


def test_dspy_cache_array_roundtrip_after_reopen(tmp_path):
    options = {"enable_disk_cache": True, "enable_memory_cache": False, "disk_cache_dir": str(tmp_path)}
    request = {"model": "test", "prompt": "embedding"}
    value = np.arange(1200, dtype=np.float32).reshape(30, 40)
    cache = Cache(**options)
    cache.put(request, value)
    cache.disk_cache.close()

    reopened = Cache(**options)
    np.testing.assert_array_equal(reopened.get(request), value)


def test_dspy_cache_nested_arrays_roundtrip_after_reopen(tmp_path):
    options = {"enable_disk_cache": True, "enable_memory_cache": False, "disk_cache_dir": str(tmp_path)}
    request = {"model": "test", "prompt": "nested embedding"}
    cache = Cache(**options)
    cache.put(request, {"embeddings": [np.array([1, 2], dtype=np.int32), {"second": np.array([3.5])}]})
    cache.disk_cache.close()

    reopened = Cache(**options)
    result = reopened.get(request)
    np.testing.assert_array_equal(result["embeddings"][0], np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(result["embeddings"][1]["second"], np.array([3.5]))


def test_numpy_subclass_is_not_silently_flattened(tmp_path):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()))
    value = np.ma.array([1, 2], mask=[True, False])
    with pytest.raises(TypeError, match="subclasses cannot be safely serialized"):
        cache["masked"] = value
    with pytest.raises(TypeError, match="subclasses cannot be safely serialized"):
        cache["nested"] = {"masked": value}
    assert "masked" not in cache
    assert "nested" not in cache


def test_file_backed_value_with_symlinked_cache_root(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    (tmp_path / "alias").symlink_to(root, target_is_directory=True)
    cache = diskcache.Cache(tmp_path / "alias", disk=restricted_disk(_allowed_for(AllowedDataclass)), disk_min_file_size=1)
    cache["key"] = b"data"
    cache["text"] = "hello\rworld"
    cache["model"] = AllowedDataclass(name="hello", value=7)
    assert cache["key"] == b"data"
    assert cache["text"] == "hello\nworld"
    assert cache["model"] == AllowedDataclass(name="hello", value=7)
    cache.delete("key")
    assert "key" not in cache


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

def test_direct_callable_value_remains_blocked():
    payload = b"cbuiltins\neval\n(S'2 + 3'\ntR."
    with pytest.raises(DeserializationError):
        _restricted_load(io.BytesIO(payload), set())


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


def test_trusted_module_alias_cannot_resolve_an_untrusted_callable(monkeypatch):
    module = importlib.import_module("openai.types.chat.chat_completion")
    monkeypatch.setattr(module, "injected_alias", eval, raising=False)
    payload = b"copenai.types.chat.chat_completion\ninjected_alias\n."

    with pytest.raises(DeserializationError):
        _restricted_load(io.BytesIO(payload), set())


def test_pickled_keys_use_the_same_allowlist(tmp_path):
    cache = _make_cache(str(tmp_path), set())
    cache[_Outer.Inner(value=7)] = "stored"

    with pytest.raises(DeserializationError):
        list(cache)

    allowed_cache = _make_cache(str(tmp_path), _allowed_for(_Outer.Inner))
    assert list(allowed_cache) == [_Outer.Inner(value=7)]


def test_legacy_pickled_numpy_array_is_a_miss():
    # Array BUILD state is not validated by pickle's find_class hook.
    with pytest.raises(DeserializationError):
        _restricted_load(io.BytesIO(pickle.dumps(np.arange(3))), set())


def test_unknown_persistent_id_is_rejected():
    class UnknownIdPickler(pickle.Pickler):
        def persistent_id(self, value):
            return "unknown" if value == "placeholder" else None

    buffer = io.BytesIO()
    UnknownIdPickler(buffer).dump(["placeholder"])
    with pytest.raises(DeserializationError, match="Invalid cache array reference"):
        _restricted_load(io.BytesIO(buffer.getvalue()), set())


def test_array_reference_rejects_non_array_data():
    archive = io.BytesIO()
    np.savez(archive, values=np.arange(3))

    class ArchivePickler(pickle.Pickler):
        def persistent_id(self, value):
            return (b"\x93NUMPY", archive.getvalue()) if value == "placeholder" else None

    buffer = io.BytesIO()
    ArchivePickler(buffer).dump(["placeholder"])
    with pytest.raises(DeserializationError, match="Invalid cache array reference"):
        _restricted_load(io.BytesIO(buffer.getvalue()), set())


@pytest.mark.parametrize("cleanup", ["delete", "cull"])
def test_tampered_filename_cannot_delete_outside_cache(tmp_path, cleanup):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()), disk_min_file_size=1)
    cache.set("old", b"data")
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    with sqlite3.connect(tmp_path / "cache" / "cache.db") as db:
        db.execute(
            "UPDATE Cache SET filename = ?, expire_time = ? WHERE key = 'old'",
            ("../outside", 0 if cleanup == "cull" else None),
        )

    if cleanup == "delete":
        with pytest.raises(DeserializationError):
            cache["old"]
        cache.delete("old")
        assert "old" not in cache
    else:
        cache.set("new", b"data")
        assert cache["new"] == b"data"

    assert outside.read_bytes() == b"keep"


def test_file_backed_cache_reads_do_not_follow_symlinks(tmp_path):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()), disk_min_file_size=1)
    cache.set("key", b"data")
    with sqlite3.connect(tmp_path / "cache" / "cache.db") as db:
        filename = db.execute("SELECT filename FROM Cache WHERE key = 'key'").fetchone()[0]
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    path = tmp_path / "cache" / filename
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(DeserializationError):
        cache["key"]
    cache.delete("key")
    assert outside.read_bytes() == b"secret"


def test_file_backed_cache_cleanup_does_not_follow_directory_symlinks(tmp_path):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()), disk_min_file_size=1)
    cache.set("key", b"data")
    with sqlite3.connect(tmp_path / "cache" / "cache.db") as db:
        filename = db.execute("SELECT filename FROM Cache WHERE key = 'key'").fetchone()[0]
    path = tmp_path / "cache" / filename
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / path.name).write_bytes(b"secret")
    path.unlink()
    path.parent.rmdir()
    path.parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DeserializationError):
        cache["key"]
    cache.delete("key")
    assert (outside / path.name).read_bytes() == b"secret"


def test_file_backed_cache_writes_do_not_follow_directory_symlinks(tmp_path, monkeypatch):
    cache = diskcache.Cache(tmp_path / "cache", disk=restricted_disk(set()), disk_min_file_size=1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "cache" / "ab").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr("diskcache.core.os.urandom", lambda size: b"\xab" * size)

    with pytest.raises(OSError):
        cache["key"] = b"data"
    assert list(outside.iterdir()) == []


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
        id="t", model="m", created=0, object="chat.completion",
        choices=[{"message": {"role": "assistant", "content": "hi"}, "index": 0, "finish_reason": "stop"}],
    )
    result = _roundtrip(response)
    assert isinstance(result, ModelResponse)


def test_all_cached_lm_types_roundtrip():
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
        result = _roundtrip(value)
        assert type(result) is type(value), f"Failed roundtrip for {type(value).__name__}"


if __name__ == "__main__":
    _SUBPROCESS_CASES[sys.argv[1]](sys.argv[2])
