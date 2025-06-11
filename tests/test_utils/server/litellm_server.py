import json
import os
from typing import AsyncIterator, Iterator

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR = "LITELLM_TEST_SERVER_LOG_FILE_PATH"


class DSPyTestModel(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response(kwargs)

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response(kwargs)

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": '{"output_text": "Hello!"}',
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        return generic_streaming_chunk  # type: ignore

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": '{"output_text": "Hello!"}',
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        yield generic_streaming_chunk


def _get_mock_llm_response(request_kwargs):
    _throw_exception_based_on_content_if_applicable(request_kwargs)
    return litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        mock_response="Hi!",
    )


def _throw_exception_based_on_content_if_applicable(request_kwargs):
    """
    Throws an exception, for testing purposes, based on the content of the request message.
    """
    model = request_kwargs["model"]
    content = request_kwargs["messages"][0]["content"]
    if "429" in content:
        raise litellm.RateLimitError(message="Rate limit exceeded", llm_provider=None, model=model)
    elif "504" in content:
        raise litellm.Timeout("Request timed out!", llm_provider=None, model=model)
    elif "400" in content:
        raise litellm.BadRequestError(message="Bad request", llm_provider=None, model=model)
    elif "401" in content:
        raise litellm.AuthenticationError(message="Authentication error", llm_provider=None, model=model)


def _append_request_to_log_file(completion_kwargs):
    log_file_path = os.environ.get(LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR)
    if log_file_path is None:
        raise ValueError(
            "Server logs file path is not defined! Please set the path using the"
            + f" {LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR} environment variable."
        )

    with open(log_file_path, "a") as f:
        log_blob = (
            {
                "model": completion_kwargs["model"],
                "messages": completion_kwargs["messages"],
            },
        )
        json.dump(log_blob, f)
        f.write("\n")


dspy_test_model = DSPyTestModel()
