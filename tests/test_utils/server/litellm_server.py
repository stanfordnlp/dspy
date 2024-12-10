import json
import os
import time
from typing import AsyncIterator, Iterator

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR = "LITELLM_TEST_SERVER_LOG_FILE_PATH"


class DSPyTestModel(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response()

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response()

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


def _get_mock_llm_response():
    return litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
        mock_response="Hi!",
    )


def _append_request_to_log_file(completion_kwargs):
    log_file_path = os.environ.get(LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR)
    if log_file_path is None:
        raise ValueError(
            f"Server logs file path is not defined! Please set the path using the"
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
