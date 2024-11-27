import json
import os

import litellm
from litellm import CustomLLM

LITELLM_TEST_SERVER_LOG_FILE_PATH_ENV_VAR = "LITELLM_TEST_SERVER_LOG_FILE_PATH"


class DSPyTestModel(CustomLLM):
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response(kwargs)

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        _append_request_to_log_file(kwargs)
        return _get_mock_llm_response(kwargs)


def _get_mock_llm_response(request_kwargs):
    _throw_exception_based_on_content_if_applicable(request_kwargs)
    return litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
        mock_response="Hi!",
    )


def _throw_exception_based_on_content_if_applicable(request_kwargs):
    """
    Throws an exception, for testing purposes, based on the content of the request message.
    """
    model = request_kwargs["model"]
    content = request_kwargs["messages"][0]["content"]
    if content == "429":
        raise litellm.RateLimitError(message="Rate limit exceeded", llm_provider=None, model=model)
    elif content == "504":
        raise litellm.Timeout("Request timed out!")
    elif content == "400":
        raise litellm.BadRequestError(message="Bad request", llm_provider=None, model=model)
    elif content == "401":
        raise litellm.AuthenticationError(message="Authentication error", llm_provider=None, model=model)


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
