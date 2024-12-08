from unittest import mock

import pydantic
import pytest

import dspy
from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs


def test_chat_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert azure_openai_lm("azure openai query") == expected_response


def test_text_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert azure_openai_lm("azure openai query") == expected_response


def test_lm_calls_support_callables(litellm_test_server):
    api_base, server_log_file_path = litellm_test_server

    lm_with_callable = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        # Define a callable kwarg for the LM to use during inference
        azure_ad_token_provider=lambda *args, **kwargs: None,
    )
    # Invoke the LM twice; the second call should be cached in memory
    lm_with_callable("Query")
    lm_with_callable("Query")

    # Define and invoke a nearly-identical LM that lacks the callable kwarg
    lm_without_callable = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    lm_without_callable("Query")

    # Verify that 2 requests were made to the LiteLLM server - one for each LM.
    # This verifies that there wasn't a cache collision between the LMs due to
    # the callable
    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == 2


def test_lm_calls_support_pydantic_models(litellm_test_server):
    api_base, _ = litellm_test_server

    class ResponseFormat(pydantic.BaseModel):
        response: str

    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        response_format=ResponseFormat,
    )
    lm("Query")
