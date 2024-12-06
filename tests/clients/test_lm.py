from unittest import mock

import pydantic
import pytest

import dspy
from tests.test_utils.server import litellm_test_server


def test_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    openai_lm("openai query")

    azure_openai_lm = dspy.LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
    )
    azure_openai_lm("azure openai query")


def test_lm_calls_support_unhashable_types(litellm_test_server):
    api_base, server_log_file_path = litellm_test_server

    lm_with_unhashable_callable = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        # Define a callable kwarg for the LM to use during inference
        azure_ad_token_provider=lambda *args, **kwargs: None,
    )
    lm_with_unhashable_callable("Query")


def test_lm_calls_support_pydantic_models(litellm_test_server):
    api_base, server_log_file_path = litellm_test_server

    class ResponseFormat(pydantic.BaseModel):
        response: str

    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        response_format=ResponseFormat,
    )
    lm("Query")
