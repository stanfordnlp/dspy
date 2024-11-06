from unittest import mock

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
