from unittest import mock

import litellm
import pytest

import dspy
from tests.test_utils.server import litellm_test_server, read_litellm_test_server_request_logs


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


@pytest.mark.parametrize(
    ("error_code", "expected_exception", "expected_num_retries"),
    [
        ("429", litellm.RateLimitError, 2),
        ("504", litellm.Timeout, 2),
        # Don't retry on user errors
        ("400", litellm.BadRequestError, 0),
        ("401", litellm.AuthenticationError, 0),
        # TODO: LiteLLM retry logic isn't implemented properly for internal server errors
        # and content policy violations, both of which may be transient and should be retried
        # ("content-policy-violation, litellm.BadRequestError, 2),
        # ("500", litellm.InternalServerError, 0, 2),
    ],
)
def test_lm_chat_calls_are_retried_for_expected_failures(
    litellm_test_server,
    error_code,
    expected_exception,
    expected_num_retries,
):
    api_base, server_log_file_path = litellm_test_server

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        num_retries=1,
        model_type="chat",
    )
    with pytest.raises(expected_exception):
        openai_lm(error_code)

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == expected_num_retries


@pytest.mark.parametrize(
    ("error_code", "expected_exception", "expected_num_retries"),
    [
        ("429", litellm.RateLimitError, 2),
        ("504", litellm.Timeout, 2),
        # Don't retry on user errors
        ("400", litellm.BadRequestError, 0),
        ("401", litellm.AuthenticationError, 0),
        # TODO: LiteLLM retry logic isn't implemented properly for internal server errors
        # and content policy violations, both of which may be transient and should be retried
        # ("content-policy-violation, litellm.BadRequestError, 2),
        # ("500", litellm.InternalServerError, 0, 2),
    ],
)
def test_lm_text_calls_are_retried_for_expected_failures(
    litellm_test_server,
    error_code,
    expected_exception,
    expected_num_retries,
):
    api_base, server_log_file_path = litellm_test_server

    openai_lm = dspy.LM(
        model="openai/dspy-test-text-model",
        api_base=api_base,
        api_key="fakekey",
        num_retries=2,
        model_type="text",
    )
    with pytest.raises(expected_exception):
        openai_lm(error_code)

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == expected_num_retries  # 1 initial request + 2 retries
