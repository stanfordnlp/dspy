from unittest import mock

import litellm
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
    api_base, _ = litellm_test_server

    with mock.patch("litellm.completion", autospec=True, wraps=litellm.completion) as spy_completion:
        azure_ad_token_provider = lambda *args, **kwargs: None
        lm_with_callable = dspy.LM(
            model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            azure_ad_token_provider=azure_ad_token_provider,
        )
        lm_with_callable("Query")

        spy_completion.assert_called_once()
        call_args = spy_completion.call_args.kwargs
        assert call_args["model"] == "openai/dspy-test-model"
        assert call_args["api_base"] == api_base
        assert call_args["api_key"] == "fakekey"
        assert call_args["azure_ad_token_provider"] is azure_ad_token_provider


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


@pytest.mark.parametrize(
    ("error_code", "expected_exception", "expected_num_retries"),
    [
        ("429", litellm.RateLimitError, 2),
        ("504", litellm.Timeout, 3),
        # Don't retry on user errors
        ("400", litellm.BadRequestError, 0),
        ("401", litellm.AuthenticationError, 0),
        # TODO: LiteLLM retry logic isn't implemented properly for internal server errors
        # and content policy violations, both of which may be transient and should be retried
        # ("content-policy-violation, litellm.BadRequestError, 1),
        # ("500", litellm.InternalServerError, 0, 1),
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
        num_retries=expected_num_retries,
        model_type="chat",
    )
    with pytest.raises(expected_exception):
        openai_lm(error_code)

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == expected_num_retries + 1  # 1 initial request + 1 retries


@pytest.mark.parametrize(
    ("error_code", "expected_exception", "expected_num_retries"),
    [
        ("429", litellm.RateLimitError, 2),
        ("504", litellm.Timeout, 3),
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
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        num_retries=expected_num_retries,
        model_type="text",
    )
    with pytest.raises(expected_exception):
        openai_lm(error_code)

    request_logs = read_litellm_test_server_request_logs(server_log_file_path)
    assert len(request_logs) == expected_num_retries + 1  # 1 initial request + 1 retries
