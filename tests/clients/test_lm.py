from unittest import mock

import pytest
from litellm.router import RetryPolicy

from dspy.clients.lm import LM, _get_litellm_router


@pytest.mark.parametrize("keys_in_env_vars", [True, False])
def test_lm_chat_respects_max_retries(keys_in_env_vars, monkeypatch):
    model_name = "openai/gpt4o"
    num_retries = 17
    temperature = 0.5
    max_tokens = 100
    prompt = "Hello, world!"
    api_version = "2024-02-01"
    api_key = "apikey"

    lm_kwargs = {
        "model": model_name,
        "model_type": "chat",
        "num_retries": num_retries,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if keys_in_env_vars:
        api_base = "http://testfromenv.com"
        monkeypatch.setenv("OPENAI_API_KEY", api_key)
        monkeypatch.setenv("OPENAI_API_BASE", api_base)
        monkeypatch.setenv("OPENAI_API_VERSION", api_version)
    else:
        api_base = "http://test.com"
        lm_kwargs["api_key"] = api_key
        lm_kwargs["api_base"] = api_base
        lm_kwargs["api_version"] = api_version

    lm = LM(**lm_kwargs)

    MockRouter = mock.MagicMock()
    mock_completion = mock.MagicMock()
    MockRouter.completion = mock_completion

    with mock.patch("dspy.clients.lm.Router", return_value=MockRouter) as MockRouterConstructor:
        lm(prompt=prompt)

        MockRouterConstructor.assert_called_once_with(
            model_list=[
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": model_name,
                        "api_key": api_key,
                        "api_base": api_base,
                        "api_version": api_version,
                    },
                }
            ],
            retry_policy=RetryPolicy(
                TimeoutErrorRetries=num_retries,
                RateLimitErrorRetries=num_retries,
                InternalServerErrorRetries=num_retries,
                BadRequestErrorRetries=0,
                AuthenticationErrorRetries=0,
                ContentPolicyViolationErrorRetries=0,
            ),
        )
        mock_completion.assert_called_once_with(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            cache=mock.ANY,
        )


@pytest.mark.parametrize("keys_in_env_vars", [True, False])
def test_lm_completions_respects_max_retries(keys_in_env_vars, monkeypatch):
    model_name = "azure/gpt-3.5-turbo"
    expected_model = "text-completion-openai/" + model_name.split("/")[-1]
    num_retries = 17
    temperature = 0.5
    max_tokens = 100
    prompt = "Hello, world!"
    api_version = "2024-02-01"
    api_key = "apikey"
    azure_ad_token = "adtoken"

    lm_kwargs = {
        "model": model_name,
        "model_type": "text",
        "num_retries": num_retries,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if keys_in_env_vars:
        api_base = "http://testfromenv.com"
        monkeypatch.setenv("AZURE_API_KEY", api_key)
        monkeypatch.setenv("AZURE_API_BASE", api_base)
        monkeypatch.setenv("AZURE_API_VERSION", api_version)
        monkeypatch.setenv("AZURE_AD_TOKEN", azure_ad_token)
    else:
        api_base = "http://test.com"
        lm_kwargs["api_key"] = api_key
        lm_kwargs["api_base"] = api_base
        lm_kwargs["api_version"] = api_version
        lm_kwargs["azure_ad_token"] = azure_ad_token

    lm = LM(**lm_kwargs)

    MockRouter = mock.MagicMock()
    mock_text_completion = mock.MagicMock()
    MockRouter.text_completion = mock_text_completion

    with mock.patch("dspy.clients.lm.Router", return_value=MockRouter) as MockRouterConstructor:
        lm(prompt=prompt)

        MockRouterConstructor.assert_called_once_with(
            model_list=[
                {
                    "model_name": expected_model,
                    "litellm_params": {
                        "model": expected_model,
                        "api_key": api_key,
                        "api_base": api_base,
                        "api_version": api_version,
                        "azure_ad_token": azure_ad_token,
                    },
                }
            ],
            retry_policy=RetryPolicy(
                TimeoutErrorRetries=num_retries,
                RateLimitErrorRetries=num_retries,
                InternalServerErrorRetries=num_retries,
                BadRequestErrorRetries=0,
                AuthenticationErrorRetries=0,
                ContentPolicyViolationErrorRetries=0,
            ),
        )
        mock_text_completion.assert_called_once_with(
            model=expected_model,
            prompt=prompt + "\n\nBEGIN RESPONSE:",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=mock.ANY,
        )
