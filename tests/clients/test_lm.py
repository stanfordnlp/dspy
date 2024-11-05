from unittest import mock

from litellm.router import RetryPolicy

from dspy.clients.lm import LM, _get_litellm_router


def test_lm_chat_respects_max_retries():
    model_name = "openai/gpt4o"
    num_retries = 17
    temperature = 0.5
    max_tokens = 100
    prompt = "Hello, world!"

    lm = LM(
        model=model_name, model_type="chat", num_retries=num_retries, temperature=temperature, max_tokens=max_tokens
    )

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


def test_lm_completions_respects_max_retries():
    model_name = "openai/gpt-3.5-turbo"
    expected_model = "text-completion-" + model_name
    num_retries = 17
    temperature = 0.5
    max_tokens = 100
    prompt = "Hello, world!"
    api_base = "http://test.com"
    api_key = "apikey"

    lm = LM(
        model=model_name,
        model_type="text",
        num_retries=num_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
        api_key=api_key,
    )

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
            api_key=api_key,
            api_base=api_base,
            cache=mock.ANY,
        )
