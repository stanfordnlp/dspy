from unittest import mock

from dspy.clients.lm import LM


def test_lm_chat_respects_max_retries():
    lm = LM(model="openai/gpt4o", model_type="chat", max_retries=17)

    with mock.patch("dspy.clients.lm.litellm.completion") as litellm_completion_api:
        lm(messages=[{"content": "Hello, world!", "role": "user"}])

    assert litellm_completion_api.call_count == 1
    assert litellm_completion_api.call_args[1]["max_retries"] == 17
    assert litellm_completion_api.call_args[1]["retry_strategy"] == "exponential_backoff_retry"


def test_lm_completions_respects_max_retries():
    lm = LM(model="openai/gpt-3.5-turbo", model_type="completions", max_retries=17)

    with mock.patch("dspy.clients.lm.litellm.text_completion") as litellm_completion_api:
        lm(prompt="Hello, world!")

    assert litellm_completion_api.call_count == 1
    assert litellm_completion_api.call_args[1]["max_retries"] == 17
    assert litellm_completion_api.call_args[1]["retry_strategy"] == "exponential_backoff_retry"
