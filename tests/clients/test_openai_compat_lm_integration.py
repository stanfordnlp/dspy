"""Local HTTP integration tests for the `_OpenAICompatLM` engine."""

import pytest

import dspy
from dspy.clients.openai_compat_lm import _OpenAICompatLM


def _completion(text):
    return {
        "id": "chatcmpl-local",
        "model": "local-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def test_openai_compat_lm_drives_predict_over_http(openai_compat_server):
    base_url, server = openai_compat_server
    server.reply(200, _completion("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"))
    lm = _OpenAICompatLM(
        "local-model",
        base_url,
        api_key="local",
        temperature=0.2,
        cache=False,
        num_retries=0,
    )
    dspy.configure(lm=lm)

    result = dspy.Predict("question -> answer")(question="What is the capital of France?")

    assert result.answer == "Paris"
    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["model"] == "local-model"
    assert request["body"]["temperature"] == 0.2
    assert "capital of France" in request["body"]["messages"][-1]["content"]
    assert request["headers"]["Authorization"] == "Bearer local"


def test_openai_compat_lm_maps_rate_limit_over_http(openai_compat_server):
    base_url, server = openai_compat_server
    server.reply(429, {"error": {"code": "rate_limit_exceeded", "message": "slow down"}})
    lm = _OpenAICompatLM("local-model", base_url, cache=False, num_retries=0)

    with pytest.raises(dspy.LMRateLimitError, match="slow down"):
        lm("429")


def test_openai_compat_lm_maps_context_window_over_http(openai_compat_server):
    base_url, server = openai_compat_server
    server.reply(400, {"error": {"code": "context_length_exceeded", "message": "prompt too long"}})
    lm = _OpenAICompatLM("local-model", base_url, cache=False, num_retries=0)

    with pytest.raises(dspy.ContextWindowExceededError, match="prompt too long"):
        lm("too long")
