"""Metadata loading, native capability decisions, and provider token accounting."""

import json
from unittest.mock import patch

import pytest

import dspy
import dspy.clients.model_metadata as metadata
from dspy.clients.costs import estimate_cost
from dspy.lm15 import Config, Message, Request, Response, Usage


@pytest.fixture(autouse=True)
def local_metadata(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(metadata, "_data", None)
    monkeypatch.setattr(metadata, "_source", {})


def test_snapshot_capabilities_do_not_import_litellm():
    with patch("dspy.clients.lm._get_litellm", side_effect=AssertionError("LiteLLM accessed")):
        lm = dspy.LM("openai/gpt-4o", engine="lm15")
        assert lm.supports_function_calling
        assert lm.supports_response_schema
        assert "response_format" in lm.supported_params
        assert not lm.supports_reasoning
        claude = dspy.LM("anthropic/claude-sonnet-4-5", engine="lm15")
        assert claude.supports_reasoning
        assert claude.supports_response_schema
        assert metadata.source_info()["is_env_forced"]


def test_unknown_and_custom_capabilities():
    assert not dspy.LM("openai/unlisted").supports_function_calling

    class Engine:
        supports_function_calling = True
        supported_params = {"tools"}

        def complete(self, request):
            raise AssertionError("No generation during capability lookup")

    lm = dspy.LM("custom", engine=Engine())
    assert lm.supports_function_calling
    assert lm.supported_params == {"tools"}


def test_lm15_restriction_beats_catalog(monkeypatch):
    monkeypatch.setattr(metadata, "_data", {"zai/glm-test": {
        "litellm_provider": "zai", "supports_response_schema": True,
        "supports_function_calling": True,
    }})
    lm = dspy.LM("zai:glm-test", engine="lm15")
    assert not lm.supports_response_schema
    assert "response_format" in lm.supported_params  # JSON mode, not schema enforcement


def test_missing_flags_inherit_but_false_and_prices_do_not(monkeypatch):
    monkeypatch.setattr(metadata, "_data", {
        "openai/test": {"supports_reasoning": False},
        "test": {"litellm_provider": "openai", "supports_reasoning": True,
                 "supports_function_calling": True, "input_cost_per_token": 9},
    })
    info = metadata.model_info("openai-chat", "test")
    assert info["supports_reasoning"] is False
    assert info["supports_function_calling"] is True
    assert "input_cost_per_token" not in info
    info["supports_reasoning"] = True
    assert metadata.model_info("openai-chat", "test")["supports_reasoning"] is False


def test_remote_first_once_and_fallback(monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP")
    with patch.object(metadata.urllib.request, "urlopen", side_effect=OSError("secret")) as fetch:
        metadata.model_info("openai-chat", "gpt-4o")
        metadata.model_info("openai-chat", "gpt-4o")
    assert fetch.call_count == 1
    assert metadata.source_info()["source"] == "local"
    assert metadata.source_info()["fallback_reason"] == "OSError"


def test_valid_remote_map_is_adopted(monkeypatch):
    import io

    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP")
    data = metadata._snapshot()
    data["gpt-4o"]["supports_reasoning"] = True
    with patch.object(metadata.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(data).encode())):
        assert metadata.model_info("openai-chat", "gpt-4o")["supports_reasoning"]
    assert metadata.source_info()["source"] == "remote"


def answer(model, **usage):
    return Response(id=None, model=model, message=Message.assistant("ok"),
                    finish_reason="stop", usage=Usage(**usage))


def test_openai_cached_input_and_reasoning_are_not_double_counted():
    response = answer("gpt-4o", input_tokens=100, output_tokens=30, cache_read_tokens=40, reasoning_tokens=10)
    cost, details = estimate_cost(response, provider="openai-chat", requested_model=response.model)
    assert cost == pytest.approx(60 * 2.5e-6 + 40 * 1.25e-6 + 30 * 1e-5)
    assert details["kind"] == "estimate"


def test_anthropic_disjoint_cache_counters():
    response = answer("claude-sonnet-4-5", input_tokens=100, output_tokens=30,
                      cache_read_tokens=40, cache_write_tokens=20)
    cost, _ = estimate_cost(response, provider="anthropic", requested_model=response.model)
    assert cost == pytest.approx(100 * 3e-6 + 40 * 3e-7 + 20 * 3.75e-6 + 30 * 1.5e-5)


def test_gemini_reasoning_is_additional_output():
    response = answer("gemini-2.5-flash", input_tokens=100, output_tokens=30, reasoning_tokens=10)
    cost, _ = estimate_cost(response, provider="gemini", requested_model=response.model)
    assert cost == pytest.approx(100 * 3e-7 + 40 * 2.5e-6)


def test_long_context_and_priority_rates(monkeypatch):
    monkeypatch.setattr(metadata, "_data", {"openai/test": {
        "input_cost_per_token_above_200k_tokens_priority": 2e-6,
        "output_cost_per_token_above_200k_tokens_priority": 8e-6,
    }})
    response = answer("test", input_tokens=200001, output_tokens=30, reasoning_tokens=10)
    request = Request(model="test", messages=(Message.user("hi"),), config=Config(service_tier="priority"))
    cost, _ = estimate_cost(response, provider="openai-chat", requested_model="test", request=request)
    assert cost == pytest.approx(200001 * 2e-6 + 30 * 8e-6)


def test_unknown_price_and_malformed_metadata_do_not_fail_generation(monkeypatch):
    response = answer("unknown", input_tokens=1, output_tokens=1)
    assert estimate_cost(response, provider="openai-chat", requested_model="unknown")[0] is None
    monkeypatch.setattr("dspy.clients.costs.model_info", lambda *args: {"broken": object()})
    assert estimate_cost(response, provider="openai-chat", requested_model="unknown")[0] is None


def test_cost_provenance_survives_cache_record():
    from dspy.clients.call_result import CallResult, combine

    response = answer("gpt-4o", input_tokens=1, output_tokens=1)
    result = CallResult.native(response)
    result.cost, result.cost_details = estimate_cost(response, provider="openai-chat", requested_model=response.model)
    cached = CallResult.load(result.dump())
    assert cached.cost == result.cost
    assert cached.cost_details == result.cost_details
    assert cached.usage == {}
    assert combine([result, result], model_type="chat").cost == result.cost * 2
