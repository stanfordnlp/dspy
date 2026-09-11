"""Capabilities must follow the configuration-based execution backend."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients import capabilities as hints
from dspy.clients.call_result import CallResult
from dspy.clients.engines import AsyncLiteLLMEngine, AsyncLM15Engine, LiteLLMEngine, LM15Engine
from dspy.clients.execution import _engine, prepare
from dspy.lm15 import Message, Response, Usage


@pytest.fixture
def conflicting_metadata(monkeypatch):
    calls = []

    def capability(name, result):
        def lookup(*args, **kwargs):
            calls.append((name, args, kwargs, threading.get_ident()))
            return result
        return lookup

    litellm = SimpleNamespace(
        supports_function_calling=capability("tools", False),
        supports_reasoning=capability("reasoning", False),
        supports_response_schema=capability("schema", False),
        get_supported_openai_params=capability("params", []),
    )
    monkeypatch.setattr("dspy.clients.lm._get_litellm", lambda: litellm)
    monkeypatch.setattr(hints, "model_info", lambda *args: {
        "supports_function_calling": True, "supports_reasoning": True, "supports_response_schema": True,
    })
    monkeypatch.setattr("dspy.clients.costs.estimate_cost", lambda *args, **kwargs: (None, {}))
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "AZURE_API_BASE", "AZURE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return calls


@pytest.mark.parametrize("model,options", [
    ("openai/gpt-4o", {"custom_llm_provider": "alternate"}),
    ("openai/gpt-4o", {"headers": {"x-test": "yes"}}),
    ("openai/gpt-4o", {"timeout": 10}),
    ("azure/gpt-4o", {"api_base": "https://example.invalid"}),
    ("unmapped/model", {}),
    ("openai/gpt-4o", {"model_type": "text"}),
])
def test_auto_compatibility_settings_use_litellm_capabilities(model, options, conflicting_metadata):
    lm = dspy.LM(model, cache=False, **options)
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    try:
        assert isinstance(backend, LiteLLMEngine)
        assert not lm.supports_function_calling
        assert not lm.supports_reasoning
        assert not lm.supports_response_schema
        assert lm.supported_params == set()
        assert conflicting_metadata
        if options.get("custom_llm_provider"):
            assert all(c[2]["custom_llm_provider"] == "alternate" for c in conflicting_metadata)
    finally:
        lm.close()


@pytest.mark.parametrize("variable", ["OPENAI_API_BASE", "OPENAI_BASE_URL"])
def test_environment_gateway_uses_same_backend_for_capabilities(variable, conflicting_metadata, monkeypatch):
    monkeypatch.setenv(variable, "https://gateway.invalid")
    lm = dspy.LM("openai/gpt-4o", cache=False)
    assert isinstance(_engine(lm, prepare(lm, "hello", None, {}), False)[0], LiteLLMEngine)
    assert not lm.supports_function_calling
    assert not lm.supports_response_schema
    assert conflicting_metadata


def test_native_compatible_settings_do_not_consult_litellm(conflicting_metadata):
    lm = dspy.LM("openai/gpt-4o", api_key="fake", api_base="https://example.invalid", cache=False)
    try:
        assert isinstance(_engine(lm, prepare(lm, "hello", None, {}), False)[0], LM15Engine)
        assert lm.supports_function_calling
        assert lm.supports_reasoning
        assert lm.supports_response_schema
        assert "response_format" in lm.supported_params
        assert not conflicting_metadata
    finally:
        lm.close()


def test_copy_recomputes_effective_backend(conflicting_metadata):
    lm = dspy.LM("openai/gpt-4o")
    assert lm.supports_response_schema
    assert not lm.copy(custom_llm_provider="alternate").supports_response_schema
    assert lm.supports_response_schema


def test_forced_native_configuration_error_matches_execution(conflicting_metadata):
    lm = dspy.LM("openai/gpt-4o", engine="lm15", headers={"x-test": "yes"})
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        _engine(lm, prepare(lm, "hello", None, {}), False)
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        _ = lm.supports_response_schema
    assert not conflicting_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_json_adapter_honours_call_time_backend(asynchronous, conflicting_metadata, monkeypatch):
    requests = []

    def complete(self, lm, request, **context):
        requests.append(request)
        assert request["custom_llm_provider"] == "alternate"
        assert "response_format" not in request
        return CallResult(outputs=['{"answer":"compatible"}'], response_model=lm.model)

    async def acomplete(self, lm, request, **context):
        await asyncio.sleep(0)
        return complete(self, lm, request, **context)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", complete)
    monkeypatch.setattr(AsyncLiteLLMEngine, "complete_legacy", acomplete)
    lm = dspy.LM("openai/gpt-4o", cache=False)
    signature = dspy.Signature("question -> answer")
    kwargs = {"custom_llm_provider": "alternate"}
    args = (lm, kwargs, signature, [], {"question": "hello"})
    adapter = dspy.JSONAdapter()
    thread = threading.get_ident()
    result = await adapter.acall(*args) if asynchronous else adapter(*args)
    assert result == [{"answer": "compatible"}]
    assert len(requests) == 1
    if asynchronous:
        assert all(c[3] != thread for c in conflicting_metadata)
    assert "custom_llm_provider" not in lm.kwargs
    assert lm.supports_response_schema  # planning override did not leak


def test_chat_adapter_does_not_enable_unsupported_tools(conflicting_metadata, monkeypatch):
    def lookup(query: str) -> str:
        return query

    signature = dspy.Signature({
        "tools": (list[dspy.Tool], dspy.InputField()),
        "calls": (dspy.ToolCalls, dspy.OutputField()),
    })

    def complete(self, lm, request, **context):
        assert "tools" not in request
        assert "tool_choice" not in request
        return CallResult(outputs=['[[ ## calls ## ]]\n{"tool_calls": []}'], response_model=lm.model)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", complete)
    lm = dspy.LM("openai/gpt-4o", cache=False)
    result = dspy.ChatAdapter(use_native_function_calling=True, use_json_adapter_fallback=False)(
        lm, {"custom_llm_provider": "alternate"}, signature, [], {"tools": [dspy.Tool(lookup)]},
    )
    assert result[0]["calls"].tool_calls == []


@pytest.mark.asyncio
async def test_concurrent_planning_scopes_are_isolated(conflicting_metadata, monkeypatch):
    async def native(self, request):
        await asyncio.sleep(0)
        assert request.config.response_format is not None
        return Response(None, request.model, Message.assistant('{"answer":"native"}'), "stop", Usage())

    async def compatibility(self, lm, request, **context):
        await asyncio.sleep(0)
        assert "response_format" not in request
        return CallResult(outputs=['{"answer":"compatible"}'], response_model=lm.model)

    monkeypatch.setattr(AsyncLM15Engine, "complete", native)
    monkeypatch.setattr(AsyncLiteLLMEngine, "complete_legacy", compatibility)
    lm = dspy.LM("openai/gpt-4o", cache=False)
    adapter = dspy.JSONAdapter()
    signature = dspy.Signature("question -> answer")
    try:
        result = await asyncio.gather(
            adapter.acall(lm, {}, signature, [], {"question": "a"}),
            adapter.acall(lm, {"custom_llm_provider": "alternate"}, signature, [], {"question": "b"}),
        )
        assert result == [[{"answer": "native"}], [{"answer": "compatible"}]]
        assert lm.supports_response_schema
    finally:
        await lm.aclose()
