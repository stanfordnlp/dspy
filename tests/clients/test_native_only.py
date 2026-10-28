"""Native inference has no optional SDK fallback, even when LiteLLM is absent."""

import json
import os
import pickle
import subprocess
import sys

import pytest

import dspy
from tests.clients.test_lm import responses_wire
from tests.clients.test_lm_engine_execution import native_transport, wire


def test_clean_import_and_inference_without_litellm(tmp_path):
    code = """
import importlib.abc
import sys

class NoLiteLLM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "litellm" or fullname.startswith("litellm."):
            raise AssertionError("LiteLLM import attempted: " + fullname)

sys.meta_path.insert(0, NoLiteLLM())
import dspy
from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
from dspy.clients.engines import LM15Engine
from dspy.lm15 import RouterConfig
import dspy.clients.execution as execution

transport = FakeTransport([FakeResponse(status=200, body=b'{"model":"gpt-4o-mini","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}')])
execution.RouterConfig = lambda **kw: RouterConfig(**{**kw, "api_keys": {"openai-chat": "fake"}, "transport": transport})
assert dspy.LM("openai/gpt-4o-mini", cache=False)("hello") == ["ok"]
assert len(transport.requests) == 1
assert not any(n == "litellm" or n.startswith("litellm.") for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], env={
        "HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entry", ["constructor", "json", "pickle"])
def test_removed_engine_is_refused_with_migration_message(entry):
    lm = dspy.LM("openai/gpt-4o-mini")
    state = {**lm.dump_state(), "engine": "litellm"}
    lm._engine_spec = "litellm"  # Emulate an old trusted pickle; not a supported mutation.
    saved = pickle.dumps(lm)
    with pytest.raises(ValueError, match=r"(?i)litellm") as caught:
        if entry == "constructor":
            dspy.LM("openai/gpt-4o-mini", engine="litellm")
        elif entry == "json":
            dspy.LM.load_state(state)
        else:
            pickle.loads(saved)
    assert "lm15" in str(caught.value)
    assert "migration" in str(caught.value)


@pytest.mark.parametrize("engine", ["auto", "lm15"])
def test_native_engine_pickle_roundtrip_keeps_selection(engine, monkeypatch):
    transport = native_transport(monkeypatch, [wire("restored")])
    lm = pickle.loads(pickle.dumps(dspy.LM("openai/gpt-4o-mini", engine=engine, cache=False)))
    assert lm.engine == engine
    assert lm("hi") == ["restored"]
    assert len(transport.requests) == 1
    lm.close()


@pytest.mark.parametrize("engine", ["auto", "lm15"])
def test_removed_text_mode_is_refused_at_construction(engine):
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="model_type"):
        dspy.LM("openai/gpt-4o-mini", engine=engine, model_type="text")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine", ["auto", "lm15"])
async def test_known_native_request_works(engine, asynchronous, monkeypatch):
    transport = native_transport(monkeypatch, [wire("native")])
    lm = dspy.LM("openai/gpt-4o-mini", engine=engine, cache=False)
    try:
        assert (await lm.acall("hi") if asynchronous else lm("hi")) == ["native"]
        assert len(transport.requests) == 1
    finally:
        await lm.aclose()


@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4"])
@pytest.mark.parametrize("options", [
    {"headers": {"X-Test": "yes"}}, {"extra_headers": {"X-Test": "yes"}},
    {"api_version": "2025-01-01"}, {"custom_llm_provider": "other"},
])
def test_client_knobs_refuse_actionably_without_io(engine, thinking, model, options, monkeypatch):
    transport = native_transport(monkeypatch, [])
    with pytest.raises(dspy.LMUnsupportedFeatureError) as caught:
        dspy.LM(model, engine=engine, cache=False, **options)("hello", **({"thinking": thinking} if thinking else {}))
    assert next(iter(options)) in str(caught.value)
    assert "custom engine" in str(caught.value)
    assert not transport.requests


@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_unknown_provider_never_falls_back(engine, thinking, monkeypatch):
    transport = native_transport(monkeypatch, [])
    with pytest.raises(dspy.LMConfigurationError) as caught:
        dspy.LM("unregistered-provider/model", engine=engine, cache=False)(
            "hello", **({"thinking": thinking} if thinking else {}))
    assert "provider" in str(caught.value).lower()
    assert not transport.requests


@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("on_constructor", [False, True])
def test_unmapped_provider_options_refuse_without_io(engine, on_constructor, monkeypatch):
    transport = native_transport(monkeypatch, [])
    options = {"audio": {"voice": "alloy", "format": "wav"}}
    lm = dspy.LM("openai/gpt-4o-mini", engine=engine, cache=False, **(options if on_constructor else {}))
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="audio"):
        lm("hello", **({} if on_constructor else options))
    assert not transport.requests


@pytest.mark.parametrize("engine", ["auto", "lm15"])
@pytest.mark.parametrize("model,prefix,model_type", [
    ("openai/gpt-4o-mini", "OPENAI", "chat"), ("openai:gpt-4o-mini", "OPENAI", "responses"),
    ("gpt-4o-mini", "OPENAI", "chat"), ("azure/deployment", "AZURE", "chat"),
])
@pytest.mark.parametrize("variable", ["API_BASE", "BASE_URL"])
@pytest.mark.parametrize("explicit", [None, "api_base", "base_url"])
def test_gateway_environment_and_explicit_endpoint_priority(engine, model, prefix, model_type, variable, explicit, monkeypatch):
    transport = native_transport(monkeypatch, [responses_wire() if model_type == "responses" else wire("gateway")])
    for suffix in ("API_BASE", "BASE_URL"):
        monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    monkeypatch.setenv(f"{prefix}_{variable}", "https://ambient.test")
    options = {explicit: "https://explicit.test"} if explicit else {}
    lm = dspy.LM(model, engine=engine, model_type=model_type, api_key="fake", cache=False, num_retries=0, **options)
    try:
        assert lm("hi") == ([{"text": "OK"}] if model_type == "responses" else ["gateway"])
        assert len(transport.requests) == 1
        expected = "https://explicit.test/" if explicit else "https://ambient.test/"
        assert transport.requests[0].url.startswith(expected)
        body = json.loads(transport.requests[0].body)
        if model_type == "chat":
            assert body["messages"][-1]["content"] == "hi"
        else:
            assert body["input"][-1]["content"][0]["text"] == "hi"
    finally:
        lm.close()


@pytest.mark.parametrize("engine", ["auto", "lm15"])
def test_xai_api_key_alias_is_used_natively(engine, monkeypatch):
    transport = native_transport(monkeypatch, [wire("xai")])
    monkeypatch.setenv("XAI_API_KEY", "alias-test-key")
    lm = dspy.LM("xai/grok-4", engine=engine, cache=False, num_retries=0)
    try:
        assert lm("hi") == ["xai"]
        headers = {name.lower(): value for name, value in transport.requests[0].headers}
        assert headers["authorization"] == "Bearer alias-test-key"
        assert transport.requests[0].url.startswith("https://api.x.ai/")
    finally:
        lm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_openrouter_model_fallback_survives_native_predict(asynchronous, monkeypatch):
    transport = native_transport(monkeypatch, [wire("<answer>Paris</answer>")])
    models = ["openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5"]
    lm = dspy.LM("openrouter/openai/gpt-4.1-mini", api_key="fake", cache=False, num_retries=0,
                 models=models, route="fallback")
    try:
        with dspy.context(lm=lm):
            program = dspy.Predict("question -> answer")
            result = await program.acall(question="Capital of France?") if asynchronous else program(question="Capital of France?")
        assert result.answer == "Paris"
        body = json.loads(transport.requests[0].body)
        assert body["models"] == models
        assert body["route"] == "fallback"
        assert body["model"] == "openai/gpt-4.1-mini"
    finally:
        await lm.aclose()


def test_openrouter_fallback_options_are_not_sent_to_other_providers(monkeypatch):
    transport = native_transport(monkeypatch, [])
    lm = dspy.LM("openai/gpt-4.1-mini", api_key="fake", cache=False, models=["other/model"], route="fallback")
    with pytest.raises(dspy.LMUnsupportedFeatureError, match=r"models.*route"):
        lm("Hello")
    assert not transport.requests
