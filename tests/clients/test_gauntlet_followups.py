"""Gauntlet follow-ups fixed directly in DSPy, checked without provider calls."""

import re

import pytest

import dspy
from dspy import lm15
from dspy.clients.backend_selection import select_backend
from dspy.clients.engines.lm15_engine import timeouts_for
from dspy.clients.execution import _engine, prepare
from dspy.clients.lm import _is_openai_reasoning_model

# ─── gpt-5.6 style names are reasoning models ────────────────────────

@pytest.mark.parametrize("model,expected", [
    ("openai/gpt-5.6-luna", True),
    ("openai/gpt-5.6", True),
    ("openai/gpt-5.6-mini-2026-03-17", True),
    ("azure/gpt-5.12-pro", True),
    ("openai/gpt-5", True),
    ("openai/gpt-5-mini", True),
    ("openai/GPT-5.6-Luna", True),
    ("openai/gpt-5.6-chat", False),
    ("openai/gpt-5-chat-latest", False),
    ("openai/gpt-5.6chat", False),  # not a known naming form; do not guess a reasoning family
    ("openai/gpt-50", False),
    ("openai/gpt-5x", False),
    ("openai/gpt-4.1", False),
])
def test_reasoning_model_detection_handles_dotted_versions(model, expected):
    assert _is_openai_reasoning_model(model) is expected


def test_dotted_reasoning_model_gets_validation_and_token_field():
    lm = dspy.LM("openai/gpt-5.6-luna", temperature=1.0, max_tokens=16_000)
    assert lm.kwargs["max_completion_tokens"] == 16_000 and "max_tokens" not in lm.kwargs
    with pytest.raises(dspy.LMConfigurationError, match="reasoning models require"):
        dspy.LM("openai/gpt-5.6-luna", temperature=0.7)
    with pytest.raises(dspy.LMConfigurationError):
        dspy.LM("openai/gpt-5.6-luna", max_tokens=1000)
    # The existing gpt-5 rule treats temperature 0 as "unset"; gpt-5.6 now
    # follows that same rule rather than a different one.
    assert dspy.LM("openai/gpt-5.6-luna", temperature=0.0).kwargs["temperature"] == 0.0


# ─── malformed text fails as an input error on every engine choice ──

@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("engine", ["auto", "lm15", "litellm"])
@pytest.mark.parametrize("where", ["prompt", "messages", "request", "system"])
async def test_lone_surrogate_is_refused_before_execution(asynchronous, engine, where, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: pytest.fail("must not reach the provider SDK"))
    monkeypatch.setattr(litellm, "acompletion", lambda **kw: pytest.fail("must not reach the provider SDK"))
    lm = dspy.LM("openai/gpt-4o", engine=engine, api_key="k", cache=False, num_retries=3)
    bad = "hello \ud800 world"
    if where == "prompt":
        args, kwargs = (bad,), {}
    elif where == "messages":
        args, kwargs = (), {"messages": [{"role": "user", "content": bad}]}
    elif where == "system":
        args, kwargs = (lm15.Request(model=lm.model, system=bad, messages=(lm15.Message.user("hi"),)),), {}
    else:
        args, kwargs = (lm15.Request(model=lm.model, messages=(lm15.Message.user(bad),)),), {}
    with pytest.raises(ValueError, match="U\\+D800") as caught:
        if asynchronous:
            await lm.acall(*args, **kwargs)
        else:
            lm(*args, **kwargs)
    assert not isinstance(caught.value, dspy.LMError)
    assert not lm.history


def test_valid_unicode_is_not_refused():
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="k", cache=False)
    call = prepare(lm, "héllo 🌍 \u200f", None, {})
    assert call.legacy["messages"][0]["content"] == "héllo 🌍 \u200f"


# ─── disabled timeouts are not silently replaced by 600 seconds ─────

def test_httpx_timeout_with_a_disabled_component_is_not_native():
    httpx = pytest.importorskip("httpx")
    with pytest.raises(lm15.UnsupportedFeatureError, match="disables pool, read, write") as caught:
        timeouts_for(httpx.Timeout(connect=5.0, read=None, write=None, pool=None))
    assert caught.value.feature == "timeout"
    with pytest.raises(lm15.UnsupportedFeatureError, match="disables connect, pool, read, write"):
        timeouts_for(httpx.Timeout(None))
    assert timeouts_for(httpx.Timeout(connect=5.0, read=900.0, write=30.0, pool=15.0)) == lm15.Timeouts(
        connect=5.0, read=900.0, write=30.0, pool=15.0,
    )


@pytest.mark.parametrize("value", [None, "unset"])
def test_disabled_timeout_selects_litellm_under_auto_and_refuses_under_lm15(value):
    httpx = pytest.importorskip("httpx")
    timeout = httpx.Timeout(None) if value is None else httpx.Timeout(connect=5.0, read=None, write=None, pool=None)
    assert not select_backend(dspy.LM("openai/gpt-4o", api_key="k", cache=False, timeout=timeout)).native
    with pytest.raises(lm15.UnsupportedFeatureError, match="disables"):
        select_backend(dspy.LM("openai/gpt-4o", api_key="k", cache=False, engine="lm15", timeout=timeout))
    # Through the public call the refusal is a DSPy error naming the setting.
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="disables") as caught:
        dspy.LM("openai/gpt-4o", api_key="k", cache=False, engine="lm15", timeout=timeout)("hi")
    assert caught.value.feature == "timeout"
    assert select_backend(dspy.LM("openai/gpt-4o", api_key="k", cache=False, timeout=httpx.Timeout(60.0))).native


# ─── top_k survives the typed LiteLLM route ─────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_typed_litellm_chat_forwards_top_k_and_records_nothing(asynchronous, monkeypatch):
    import litellm

    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        return {"model": "claude-haiku-4-5", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    async def areply(**kwargs):
        return reply(**kwargs)

    monkeypatch.setattr(litellm, "completion", reply)
    monkeypatch.setattr(litellm, "acompletion", areply)
    lm = dspy.LM("anthropic/claude-haiku-4-5", engine="litellm", cache=False, num_retries=0)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),), config=lm15.Config(top_k=7))
    response = await (lm.acall(request) if asynchronous else __import__("anyio").to_thread.run_sync(lambda: lm(request)))
    assert calls[0]["top_k"] == 7
    assert response.text == "ok" and response.adaptations == ()


def test_typed_litellm_responses_records_a_dropped_top_k(monkeypatch):
    import litellm

    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        return {"id": "r", "model": "gpt-4o", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}

    monkeypatch.setattr(litellm, "responses", reply)
    lm = dspy.LM("openai/gpt-4o", model_type="responses", engine="litellm", cache=False, num_retries=0)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),), config=lm15.Config(top_k=7))
    response = lm(request)
    assert "top_k" not in calls[0]
    note, = response.adaptations
    assert (note.field, note.action, note.asked) == ("config.top_k", "dropped", 7)
    assert re.search(r"Responses", note.reason)


def test_native_anthropic_top_k_stays_native():
    lm = dspy.LM("anthropic/claude-haiku-4-5", api_key="k", cache=False)
    backend, canonical, provider = _engine(lm, prepare(lm, "hi", None, {"top_k": 7}), False)
    assert provider == "anthropic" and canonical.config.top_k == 7
    assert type(backend).__name__ == "LM15Engine"
