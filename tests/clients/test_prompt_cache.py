"""The prompt-cache bridge preserves lm15 objects without changing answer caching."""

import json

import pytest

import dspy
from dspy.clients.execution import _canonical, prepare
from dspy.lm15 import CacheConfig, Config, Message, Request, Response, Usage


class Engine:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return Response(None, request.model, Message.assistant("ok"), "stop", Usage(input_tokens=1, output_tokens=1))


class AsyncEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_defaults_overrides_and_typed_requests(asynchronous):
    engine = Engine()
    stable = CacheConfig(prefix="stable")
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, prompt_cache=stable)

    async def call(*args, **kwargs):
        return await lm.acall(*args, **kwargs) if asynchronous else lm(*args, **kwargs)

    assert await call("hello") == ["ok"]
    assert engine.requests[-1].config.cache == stable
    override = CacheConfig(prefix="history")
    await call("hello", prompt_cache=override)
    assert engine.requests[-1].config.cache == override
    await call("hello", prompt_cache=None)
    assert engine.requests[-1].config.cache is None
    assert lm.kwargs["prompt_cache"] == stable
    request = Request(model=lm.model, messages=(Message.user("typed"),), config=Config())
    assert isinstance(await call(request), Response)
    assert engine.requests[-1].config.cache is None  # typed requests own their complete config
    with pytest.raises(TypeError, match=r"Request\.config"):
        await call(request, prompt_cache=override)


def test_copy_and_json_state_roundtrip():
    config = CacheConfig(prefix="stable", retention="long", key="prefix")
    lm = dspy.LM("openai/gpt-4o", engine="lm15", prompt_cache=config)
    restored = dspy.BaseLM.load_state(json.loads(json.dumps(lm.dump_state())))
    assert restored.kwargs["prompt_cache"] == config
    assert lm.copy().kwargs["prompt_cache"] == config
    assert "prompt_cache" not in lm.copy(prompt_cache=None).kwargs
    assert lm.kwargs["prompt_cache"] == config
    assert lm.copy(prompt_cache=CacheConfig(mode="off")).kwargs["prompt_cache"].mode == "off"


@pytest.mark.parametrize("value", [True, "stable", {"prefix": "stable"}])
def test_invalid_values_are_rejected(value):
    with pytest.raises(TypeError, match="CacheConfig"):
        dspy.LM("openai/gpt-4o", prompt_cache=value)
    lm = dspy.LM("custom", engine=Engine(), cache=False)
    with pytest.raises(TypeError, match="CacheConfig"):
        lm("hello", prompt_cache=value)
    with pytest.raises(TypeError, match="CacheConfig"):
        lm.copy(prompt_cache=value)


def test_answer_cache_keys_serialize_and_separate_prompt_policies():
    lm = dspy.LM("openai/gpt-4o")
    plain = prepare(lm, "hello", None, {}).key(lm, False)
    disabled = prepare(lm, "hello", None, {"prompt_cache": None}).key(lm, False)
    stable = prepare(lm, "hello", None, {"prompt_cache": CacheConfig(prefix="stable")}).key(lm, False)
    history = prepare(lm, "hello", None, {"prompt_cache": CacheConfig(prefix="history")}).key(lm, False)
    assert plain == disabled
    assert dspy.cache.cache_key(stable) != dspy.cache.cache_key(history)
    assert json.loads(json.dumps(stable))["prompt_cache"] == {"mode": "auto", "prefix": "stable"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_answer_caching_still_works(asynchronous):
    engine = Engine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), prompt_cache=CacheConfig(prefix="stable"))
    for _ in range(2):
        result = await lm.acall("hello") if asynchronous else lm("hello")
        assert result == ["ok"]
    assert len(engine.requests) == 1
    assert lm.history[-1]["usage"] == {}


@pytest.mark.parametrize("options", [
    {"engine": "litellm"}, {"engine": "auto", "headers": {"x-test": "yes"}},
    {"engine": "auto", "model_type": "text"},
])
def test_no_compatibility_fallback(options, monkeypatch):
    from dspy.clients.engines import LiteLLMEngine

    def forbidden(*args, **kwargs):
        raise AssertionError("prompt_cache must not reach LiteLLM")

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", forbidden)
    lm = dspy.LM("openai/gpt-4o", cache=False, prompt_cache=CacheConfig(prefix="stable"), **options)
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        lm("hello")


def test_unrepresentable_input_does_not_fallback_with_prompt_cache(monkeypatch):
    from dspy.clients.engines import LiteLLMEngine

    def forbidden(*args, **kwargs):
        raise AssertionError("Compatibility fallback attempted")

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", forbidden)
    lm = dspy.LM("openai/gpt-4o", cache=False, prompt_cache=CacheConfig(prefix="stable"))
    with pytest.raises(dspy.LMUnsupportedFeatureError):
        lm("hello", prediction={"type": "content", "content": "hello"})


def test_provider_cache_options_cannot_conflict():
    lm = dspy.LM("openai/gpt-4o", prompt_cache=CacheConfig(prefix="stable"))
    call = prepare(lm, "hello", None, {"prompt_cache_key": "other"})
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="Do not combine"):
        _canonical(call)


def test_none_is_not_forwarded_to_litellm(monkeypatch):
    from dspy.clients.call_result import CallResult
    from dspy.clients.engines import LiteLLMEngine

    def complete(self, lm, request, **context):
        assert "prompt_cache" not in request
        return CallResult(outputs=["ok"], response_model=lm.model)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", complete)
    assert dspy.LM("openai/gpt-4o", engine="litellm", cache=False)("hello", prompt_cache=None) == ["ok"]


def test_native_anthropic_marks_system_prefix(monkeypatch):
    import dspy.clients.execution as execution
    from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
    from dspy.lm15 import RouterConfig

    transport = FakeTransport([FakeResponse(status=200, body=json.dumps({
        "id": "answer", "model": "claude-haiku-4-5", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 100},
    }).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(**kwargs, transport=transport))
    lm = dspy.LM("anthropic/claude-haiku-4-5", engine="lm15", api_key="fake", cache=False,
                 prompt_cache=CacheConfig(prefix="stable"))
    try:
        assert lm(messages=[{"role": "system", "content": "stable instructions"},
                            {"role": "user", "content": "changing input"}]) == ["ok"]
        payload = json.loads(transport.requests[0].body)
        assert payload["system"] == [{"type": "text", "text": "stable instructions",
                                       "cache_control": {"type": "ephemeral"}}]
        assert "prompt_cache" not in payload
        assert "cache_control" not in payload["messages"][0]["content"][0]
        assert lm.history[-1]["usage"]["prompt_tokens_details"]["cache_creation_tokens"] == 100
    finally:
        lm.close()
