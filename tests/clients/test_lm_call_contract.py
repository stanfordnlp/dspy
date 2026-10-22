"""The DSPy 3.5 LM call contract: one string convenience, otherwise lm15 objects."""

import pytest

import dspy
from dspy.lm15 import Config, Message, Request, Response
from tests.test_utils.engines import RecordingEngine, make_response, recording_lm


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("media,expected", [
    (dspy.Audio(data="QUJD", audio_format="wav"), {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}),
    (dspy.File(file_id="file-123"), {"type": "file", "file": {"file_id": "file-123"}}),
])
async def test_auto_media_predictions_fall_back_before_native_execution(media, expected, asynchronous, monkeypatch):
    import litellm

    seen = []

    def complete(**kwargs):
        seen.append(kwargs)
        return litellm.ModelResponse(choices=[{"message": {
            "role": "assistant", "content": "[[ ## answer ## ]]\nReceived\n\n[[ ## completed ## ]]"},
            "finish_reason": "stop"}])

    async def acomplete(**kwargs):
        return complete(**kwargs)

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(litellm, "acompletion", acomplete)
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, num_retries=0)
    predict = dspy.Predict(f"media: dspy.{type(media).__name__} -> answer")
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        output = await predict.acall(media=media) if asynchronous else predict(media=media)
    assert output.answer == "Received"
    assert len(seen) == 1
    assert expected in seen[0]["messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("typed", [False, True])
async def test_litellm_citations_survive_canonical_calls_and_cache(asynchronous, typed, monkeypatch):
    import litellm

    citations = [[{"cited_text": "First source", "document_title": "Report", "document_index": 2}],
                 {"text": "Second source", "url": "https://example.com/source"}]
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return litellm.ModelResponse(
            model="anthropic/claude-sonnet-4-5", choices=[{"message": {
                "role": "assistant", "content": "An answer", "provider_specific_fields": {"citations": citations}},
                "finish_reason": "stop"}], usage={"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
        )

    async def acomplete(**kwargs):
        return complete(**kwargs)

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(litellm, "acompletion", acomplete)
    lm = dspy.LM("anthropic/claude-sonnet-4-5", engine="litellm")
    prompt = Request(model=lm.model, messages=(Message.user("hello"),)) if typed else "hello"
    expected = [{"text": "First source", "title": "Report"},
                {"text": "Second source", "url": "https://example.com/source"}]
    for _ in range(2):
        result = await lm.acall(prompt) if asynchronous else lm(prompt)
        if typed:
            assert [(c.text, c.title, c.url) for c in result.citations] == [
                ("First source", "Report", None), ("Second source", None, "https://example.com/source"),
            ]
        else:
            assert result == [{"text": "An answer", "citations": expected}]
        raw = lm.history[-1]["response"].provider_data
        assert raw["choices"][0]["message"]["provider_specific_fields"]["citations"] == citations
    assert len(calls) == 1
    assert lm.history[0]["usage"]["total_tokens"] == 12
    assert lm.history[1]["usage"] == {}


@pytest.mark.parametrize("setting", ["headers", "extra_query", "custom_llm_provider"])
@pytest.mark.parametrize("on_constructor", [False, True])
@pytest.mark.parametrize("via_predict", [False, True])
def test_call_cache_includes_effective_client_settings(setting, on_constructor, via_predict, monkeypatch):
    seen = []

    def complete(**kwargs):
        value = kwargs[setting]
        answer = value if isinstance(value, str) else value["route"]
        seen.append(answer)
        content = f"<answer>{answer}</answer>" if via_predict else answer
        return {"model": kwargs["model"], "choices": [{"message": {
            "role": "assistant", "content": content},
            "finish_reason": "stop"}]}

    monkeypatch.setattr("litellm.completion", complete)
    first = "first" if setting == "custom_llm_provider" else {"route": "first"}
    second = "second" if setting == "custom_llm_provider" else {"route": "second"}
    lm = dspy.LM("openai/gpt-4o", engine="litellm", **{setting: first})
    other = dspy.LM("openai/gpt-4o", engine="litellm", **{setting: second}) if on_constructor else lm
    predictor = dspy.Predict("question -> answer")
    for _ in range(2):
        options = {} if on_constructor else {setting: second}
        if via_predict:
            assert predictor(question="hello", lm=lm).answer == "first"
            assert predictor(question="hello", lm=other, config=options).answer == "second"
        else:
            assert lm("hello") == ["first"]
            assert other("hello", **options) == ["second"]
    assert seen == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("setting,default", [
    ("headers", {"route": "first"}), ("extra_query", {"route": "first"}),
    ("custom_llm_provider", "first"), ("timeout", 7),
])
async def test_predict_can_clear_client_defaults(asynchronous, setting, default, monkeypatch):
    seen = []

    def complete(**kwargs):
        seen.append(kwargs)
        return {"model": kwargs["model"], "choices": [{"message": {
            "role": "assistant", "content": "<answer>hello</answer>"},
            "finish_reason": "stop"}]}

    async def acomplete(**kwargs):
        return complete(**kwargs)

    monkeypatch.setattr("litellm.completion", complete)
    monkeypatch.setattr("litellm.acompletion", acomplete)
    lm = dspy.LM("openai/gpt-4o", engine="litellm", cache=False, **{setting: default})
    predictor = dspy.Predict("question -> answer")
    options = {"question": "hi", "lm": lm, "config": {setting: None}}
    result = await predictor.acall(**options) if asynchronous else predictor(**options)
    assert result.answer == "hello"
    assert "route" not in seen[0][setting] if setting == "headers" else seen[0][setting] is None
    assert lm.kwargs[setting] == default


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_text_predictions_forward_top_k(asynchronous, monkeypatch):
    seen = []

    def complete(**kwargs):
        seen.append(kwargs)
        return {"model": kwargs["model"], "choices": [{
            "text": "<answer>hello</answer>", "finish_reason": "stop"}]}

    async def acomplete(**kwargs):
        return complete(**kwargs)

    monkeypatch.setattr("litellm.text_completion", complete)
    monkeypatch.setattr("litellm.atext_completion", acomplete)
    lm = dspy.LM("openai/local", model_type="text", engine="litellm", cache=False, top_k=7)
    predictor = dspy.Predict("question -> answer")
    result = await predictor.acall(question="hi", lm=lm) if asynchronous else predictor(question="hi", lm=lm)
    assert result.answer == "hello"
    assert seen[0]["top_k"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_string_prompt_is_one_user_message_with_lm_defaults(asynchronous):
    lm = recording_lm(["pong"], temperature=0.3, max_tokens=50)

    outputs = await lm.acall("ping") if asynchronous else lm("ping")

    assert outputs == ["pong"]
    [request] = lm.engine.requests
    assert request.messages == (Message.user("ping"),)
    assert request.system is None
    assert request.config.temperature == 0.3
    assert request.config.max_tokens == 50
    assert lm.history[-1]["prompt"] == "ping"
    assert lm.history[-1]["outputs"] == ["pong"]
    assert isinstance(lm.history[-1]["response"], Response)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("seed", [None, 0, 41])
async def test_string_prompt_accepts_generation_overrides_and_candidates(asynchronous, seed):
    lm = recording_lm(["a", "b"], temperature=0.0, seed=seed, cache=True)
    options = {"temperature": 0.9, "n": 2, "stop": ["END"]}

    outputs = await lm.acall("ping", **options) if asynchronous else lm("ping", **options)

    assert outputs == ["a", "b"]
    assert [r.config.temperature for r in lm.engine.requests] == [0.9, 0.9]
    assert [r.config.seed for r in lm.engine.requests] == [seed, None if seed is None else seed + 1]
    assert lm.kwargs["seed"] == seed
    assert lm.engine.requests[0].config.stop == ("END",)
    assert len(lm.history) == 1
    cached = await lm.acall("ping", **options) if asynchronous else lm("ping", **options)
    assert cached == outputs
    assert len(lm.engine.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_request_returns_one_response(asynchronous):
    lm = recording_lm([make_response("hello", thinking="thought")], n=3)
    request = Request(model=lm.model, system="Be brief.", messages=(Message.user("hi"),), config=Config(max_tokens=5))

    response = await lm.acall(request) if asynchronous else lm(request)

    assert isinstance(response, Response)
    assert response.text == "hello"
    assert lm.engine.requests == [request]
    assert lm.history[-1]["request"] is request
    assert lm.history[-1]["messages"][0] == {"role": "system", "content": "Be brief."}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("override", [None, 2, 1])
@pytest.mark.parametrize("seed", [None, 0, 41])
async def test_predict_candidates_are_individually_intercepted_cached_and_recorded(asynchronous, override, seed):
    from dspy.primitives.facade import SandboxLM
    from dspy.utils.callback import BaseCallback
    from dspy.utils.usage_tracker import track_usage

    class Calls(BaseCallback):
        def __init__(self):
            self.inputs = []
            self.outputs = []

        def on_lm_start(self, call_id, instance, inputs):
            self.inputs.append(inputs)

        def on_lm_end(self, call_id, outputs, exception):
            assert exception is None
            self.outputs.append(outputs)

    callback = Calls()
    answers = [f'{{"answer":"{i}"}}' for i in range(6)]
    lm = recording_lm(answers, cache=True, n=3, rollout_id=7, seed=seed, temperature=0.7, callbacks=[callback])
    reserved = []
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(SandboxLM(lm, reserve=reserved.append))
    n = 3 if override is None else override
    config = {} if override is None else {"n": override}

    async def predict(**options):
        if asynchronous:
            return await predictor.acall(question="count", config={**config, **options})
        return predictor(question="count", config={**config, **options})

    with dspy.context(adapter=dspy.JSONAdapter()), track_usage() as usage:
        first = await predict()
        repeated = await predict(rollout_id=7)
        adjacent = await predict(rollout_id=8)

    assert first.completions.answer == repeated.completions.answer == [str(i) for i in range(n)]
    assert adjacent.completions.answer == [str(i) for i in range(n, 2 * n)]
    assert reserved == [1] * (3 * n)
    assert len(lm.engine.requests) == 2 * n
    assert [r.config.seed for r in lm.engine.requests] == [None if seed is None else seed + i for i in range(n)] * 2
    assert lm.kwargs["seed"] == seed
    assert len(lm.history) == len(callback.inputs) == len(callback.outputs) == 3 * n
    assert all(isinstance(r, Response) for r in callback.outputs)
    assert all(isinstance(h["response"], Response) and len(h["outputs"]) == 1 for h in lm.history)
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 4 * n
    if n == 1:
        assert [inputs["rollout_id"] for inputs in callback.inputs] == [None, 7, 8]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_failed_candidate_preserves_completed_calls_and_their_cache(asynchronous):
    from dspy.utils.usage_tracker import track_usage

    lm = recording_lm(['{"answer":"first"}', dspy.LMInvalidRequestError("bad candidate"), '{"answer":"second"}'],
                      cache=True, num_retries=0)
    predictor = dspy.Predict("question -> answer", n=2)

    async def predict():
        if asynchronous:
            return await predictor.acall(question="count", lm=lm)
        return predictor(question="count", lm=lm)

    with dspy.context(adapter=dspy.JSONAdapter()), track_usage() as usage:
        with pytest.raises(dspy.LMInvalidRequestError, match="bad candidate"):
            await predict()
        assert len(lm.history) == 1
        assert usage.get_total_tokens()[lm.model]["total_tokens"] == 2
        result = await predict()

    assert result.completions.answer == ["first", "second"]
    assert len(lm.engine.requests) == 3
    assert len(lm.history) == 3  # first success, cached first candidate, second success
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 4


@pytest.mark.asyncio
async def test_cancelled_prediction_keeps_completed_candidate_history_and_usage():
    import asyncio

    from dspy.utils.usage_tracker import track_usage

    waiting = asyncio.Event()

    class Engine:
        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 2:
                waiting.set()
                await asyncio.Event().wait()
            return make_response('{"answer":"first"}')

    lm = dspy.LM("test/model", engine=RecordingEngine(), async_engine=Engine(), cache=True)
    predictor = dspy.Predict("question -> answer", n=2)
    with dspy.context(adapter=dspy.JSONAdapter()), track_usage() as usage:
        task = asyncio.create_task(predictor.acall(question="count", lm=lm))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert len(lm.history) == 1
    assert lm.history[0]["response"].text == '{"answer":"first"}'
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 2


def test_openai_style_messages_are_refused_with_a_migration_hint():
    lm = recording_lm()
    with pytest.raises(TypeError, match=r"lm\(messages=\[\.\.\.\]\) was removed in DSPy 3\.5"):
        lm(messages=[{"role": "user", "content": "hi"}])
    assert lm.engine.requests == []


def test_request_calls_take_no_generation_options():
    lm = recording_lm()
    request = Request(model=lm.model, messages=(Message.user("hi"),))
    with pytest.raises(TypeError, match=r"Request\.config"):
        lm(request, temperature=0.5)
    for count_option in ("n", "num_generations"):
        with pytest.raises(TypeError, match="returns one Response"):
            lm(request, **{count_option: 2})
    with pytest.raises(ValueError, match=r"Request\.model must match"):
        lm(Request(model="other", messages=(Message.user("hi"),)))
    with pytest.raises(TypeError, match="Request or a prompt string"):
        lm(123)


def test_forward_subclasses_are_no_longer_an_integration_point():
    class OldStyle(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return {"choices": [{"message": {"content": "x"}}]}

    lm = OldStyle("old")
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="engine"):
        lm("hi")


def test_removed_legacy_names_are_gone():
    import dspy.clients.engines as engines

    assert not hasattr(engines, "LegacyEngine")
    assert not hasattr(dspy.BaseLM, "forward")
    assert not hasattr(dspy.LM, "forward")
    assert not hasattr(dspy.Tool, "format_as_litellm_function_call")


def test_engine_capabilities_reach_the_lm():
    lm = recording_lm()
    assert lm.supports_function_calling is True
    assert "tools" in lm.supported_params
    plain = dspy.BaseLM("m", engine=type("E", (), {"complete": lambda self, r: None})())
    assert plain.supports_function_calling is False
    assert plain.supported_params == set()


def test_audio_on_a_native_chat_completions_engine_fails_before_the_wire():
    """The bundled lm15 chat writer carries text and images only; the refusal is loud, not silent."""
    lm = dspy.LM("openai/gpt-4o-mini", engine="lm15", api_key="unused", cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError, match="audio part"):
        dspy.Predict("audio: dspy.Audio -> transcript")(audio=dspy.Audio(data="QUJD", audio_format="wav"), lm=lm)
