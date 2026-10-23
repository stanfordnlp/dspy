"""The DSPy 3.5 LM call contract: one string convenience, otherwise lm15 objects."""


import pytest

import dspy
from dspy.lm15 import Config, Message, Request, Response
from tests.clients.test_lm_engine_execution import native_transport, wire
from tests.test_utils.engines import RecordingEngine, make_response, recording_lm


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("media,expected", [
    (dspy.Audio(data="QUJD", audio_format="wav"), {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}),
    (dspy.File(file_id="file-123"), {"type": "file", "file": {"file_id": "file-123"}}),
])
async def test_auto_media_predictions_refuse_before_native_execution(media, expected, asynchronous, monkeypatch):
    transport = native_transport(monkeypatch, [])
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, num_retries=0)
    predict = dspy.Predict(f"media: dspy.{type(media).__name__} -> answer")
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
        with pytest.raises(dspy.LMUnsupportedFeatureError):
            await predict.acall(media=media) if asynchronous else predict(media=media)
    assert not transport.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("typed", [False, True])
async def test_citations_survive_canonical_calls_and_cache(asynchronous, typed):
    from dspy.lm15 import CitationPart, TextPart, Usage

    response = Response(None, "test/model", Message.assistant((
        TextPart("An answer"), CitationPart(text="First source", title="Report"),
        CitationPart(text="Second source", url="https://example.com/source"),
    )), "stop", Usage(input_tokens=9, output_tokens=3, total_tokens=12))
    lm = recording_lm([response], cache=True)
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
    assert len(lm.engine.requests) == 1
    assert lm.history[0]["usage"]["total_tokens"] == 12
    assert lm.history[1]["usage"] == {}


@pytest.mark.parametrize("on_constructor", [False, True])
@pytest.mark.parametrize("via_predict", [False, True])
def test_call_cache_includes_effective_timeout(on_constructor, via_predict, monkeypatch):
    from dspy.clients.engines import LM15Engine

    calls = []

    def complete(self, request):
        calls.append(self.config.timeouts.read)
        answer = {7: "first", 9: "second"}[self.config.timeouts.read]
        return make_response(f"[[ ## answer ## ]]\n{answer}\n[[ ## completed ## ]]" if via_predict else answer)

    monkeypatch.setattr(LM15Engine, "complete", complete)
    # Timeout remains a supported client knob. Credentials and endpoint aliases
    # are deliberately excluded by DSPy's existing answer-cache contract.
    lm = dspy.LM("openai/gpt-4o", timeout=7)
    other = dspy.LM("openai/gpt-4o", timeout=9) if on_constructor else lm
    predictor = dspy.Predict("question -> answer")
    for _ in range(2):
        options = {} if on_constructor else {"timeout": 9}
        if via_predict:
            assert predictor(question="hello", lm=lm).answer == "first"
            assert predictor(question="hello", lm=other, config=options).answer == "second"
        else:
            assert lm("hello") == ["first"]
            assert other("hello", **options) == ["second"]
    assert calls == [7, 9]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_predict_can_clear_client_defaults(asynchronous, monkeypatch):
    transport = native_transport(monkeypatch, [wire("[[ ## answer ## ]]\nhello\n[[ ## completed ## ]]")])
    lm = dspy.LM("openai/gpt-4o", cache=False, base_url="https://override.test/v1")
    predictor = dspy.Predict("question -> answer")
    options = {"question": "hi", "lm": lm, "config": {"base_url": None}}
    result = await predictor.acall(**options) if asynchronous else predictor(**options)
    assert result.answer == "hello"
    assert transport.requests[0].url == "https://api.openai.com/v1/chat/completions"
    assert lm.kwargs["base_url"] == "https://override.test/v1"


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


def test_string_prompt_accepts_generation_overrides_and_candidates():
    lm = recording_lm(["a", "b"], temperature=0.0)

    outputs = lm("ping", temperature=0.9, n=2, stop=["END"])

    assert outputs == ["a", "b"]
    assert [r.config.temperature for r in lm.engine.requests] == [0.9, 0.9]
    assert lm.engine.requests[0].config.stop == ("END",)
    assert len(lm.history) == 1


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
async def test_predict_candidates_are_individually_intercepted_cached_and_recorded(asynchronous, override):
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
    lm = recording_lm(answers, cache=True, n=3, rollout_id=7, seed=41, temperature=0.7, callbacks=[callback])
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
    assert all(r.config.seed == 41 for r in lm.engine.requests)
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
    with pytest.raises(ValueError, match="Request.model must match"):
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
