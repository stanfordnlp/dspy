"""Adapters invoke the public LM once per candidate."""

import pytest

import dspy
from dspy.lm15 import Response
from tests.test_utils.engines import RecordingEngine, make_response, recording_lm


@pytest.mark.parametrize("setting", ["headers", "extra_query", "custom_llm_provider"])
@pytest.mark.parametrize("on_constructor", [False, True])
def test_predict_cache_includes_effective_client_settings(setting, on_constructor, monkeypatch):
    seen = []

    def complete(**kwargs):
        value = kwargs[setting]
        answer = value if isinstance(value, str) else value["route"]
        seen.append(answer)
        return {"model": kwargs["model"], "choices": [{"message": {
            "role": "assistant", "content": f"[[ ## answer ## ]]\n{answer}\n[[ ## completed ## ]]"},
            "finish_reason": "stop"}]}

    monkeypatch.setattr("litellm.completion", complete)
    first = "first" if setting == "custom_llm_provider" else {"route": "first"}
    second = "second" if setting == "custom_llm_provider" else {"route": "second"}
    lm = dspy.LM("openai/gpt-4o", engine="litellm", **{setting: first})
    other = dspy.LM("openai/gpt-4o", engine="litellm", **{setting: second}) if on_constructor else lm
    predictor = dspy.Predict("question -> answer")
    for _ in range(2):
        assert predictor(question="hello", lm=lm).answer == "first"
        assert predictor(question="hello", lm=other, config={} if on_constructor else {setting: second}).answer == "second"
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
            "role": "assistant", "content": "[[ ## answer ## ]]\nhello\n[[ ## completed ## ]]"},
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
            "text": "[[ ## answer ## ]]\nhello\n[[ ## completed ## ]]", "finish_reason": "stop"}]}

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
        assert [inputs["kwargs"].get("rollout_id") for inputs in callback.inputs] == [None, 7, 8]


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
