"""The rc3 integration through DSPy's public LM boundary, without provider calls."""

import copy

import pytest

import dspy
from dspy import lm15
from dspy.clients.call_result import CallResult
from dspy.clients.errors import wrap_error
from tests.clients.test_engine_errors import AsyncEngine, Engine, Sink, invoke


def response_body(text="hi preSTOPtail"):
    return {
        "id": "r", "model": "gpt-4o", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text, "logprobs": [
                {"token": token, "logprob": -0.25, "bytes": list(token.encode()), "top_logprobs": []}
                for token in ("hi ", "preSTOPtail")
            ]},
        ]}],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stop", [(), ("NEVER",), ("STOP",), ("hi",)])
async def test_typed_litellm_responses_honors_stops_and_preserves_usage(asynchronous, stop, monkeypatch):
    import litellm

    calls = []
    body = response_body()
    original = copy.deepcopy(body)

    def reply(**kwargs):
        calls.append(kwargs)
        return body

    async def areply(**kwargs):
        return reply(**kwargs)

    monkeypatch.setattr(litellm, "responses", reply)
    monkeypatch.setattr(litellm, "aresponses", areply)
    lm = dspy.LM("openai/gpt-4o", model_type="responses", engine="litellm", cache=False, num_retries=0)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),),
                           config=lm15.Config(stop=stop, logprobs=0))
    response = await invoke(lm, asynchronous, request)
    expected = "hi pre" if stop == ("STOP",) else "" if stop == ("hi",) else "hi preSTOPtail"
    assert response.text == expected
    assert response.usage == lm15.Usage(input_tokens=2, output_tokens=3, total_tokens=5)
    assert len(calls) == 1 and "stop" not in calls[0]
    assert body == original  # leave the provider evidence intact
    if stop:
        note, = response.adaptations
        assert (note.field, note.action, note.asked, note.applied) == (
            "config.stop", "client_side", list(stop), list(stop),
        )
        assert "completed" in note.reason and "not stopped early" in note.reason
        assert lm.history[-1]["adaptations"] == response.adaptations
    else:
        assert response.adaptations == ()
    if stop == ("STOP",):
        assert [token.token for token in response.logprobs] == ["hi "]
        assert not response.logprobs_complete
    elif stop == ("hi",):
        assert response.logprobs is None
    else:
        assert [token.token for token in response.logprobs] == ["hi ", "preSTOPtail"]
        assert response.logprobs_complete
    # DSPy's saved-response format keeps both the adjustment and score coverage.
    restored = CallResult.load(CallResult.native(response).dump()).responses[0]
    assert restored == response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_litellm_chat_still_sends_stop_to_provider(asynchronous, monkeypatch):
    import litellm

    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        return {"model": "gpt-4o", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "stopped upstream"}, "finish_reason": "stop"}]}

    async def areply(**kwargs):
        return reply(**kwargs)

    monkeypatch.setattr(litellm, "completion", reply)
    monkeypatch.setattr(litellm, "acompletion", areply)
    lm = dspy.LM("openai/gpt-4o", engine="litellm", cache=False, num_retries=0)
    request = lm15.Request(model=lm.model, messages=(lm15.Message.user("hi"),), config=lm15.Config(stop=("STOP",)))
    response = await invoke(lm, asynchronous, request)
    assert calls[0]["stop"] == ["STOP"]
    assert response.text == "stopped upstream" and response.adaptations == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_unsupported_feature_path_reaches_public_error(asynchronous, streaming):
    original = lm15.UnsupportedFeatureError("cannot carry image", feature="messages[1].parts[0]")
    engine = Engine(original)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=2)
    with dspy.context(send_stream=Sink() if streaming else None), pytest.raises(dspy.LMUnsupportedFeatureError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.feature == original.feature
    assert caught.value.features == [original.feature]
    assert caught.value.__cause__ is original and engine.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_collection_limit_is_preserved_without_retry_or_eager_materialization(asynchronous, streaming, monkeypatch):
    from dspy._vendor.lm15.types import LiveServerTextEvent

    accepted = LiveServerTextEvent(text="partial")
    rejected = LiveServerTextEvent(text="private rejected payload")
    original = lm15.CollectionLimitError("collection limit reached", limit="max_bytes", maximum=100,
                                        retained_bytes=50, partial_events=(accepted,), rejected_event=rejected)
    materializations = []

    def partial(error):
        materializations.append(error)
        return "lazy partial result"

    monkeypatch.setattr(lm15.CollectionLimitError, "partial", property(partial))
    engine = Engine(original)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=3)
    with dspy.context(send_stream=Sink() if streaming else None), pytest.raises(dspy.LMError) as caught:
        await invoke(lm, asynchronous, "hello")
    error = caught.value
    assert type(error) is dspy.LMCollectionLimitError
    assert error.code == "collection_limit" and not dspy.is_retryable_lm_error(error)
    assert error.__cause__ is original and engine.calls == 1
    assert error.partial_events == (accepted,) and error.rejected_event is rejected
    assert (error.limit, error.maximum, error.retained_bytes, error.retained_events) == ("max_bytes", 100, 50, 1)
    assert not materializations  # wrapping must not access the lazy property
    assert error.partial == "lazy partial result" and materializations == [original]
    assert "private rejected payload" not in str(error)
    assert not lm.history


def test_collection_limit_wrap_does_not_require_raising_to_access_partial():
    original = lm15.CollectionLimitError("limit")
    wrapped = wrap_error(original, model="custom")
    assert wrapped.partial.ended_by == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_request_conversion_refusal_keeps_the_feature(asynchronous, monkeypatch):
    import dspy.clients.execution as execution

    original = lm15.UnsupportedFeatureError("cannot convert", feature="config.custom")

    def refuse(*args, **kwargs):
        raise original

    monkeypatch.setattr(execution, "_canonical", refuse)
    lm = dspy.LM("openai/gpt-4o", engine="lm15", api_key="unused", cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnsupportedFeatureError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.feature == "config.custom" and caught.value.__cause__ is original


def test_explicit_feature_list_is_preserved():
    error = dspy.LMUnsupportedFeatureError("unsupported", feature="config.x", features=["custom"])
    assert error.feature == "config.x" and error.features == ["custom"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_judgment_response_preserves_json_prediction_and_typed_value(asynchronous, monkeypatch):
    import json
    from typing import Literal

    from dspy._vendor.lm15.providers.base import HttpResponse
    from dspy._vendor.lm15.providers.openai_chat import OpenAIChatLM
    from dspy.clients.execution import _canonical, prepare

    class Answer(dspy.Signature):
        question: str = dspy.InputField()
        accepted: bool = dspy.OutputField()
        label: Literal["yes", "no"] = dspy.OutputField()

    body = {"model": "gpt-4o", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": '{"accepted":false,"label":"yes"}',
    }, "finish_reason": "stop"}]}
    monkeypatch.setattr(OpenAIChatLM, "_send", lambda *args: HttpResponse(
        status=200, reason="OK", headers=[], body=json.dumps(body).encode(),
    ))
    # Exercise the real native parser with a fake wire, including async mirrors.
    from dspy._vendor.lm15.providers.async_base import AsyncOpenAIChatLM

    async def send(*args):
        return HttpResponse(status=200, reason="OK", headers=[], body=json.dumps(body).encode())

    monkeypatch.setattr(AsyncOpenAIChatLM, "_send", send)
    lm = dspy.LM("openai/gpt-4o", api_key="test", cache=False, num_retries=0)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        predict = dspy.Predict(Answer)
        result = await predict.acall(question="test") if asynchronous else predict(question="test")
    assert result.accepted is False and result.label == "yes"
    request = _canonical(prepare(lm, "test", None, {"response_format": {
        "type": "json_schema", "json_schema": {"name": "Answer", "schema": {
            "type": "object", "properties": {"accepted": {"type": "boolean"}},
        }},
    }}))
    response = await invoke(lm, asynchronous, request)
    assert isinstance(response.message.parts[0], lm15.DataPart)
    assert response.data == {"accepted": False, "label": "yes"}
    assert CallResult.load(CallResult.native(response).dump()).responses[0] == response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_rate_limit_evidence_survives_public_boundary(asynchronous, streaming):
    from dspy.clients.execution import _delay

    class RateLimited(Engine):
        def stream(self, request):
            yield lm15.StreamErrorEvent(error=lm15.ErrorDetail(
                code="rate_limit", message="wait", provider_code="limited",
                http_response={"retry_after": 23, "request_id": "req-123",
                               "rate_limit_headers": {"retry-after": ["23"]}},
            ))

    engine = RateLimited(lm15.RateLimitError(
        "wait", retry_after=23, request_id="req-123", provider_code="limited",
        rate_limit_headers={"retry-after": ["23"]},
    ))
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=0)
    with dspy.context(send_stream=Sink() if streaming else None), pytest.raises(dspy.LMRateLimitError) as caught:
        await invoke(lm, asynchronous, "hello")
    error = caught.value
    assert error.request_id == "req-123" and error.provider_code == "limited"
    assert error.retry_after == 23 and _delay(error, 0) == 23
    assert error.rate_limit_headers == {"retry-after": ("23",)}
