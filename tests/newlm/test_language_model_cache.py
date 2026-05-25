import uuid
import warnings

import pytest

import dspy


class CountingLM(dspy.LanguageModel):
    def __init__(self, *, cache: bool = True):
        super().__init__(model=f"test/counting-{uuid.uuid4()}", cache=cache)
        self.forward_calls = 0

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        return dspy.LMResponse.from_text(
            f"call {self.forward_calls}",
            model=request.model,
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )


class AsyncCountingLM(CountingLM):
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        return dspy.LMResponse.from_text(
            f"async call {self.forward_calls}",
            model=request.model,
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )


class StreamingCountingLM(CountingLM):
    def __init__(self, *, cache: bool = True):
        super().__init__(cache=cache)
        self.stream_calls = 0

    def forward_stream(self, request: dspy.LMRequest):
        self.stream_calls += 1
        yield dspy.LMStreamStartEvent(model=request.model)
        yield dspy.LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=dspy.LMTextDelta(text=f"stream call {self.stream_calls}"),
        )
        yield dspy.LMStreamOutputEndEvent(output_index=0, finish_reason="stop")
        yield dspy.LMStreamEndEvent(
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )

    async def aforward_stream(self, request: dspy.LMRequest):
        self.stream_calls += 1
        yield dspy.LMStreamStartEvent(model=request.model)
        yield dspy.LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=dspy.LMTextDelta(text=f"astream call {self.stream_calls}"),
        )
        yield dspy.LMStreamOutputEndEvent(output_index=0, finish_reason="stop")
        yield dspy.LMStreamEndEvent(
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )


def test_language_model_uses_dspy_request_cache_before_forward():
    lm = CountingLM(cache=True)

    first = lm("hello")
    second = lm("hello")

    assert lm.forward_calls == 1
    assert first.text == "call 1"
    assert first.cache_hit is False
    assert second.text == "call 1"
    assert second.cache_hit is True
    assert second.usage == {}
    assert second.cost is None


def test_language_model_cache_can_be_disabled_on_instance():
    lm = CountingLM(cache=False)

    first = lm("hello")
    second = lm("hello")

    assert lm.forward_calls == 2
    assert first.text == "call 1"
    assert second.text == "call 2"
    assert second.cache_hit is False


def test_language_model_cache_can_be_overridden_per_call():
    lm = CountingLM(cache=True)

    first = lm("hello", cache=False)
    second = lm("hello", cache=False)

    assert lm.forward_calls == 2
    assert first.text == "call 1"
    assert second.text == "call 2"
    assert second.cache_hit is False


def test_language_model_cache_key_includes_lm_state():
    first_lm = CountingLM(cache=True)
    second_lm = CountingLM(cache=True)

    first = first_lm("hello")
    second = second_lm("hello")

    assert first.text == "call 1"
    assert second.text == "call 1"
    assert first_lm.forward_calls == 1
    assert second_lm.forward_calls == 1


def test_zero_temperature_rollout_warns_once_for_language_model():
    lm = CountingLM(cache=True)

    with pytest.warns(UserWarning, match="rollout_id"):
        lm("hello", rollout_id=1, temperature=0)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm("hello", rollout_id=2, temperature=0)

    assert len(record) == 0


@pytest.mark.asyncio
async def test_async_language_model_uses_dspy_request_cache_before_aforward():
    lm = AsyncCountingLM(cache=True)

    first = await lm.acall("hello")
    second = await lm.acall("hello")

    assert lm.forward_calls == 1
    assert first.text == "async call 1"
    assert second.text == "async call 1"
    assert second.cache_hit is True


@pytest.mark.asyncio
async def test_sync_and_async_language_model_caches_are_separate():
    lm = AsyncCountingLM(cache=True)

    sync_response = lm("hello")
    async_response = await lm.acall("hello")

    assert lm.forward_calls == 2
    assert sync_response.text == "call 1"
    assert async_response.text == "async call 2"
    assert async_response.cache_hit is False


def test_language_model_stream_uses_dspy_request_cache():
    lm = StreamingCountingLM(cache=True)

    first = lm.stream("hello")
    first_events = list(first)
    second = lm.stream("hello")
    second_events = list(second)

    assert lm.stream_calls == 1
    assert first_events[-1].type == "end"
    assert second_events[-1].type == "end"
    assert first.result().text == "stream call 1"
    assert first.result().cache_hit is False
    assert second.result().text == "stream call 1"
    assert second.result().cache_hit is True
    assert second.result().usage == {}
    assert second.result().cost is None


@pytest.mark.asyncio
async def test_language_model_async_stream_uses_dspy_request_cache():
    lm = StreamingCountingLM(cache=True)

    first = lm.astream("hello")
    first_events = [event async for event in first]
    second = lm.astream("hello")
    second_events = [event async for event in second]

    assert lm.stream_calls == 1
    assert first_events[-1].type == "end"
    assert second_events[-1].type == "end"
    assert first.result().text == "astream call 1"
    assert second.result().text == "astream call 1"
    assert second.result().cache_hit is True
