import uuid

import pytest

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback, with_callbacks


@pytest.fixture(autouse=True)
def reset_settings():
    original_settings = dspy.settings.copy()
    yield
    dspy.configure(**original_settings)


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.calls = []
        self.active_ids_at_start = []

    def on_lm_start(self, call_id, instance, inputs):
        self.calls.append(("start", call_id, instance, inputs))
        self.active_ids_at_start.append(ACTIVE_CALL_ID.get())

    def on_lm_end(self, call_id, outputs, exception):
        self.calls.append(("end", call_id, outputs, exception))


class CallbackLM(dspy.BaseLM):
    def __init__(self, *, cache=False, callbacks=None):
        super().__init__(model=f"test/callback-{uuid.uuid4()}", cache=cache, callbacks=callbacks, temperature=0.1)
        self.forward_calls = 0

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        assert ACTIVE_CALL_ID.get() is not None
        return dspy.LMResponse.from_text(f"call {self.forward_calls}", model=request.model)

    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        assert ACTIVE_CALL_ID.get() is not None
        return dspy.LMResponse.from_text(f"async call {self.forward_calls}", model=request.model)


class ErrorLM(CallbackLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        raise NativeContextError("too long")

    def normalize_error(self, error: Exception, request: dspy.LMRequest) -> Exception:
        if isinstance(error, NativeContextError):
            return dspy.ContextWindowExceededError(model=request.model, provider="test")
        return error


class NativeContextError(Exception):
    pass


class StreamLM(CallbackLM):
    def forward_stream(self, request: dspy.LMRequest):
        assert ACTIVE_CALL_ID.get() is not None
        yield dspy.lm.LMStreamStartEvent(model=request.model)
        yield dspy.lm.LMStreamDeltaEvent(output_index=0, part_index=0, delta=dspy.lm.LMTextDelta(text="hello"))
        yield dspy.lm.LMStreamEndEvent()

    async def aforward_stream(self, request: dspy.LMRequest):
        assert ACTIVE_CALL_ID.get() is not None
        yield dspy.lm.LMStreamStartEvent(model=request.model)
        yield dspy.lm.LMStreamEndEvent(response=dspy.LMResponse.from_text("async hello", model=request.model))


def test_language_model_instance_callback_gets_normalized_request_and_response():
    callback = RecordingCallback()
    lm = CallbackLM(callbacks=[callback])

    response = lm("hello", api_key="secret")

    assert response.text == "call 1"
    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[0][1] == callback.calls[1][1]
    inputs = callback.calls[0][3]
    assert isinstance(inputs["request"], dspy.LMRequest)
    assert inputs["request"].messages == [dspy.User("hello")]
    assert inputs["request"].config.extensions["api_key"] == "<redacted>"
    assert inputs["raw"]["kwargs"]["api_key"] == "<redacted>"
    assert callback.calls[1][2] == response
    assert callback.calls[1][3] is None
    assert ACTIVE_CALL_ID.get() is None


def test_language_model_global_callback_fires():
    callback = RecordingCallback()
    dspy.configure(callbacks=[callback])

    lm = CallbackLM()
    lm("hello")

    assert [call[0] for call in callback.calls] == ["start", "end"]


def test_language_model_callbacks_fire_on_cache_hit():
    callback = RecordingCallback()
    lm = CallbackLM(cache=True, callbacks=[callback])

    first = lm("hello")
    second = lm("hello")

    assert lm.forward_calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert [call[0] for call in callback.calls] == ["start", "end", "start", "end"]
    assert callback.calls[-2][3]["request"].messages == [dspy.User("hello")]
    assert callback.calls[-1][2].cache_hit is True


@pytest.mark.asyncio
async def test_language_model_async_callbacks_match_sync_contract():
    callback = RecordingCallback()
    lm = CallbackLM(callbacks=[callback])

    response = await lm.acall("hello")

    assert response.text == "async call 1"
    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[1][2] == response


def test_language_model_callback_receives_normalized_exception():
    callback = RecordingCallback()
    lm = ErrorLM(callbacks=[callback])

    with pytest.raises(dspy.ContextWindowExceededError):
        lm("hello")

    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[1][2] is None
    assert isinstance(callback.calls[1][3], dspy.ContextWindowExceededError)


def test_language_model_stream_construction_errors_emit_callbacks():
    callback = RecordingCallback()
    lm = CallbackLM(callbacks=[callback])

    with pytest.raises(NotImplementedError):
        lm.stream("hello")

    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[1][2] is None
    assert isinstance(callback.calls[1][3], NotImplementedError)


def test_language_model_stream_callbacks_end_after_consumption():
    callback = RecordingCallback()
    lm = StreamLM(callbacks=[callback])

    stream = lm.stream("hello")
    assert callback.calls == []

    events = list(stream)

    assert events[-1].type == "end"
    assert stream.result().text == "hello"
    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[1][2].text == "hello"


@pytest.mark.asyncio
async def test_language_model_async_stream_callbacks_end_after_consumption():
    callback = RecordingCallback()
    lm = StreamLM(callbacks=[callback])

    stream = lm.astream("hello")
    assert callback.calls == []

    events = [event async for event in stream]

    assert events[-1].type == "end"
    assert stream.result().text == "async hello"
    assert [call[0] for call in callback.calls] == ["start", "end"]
    assert callback.calls[1][2].text == "async hello"


def test_with_callbacks_dispatch_treats_language_model_as_lm():
    callback = RecordingCallback()

    class DecoratedLM(CallbackLM):
        @with_callbacks
        def ping(self):
            return "pong"

    lm = DecoratedLM(callbacks=[callback])

    assert lm.ping() == "pong"
    assert [call[0] for call in callback.calls] == ["start", "end"]
