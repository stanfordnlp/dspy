"""Failed streams, cancellation, and post-completion work cannot replay a generation."""

import asyncio

import anyio
import pytest

import dspy
from dspy import lm15
from dspy.clients.engines.lifecycle import aclosing_stream, closing_stream
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import track_usage
from tests.clients.test_engine_errors import AsyncEngine, Engine, Sink, invoke


def events(case):
    start = lm15.StreamStartEvent()  # a missing model/id is legitimate
    text = lm15.StreamDeltaEvent(lm15.TextDelta("partial"))
    end = lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage(input_tokens=2, output_tokens=1))
    error = lm15.StreamErrorEvent(lm15.ErrorDetail(code="auth", message="denied"))
    return {
        "empty": [], "missing-end": [start, text], "missing-start": [text, end],
        "duplicate-start": [start, start], "duplicate-end": [start, text, end, end],
        "after-end": [start, text, end, text], "unknown-event": [start, object()],
        "error-first": [error], "error-midstream": [start, text, error],
    }[case]


class StreamEngine(Engine):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.closed = 0

    def stream(self, request):
        self.calls += 1
        try:
            yield from events(self.case)
        finally:
            self.closed += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("case", ["empty", "missing-end", "missing-start", "duplicate-start", "duplicate-end", "after-end", "unknown-event", "error-first", "error-midstream"])
async def test_all_streams_are_validated_and_failures_are_not_cached(case, asynchronous):
    engine = StreamEngine(case)
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=True, num_retries=2)
    expected = dspy.LMAuthError if case.startswith("error-") else dspy.LMStreamAssemblyError
    with dspy.context(send_stream=Sink()), track_usage() as tracker:
        for _ in range(2):
            with pytest.raises(expected) as caught:
                await invoke(lm, asynchronous, "hello")
            if case == "missing-end":
                assert caught.value.partial.text == "partial"
    assert engine.calls == 2  # no retries and no partial-success cache hits
    assert engine.closed == 2
    assert not lm.history
    assert not dspy.cache.memory_cache
    usage = tracker.get_total_tokens()
    if case in ("duplicate-end", "after-end"):
        assert usage[lm.model]["total_tokens"] == 6
    else:
        assert usage == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_transient_stream_open_failure_retries_but_failure_after_output_does_not(asynchronous):
    class Transient(Engine):
        def stream(self, request):
            self.calls += 1
            if self.calls == 1:
                raise lm15.RateLimitError("wait", retry_after=0)
            yield lm15.StreamStartEvent()
            yield lm15.StreamDeltaEvent(lm15.TextDelta("ok"))
            yield lm15.StreamEndEvent(finish_reason="stop")

    engine = Transient()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=1)
    with dspy.context(send_stream=Sink()):
        assert await invoke(lm, asynchronous, "hello") == ["ok"]
    assert engine.calls == 2

    class Partial(Engine):
        def stream(self, request):
            self.calls += 1
            yield lm15.StreamStartEvent()
            yield lm15.StreamDeltaEvent(lm15.TextDelta("partial"))
            raise lm15.RateLimitError("wait", retry_after=0)

    engine = Partial()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=1)
    with dspy.context(send_stream=Sink()), pytest.raises(dspy.LMRateLimitError):
        await invoke(lm, asynchronous, "hello")
    assert engine.calls == 1


class BrokenClose:
    def __init__(self, primary):
        self.primary = primary
        self.cleanup = RuntimeError("cleanup failure")
        self.closed = 0

    def __iter__(self):
        return self

    def __next__(self):
        raise self.primary

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.primary

    def close(self):
        self.closed += 1
        raise self.cleanup

    async def aclose(self):
        self.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_primary_error_and_cancellation_survive_cleanup(asynchronous, cancelled):
    primary = asyncio.CancelledError("cancelled") if cancelled else lm15.AuthError("denied")
    source = BrokenClose(primary)
    if asynchronous:
        with pytest.raises(type(primary)) as caught:
            async with aclosing_stream(source):
                await source.__anext__()
    else:
        with pytest.raises(type(primary)) as caught:
            with closing_stream(source):
                next(source)
    assert caught.value is primary
    assert primary.cleanup_errors == (source.cleanup,)
    assert source.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_cleanup_diagnostics_survive_public_projection(asynchronous):
    source = BrokenClose(lm15.AuthError("denied"))

    class Backend(Engine):
        def stream(self, request):
            return source

    class AsyncBackend(AsyncEngine):
        def stream(self, request):
            return source

    lm = dspy.LM("custom", engine=Backend(), async_engine=AsyncBackend(None), cache=False, num_retries=0)
    with dspy.context(send_stream=Sink()), pytest.raises(dspy.LMAuthError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.__cause__ is source.primary
    assert caught.value.cleanup_errors == (source.cleanup,)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("phase", ["pricing", "cache", "output-conversion"])
async def test_post_completion_failure_does_not_retry_and_keeps_usage(phase, asynchronous, monkeypatch):
    import dspy.clients.execution as execution

    engine = Engine()
    errors = []

    class Callback(BaseCallback):
        def on_lm_end(self, call_id, outputs, exception):
            errors.append(exception)

    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=True,
                 num_retries=3, callbacks=[Callback()])
    failure = dspy.LMRateLimitError("injected post-completion failure", retry_after=0)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(execution, {"pricing": "_price_result", "cache": "_store", "output-conversion": "_result"}[phase], fail)
    with track_usage() as tracker, pytest.raises(dspy.LMRateLimitError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value is failure
    assert engine.calls == 1
    assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert errors == [failure]
    assert not lm.history
    assert not dspy.cache.memory_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_completed_stream_cleanup_failure_does_not_retry_and_keeps_usage(asynchronous):
    failure = lm15.TransportError("close failed")

    class Source:
        def __init__(self):
            self.iterator = iter([lm15.StreamStartEvent(), lm15.StreamDeltaEvent(lm15.TextDelta("ok")),
                                  lm15.StreamEndEvent(finish_reason="stop", usage=lm15.Usage(input_tokens=2, output_tokens=1))])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.iterator)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                raise StopAsyncIteration from None

        def close(self):
            raise failure

        async def aclose(self):
            self.close()

    class Backend(Engine):
        def stream(self, request):
            self.calls += 1
            return Source()

    class AsyncBackend(AsyncEngine):
        def stream(self, request):
            return self.sync.stream(request)

    engine = Backend()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncBackend(engine), cache=True, num_retries=2)
    with dspy.context(send_stream=Sink()), track_usage() as tracker, pytest.raises(dspy.LMTransportError):
        await invoke(lm, asynchronous, "hello")
    assert engine.calls == 1
    assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert not dspy.cache.memory_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unnamed_tool_call_preserves_partial_usage(asynchronous):
    class Backend(Engine):
        def stream(self, request):
            yield lm15.StreamStartEvent()
            yield lm15.StreamDeltaEvent(lm15.ToolCallDelta(input="{}", id="call-1"))
            yield lm15.StreamEndEvent(finish_reason="tool_call", usage=lm15.Usage(input_tokens=2, output_tokens=1))

    engine = Backend()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=True, num_retries=2)
    with dspy.context(send_stream=Sink()), track_usage() as tracker, pytest.raises(dspy.LMStreamAssemblyError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.partial is not None
    assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert not dspy.cache.memory_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_parser_bug_is_not_hidden_by_adapter_fallback(asynchronous):
    class Broken(dspy.ChatAdapter):
        def parse(self, signature, completion):
            raise RuntimeError("parser implementation bug")

    engine = Engine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    args = (lm, {}, dspy.Signature("question -> answer"), [], {"question": "hi"})
    with pytest.raises(RuntimeError, match="parser implementation bug"):
        if asynchronous:
            await Broken().acall(*args)
        else:
            Broken()(*args)
    assert engine.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_parse_failure_after_visible_stream_never_falls_back(asynchronous):
    # Engine returns plain 'ok', which is not a valid ChatAdapter answer.
    engine = Engine()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    args = (lm, {}, dspy.Signature("question -> answer"), [], {"question": "hi"})
    adapter = dspy.ChatAdapter()
    with dspy.context(send_stream=Sink()), pytest.raises(dspy.AdapterParseError):
        if asynchronous:
            await adapter.acall(*args)
        else:
            await anyio.to_thread.run_sync(lambda: adapter(*args))
    assert engine.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_json_schema_fallback_is_only_for_setup(asynchronous, monkeypatch):
    import dspy.adapters.json_adapter as json_adapter

    class Backend(Engine):
        supports_response_schema = True
        supported_params = frozenset({"response_format"})

        def complete(self, request):
            response = super().complete(request)
            self.format = request.config.response_format
            return lm15.Response(None, request.model, lm15.Message.assistant('{"answer":"ok"}'), "stop", response.usage)

    engine = Backend()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False)
    adapter = dspy.JSONAdapter()
    args = (lm, {}, dspy.Signature("question -> answer"), [], {"question": "hi"})

    def bad_schema(*args):
        raise ValueError("schema cannot be represented")

    monkeypatch.setattr(json_adapter, "_get_structured_outputs_response_format", bad_schema)
    result = await adapter.acall(*args) if asynchronous else adapter(*args)
    assert result == [{"answer": "ok"}]
    assert engine.calls == 1
    assert engine.format == {"type": "json_object"}

    def bug(*args):
        raise RuntimeError("schema builder bug")

    monkeypatch.setattr(json_adapter, "_get_structured_outputs_response_format", bug)
    with pytest.raises(RuntimeError, match="schema builder bug"):
        if asynchronous:
            await adapter.acall(*args)
        else:
            adapter(*args)
    assert engine.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_materializing_backend_keeps_completed_usage_on_cleanup_failure(asynchronous):
    from dspy._vendor.lm15.result import amaterialize_response, materialize_response

    class Source(BrokenClose):
        def __init__(self):
            super().__init__(None)
            response = lm15.Response(None, "custom", lm15.Message.assistant("ok"), "stop", lm15.Usage(input_tokens=2, output_tokens=1))
            self.iterator = iter(lm15.response_to_events(response))

        def __next__(self):
            return next(self.iterator)

        async def __anext__(self):
            try:
                return next(self.iterator)
            except StopIteration:
                raise StopAsyncIteration from None

    class Backend(Engine):
        def source(self):
            self.calls += 1
            return Source()

        def complete(self, request):
            return materialize_response(self.source(), request)

    class AsyncBackend(AsyncEngine):
        async def complete(self, request):
            return await amaterialize_response(self.sync.source(), request)

    engine = Backend()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncBackend(engine), cache=True, num_retries=2)
    with track_usage() as tracker, pytest.raises(dspy.LMStreamAssemblyError) as caught:
        await invoke(lm, asynchronous, "hello")
    assert caught.value.partial.text == "ok"
    assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert engine.calls == 1
    assert not dspy.cache.memory_cache


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_compatibility_stream_keeps_usage_if_sdk_close_fails(asynchronous, monkeypatch):
    import litellm

    class Source:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

        async def aclose(self):
            raise lm15.TransportError("cleanup failed")

    calls = []

    async def complete(**kwargs):
        calls.append(kwargs)
        return Source()

    raw = litellm.ModelResponse(model="fake", choices=[{"message": {"role": "assistant", "content": "ok"}}],
                               usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
    monkeypatch.setattr(litellm, "acompletion", complete)
    monkeypatch.setattr(litellm, "stream_chunk_builder", lambda chunks: raw)
    lm = dspy.LM("openai/fake", engine="litellm", cache=True, num_retries=2)
    with dspy.context(send_stream=Sink()), track_usage() as tracker, pytest.raises(dspy.LMTransportError):
        await invoke(lm, asynchronous, "hello")
    assert len(calls) == 1
    assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert not dspy.cache.memory_cache


@pytest.mark.parametrize("adapter,module,text", [
    (dspy.ChatAdapter(), "dspy.adapters.chat_adapter", '[[ ## answer ## ]]\n{"value": 1}'),
    (dspy.JSONAdapter(), "dspy.adapters.json_adapter", '{"answer":{"value":1}}'),
    (dspy.XMLAdapter(), "dspy.adapters.xml_adapter", '<answer>{"value":1}</answer>'),
])
def test_unexpected_field_parser_errors_are_not_labelled_as_model_output_errors(adapter, module, text, monkeypatch):
    def fail(*args):
        raise RuntimeError("validator implementation bug")

    monkeypatch.setattr(module + ".parse_value", fail)
    with pytest.raises(RuntimeError, match="validator implementation bug"):
        adapter.parse(dspy.Signature("question -> answer"), text)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_usage_reporting_failure_does_not_replace_primary_error(asynchronous):
    reporting_error = RuntimeError("tracker failed")

    class Tracker:
        def add_usage(self, model, usage):
            raise reporting_error

    class Backend(Engine):
        def complete(self, request):
            if self.calls:
                self.error = lm15.AuthError("denied")
            return super().complete(request)

    engine = Backend()
    lm = dspy.LM("custom", engine=engine, async_engine=AsyncEngine(engine), cache=False, num_retries=0)
    with dspy.context(usage_tracker=Tracker()), pytest.raises(dspy.LMAuthError) as caught:
        await invoke(lm, asynchronous, "hello", n=2)
    assert caught.value.usage_errors == (reporting_error,)
    assert isinstance(caught.value.__cause__, lm15.AuthError)
    assert engine.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_direct_native_engine_stream_is_also_guarded(asynchronous, monkeypatch):
    from types import SimpleNamespace

    from dspy.clients.engines import AsyncLM15Engine, LM15Engine

    class Provider:
        provider = "fake"

        def stream(self, request):
            return iter([lm15.StreamStartEvent(), lm15.StreamDeltaEvent(lm15.TextDelta("partial"))])

    request = lm15.Request(model="custom", messages=(lm15.Message.user("hi"),))
    if asynchronous:
        async def source(request):
            for event in Provider().stream(request):
                yield event

        engine = AsyncLM15Engine()
        provider = SimpleNamespace(provider="fake", stream=source)
        monkeypatch.setattr(engine, "_target", lambda *args, **kwargs: (provider, request))
        with pytest.raises(lm15.StreamAssemblyError):
            _ = [event async for event in engine.stream(request)]
    else:
        engine = LM15Engine()
        monkeypatch.setattr(engine, "_target", lambda *args, **kwargs: (Provider(), request))
        with pytest.raises(lm15.StreamAssemblyError):
            list(engine.stream(request))


@pytest.mark.asyncio
async def test_stream_fallback_progress_is_isolated_across_concurrent_calls():
    emitted, release = asyncio.Event(), asyncio.Event()

    class Backend(AsyncEngine):
        calls = 0

        async def complete(self, request):
            self.calls += 1
            return lm15.Response(None, request.model, lm15.Message.assistant('{"answer":"ok"}'), "stop", lm15.Usage())

        async def stream(self, request):
            yield lm15.StreamStartEvent()
            yield lm15.StreamDeltaEvent(lm15.TextDelta('{"answer":"ok"}'))
            await release.wait()
            yield lm15.StreamEndEvent(finish_reason="stop")

    class SignallingSink(Sink):
        async def send(self, chunk):
            emitted.set()

    backend = Backend(None)
    lm = dspy.LM("custom", engine=Engine(), async_engine=backend, cache=False)
    adapter = dspy.ChatAdapter()
    args = (lm, {}, dspy.Signature("question -> answer"), [], {"question": "hi"})

    async def streaming():
        with dspy.context(send_stream=SignallingSink()):
            with pytest.raises(dspy.AdapterParseError):
                await adapter.acall(*args)

    task = asyncio.create_task(streaming())
    try:
        await asyncio.wait_for(emitted.wait(), 5)
        assert await adapter.acall(*args) == [{"answer": "ok"}]
        assert backend.calls == 2  # non-streaming Chat parse failure still gets one JSON fallback
    finally:
        release.set()
        await task
