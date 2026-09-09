"""Offline checks for the coordinated LM engine path."""

import json
import zipfile
from pathlib import Path

import pytest

import dspy
from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
from dspy.clients.engines import LiteLLMEngine
from dspy.lm15 import Config, Message, Request, Response, RouterConfig
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import track_usage


class DualResponse(FakeResponse):
    async def __aenter__(self):
        source = self

        class AsyncResponse:
            def __getattr__(self, name):
                return getattr(source, name)

            async def read(self):
                return source.read()

            async def __aiter__(self):
                for chunk in source:
                    yield chunk

        return AsyncResponse()

    async def __aexit__(self, *exc):
        pass


def wire(text="hello", status=200):
    body = {"id": "response-1", "model": "gpt-4o-mini", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
    ], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
    if status != 200:
        body = {"error": {"message": "try later", "type": "authentication_error" if status == 401 else "rate_limit_error"}}
    return DualResponse(status=status, body=json.dumps(body).encode())


def native_transport(monkeypatch, responses):
    import dspy.clients.execution as execution

    transport = FakeTransport(responses)
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**kwargs, "api_keys": {"openai-chat": "fake"}, "transport": transport},
    ))
    return transport


def test_default_native_cache_and_typed_request(monkeypatch):
    transport = native_transport(monkeypatch, [wire(), wire("typed")])
    lm = dspy.LM("openai/gpt-4o-mini")
    with track_usage() as usage:
        assert lm("hello") == ["hello"]
        assert lm("hello") == ["hello"]
    assert len(transport.requests) == 1
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 3
    assert lm.history[-1]["usage"] == {}
    request = Request(model=lm.model, messages=(Message.user("explicit"),), config=Config(max_tokens=12))
    response = lm(request)
    assert isinstance(response, Response)
    assert response.text == "typed"
    assert lm.history[-1]["request"] is request
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_async_native_cache_and_history(monkeypatch):
    transport = native_transport(monkeypatch, [wire("async")])
    lm = dspy.LM("openai/gpt-4o-mini")
    try:
        with track_usage() as usage:
            assert await lm.acall("hello") == ["async"]
            assert await lm.acall("hello") == ["async"]
        assert len(transport.requests) == 1
        assert len(lm.history) == 2
        assert usage.get_total_tokens()[lm.model]["total_tokens"] == 3
    finally:
        await lm.aclose()


def test_native_n_fans_out_but_bookkeeping_is_one_call(monkeypatch):
    transport = native_transport(monkeypatch, [wire("first"), wire("second")])

    class Trace(BaseCallback):
        def __init__(self):
            self.events = []

        def on_lm_start(self, call_id, instance, inputs):
            self.events.append(("start", call_id))

        def on_lm_end(self, call_id, outputs, exception):
            self.events.append(("end", call_id))

    trace = Trace()
    lm = dspy.LM("openai/gpt-4o-mini", callbacks=[trace])
    with track_usage() as usage:
        assert lm("hello", n=2) == ["first", "second"]
    assert len(transport.requests) == 2
    assert len(lm.history) == 1
    assert trace.events[1] == ("end", trace.events[0][1])
    assert len(trace.events) == 2
    assert usage.get_total_tokens()[lm.model]["total_tokens"] == 6
    assert lm("hello", n=2) == ["first", "second"]
    assert len(transport.requests) == 2


def test_native_retries_without_backend_fallback(monkeypatch):
    import dspy.clients.execution as execution

    transport = native_transport(monkeypatch, [wire(status=429), wire("recovered")])
    delays = []
    monkeypatch.setattr(execution.time, "sleep", delays.append)
    lm = dspy.LM("openai/gpt-4o-mini", num_retries=1, cache=False)
    assert lm("hello") == ["recovered"]
    assert len(transport.requests) == 2
    assert delays == [1]


def test_provider_auth_error_does_not_switch_backend(monkeypatch):
    native_transport(monkeypatch, [wire(status=401)])

    def forbidden(*args, **kwargs):
        pytest.fail("An authentication failure must not invoke LiteLLM")

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", forbidden)
    with pytest.raises(dspy.LMAuthError):
        dspy.LM("openai/gpt-4o-mini", cache=False)("hello")


def test_unrepresentable_ordinary_options_choose_compatibility_before_io(monkeypatch):
    transport = native_transport(monkeypatch, [])
    from dspy.clients.call_result import CallResult

    calls = []

    def complete(engine, lm, request, **context):
        calls.append(request)
        return CallResult(outputs=["compatible"], response_model=lm.model)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", complete)
    assert dspy.LM("openai/gpt-4o-mini")("hello", prediction={"type": "content", "content": "hello"}) == ["compatible"]
    assert calls[0]["prediction"]["content"] == "hello"
    assert not transport.requests


@pytest.mark.asyncio
async def test_legacy_plugin_sync_async_preserve_forward_inputs():
    from dspy.dsp.utils.utils import dotdict

    class Plugin(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            self.received = prompt, messages, kwargs
            return dotdict(model=self.model, usage={"total_tokens": 1}, choices=[
                dotdict(message=dotdict(content="ok"), finish_reason="stop")
            ])

        async def aforward(self, **kwargs):
            return self.forward(**kwargs)

    plugin = Plugin("custom")
    with pytest.warns(FutureWarning, match="3.5"):
        assert plugin("hello") == ["ok"]
    assert plugin.received == ("hello", None, {})
    assert await plugin.acall("hello") == ["ok"]
    assert len(plugin.history) == 2


def test_new_native_cache_uses_restricted_plain_data(monkeypatch, tmp_path):
    transport = native_transport(monkeypatch, [wire()])
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=tmp_path, restrict_pickle=True)
    try:
        lm = dspy.LM("openai/gpt-4o-mini")
        assert lm("hello") == ["hello"]
        assert lm("hello") == ["hello"]
        assert len(transport.requests) == 1
    finally:
        dspy.cache.disk_cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case_index", range(6))
async def test_old_disk_cache_hits_before_native_routing(case_index, monkeypatch, tmp_path):
    from tests.clients.test_lm_migration_compatibility import plain

    fixtures = Path(__file__).parent / "fixtures" / "lm_3_3_0"
    case = json.loads((fixtures / "manifest.json").read_text())["cases"][case_index]
    with zipfile.ZipFile(fixtures / f"{case['name']}.zip") as archive:
        archive.extractall(tmp_path)
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=tmp_path)
    import dspy.clients.execution as execution

    def forbidden(*args, **kwargs):
        pytest.fail("An old disk hit must not even initialize an engine")

    monkeypatch.setattr(execution, "_engine", forbidden)
    try:
        lm = dspy.LM(**case["init"])
        output = lm(**case["call"]) if case["mode"] == "sync" else await lm.acall(**case["call"])
        assert plain(output) == case["outputs"]
        assert lm.history[-1]["usage"] == {}
    finally:
        dspy.cache.disk_cache.close()


@pytest.mark.asyncio
async def test_native_streaming_reaches_existing_listener(monkeypatch):
    payloads = [{"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                for text in ("[[ ## answer ## ]]\\n", "native", "\\n\\n[[ ## completed ## ]]")]
    payloads.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
    chunks = [f"data: {json.dumps(payload)}\\n\\n".encode() for payload in payloads]
    chunks.append(b"data: [DONE]\\n\\n")
    transport = native_transport(monkeypatch, [FakeResponse(status=200, body=b"", chunks=chunks)])
    lm = dspy.LM("openai/gpt-4o-mini")
    predict = dspy.Predict("question -> answer")
    streamer = dspy.streamify(predict, stream_listeners=[dspy.streaming.StreamListener("answer")])
    with dspy.context(lm=lm):
        values = [value async for value in streamer(question="hello")]
        cached = [value async for value in streamer(question="hello")]
    assert values[-1].answer == "native"
    assert "".join(value.chunk for value in values if isinstance(value, dspy.streaming.StreamResponse)) == "native"
    assert len(cached) == 1 and cached[0].answer == "native"
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_partial_stream_is_not_retried_and_reports_callback_error():
    from dspy.lm15 import StreamDeltaEvent, StreamStartEvent, TextDelta

    class Engine:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            raise AssertionError("Expected a streaming call")

        def stream(self, request):
            self.calls += 1
            yield StreamStartEvent(model=request.model)
            yield StreamDeltaEvent(TextDelta("partial"))
            raise dspy.LMTransportError("stream broke")

    class Sink:
        async def send(self, chunk):
            pass

    class Trace(BaseCallback):
        error = None

        def on_lm_end(self, call_id, outputs, exception):
            self.error = exception

    engine = Engine()
    trace = Trace()
    lm = dspy.LM("custom", engine=engine, callbacks=[trace], num_retries=3)
    import anyio

    with dspy.context(send_stream=Sink()):
        with pytest.raises(dspy.LMTransportError, match="stream broke"):
            await anyio.to_thread.run_sync(lambda: lm("hello"))
    assert engine.calls == 1
    assert isinstance(trace.error, dspy.LMTransportError)
    assert lm.history == []


def test_dummy_uses_canonical_engine_without_changing_script_behavior():
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([{"answer": "one"}, {"answer": "two"}])
    with dspy.context(lm=lm):
        predict = dspy.Predict("question -> answer")
        assert predict(question="same").answer == "one"
        assert predict(question="same").answer == "two"
    assert len(lm.history) == 2
