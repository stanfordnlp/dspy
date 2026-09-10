"""Async execution boundaries exercised against a real, local HTTP transport."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import anyio
import pytest

import dspy
from dspy.clients.engines import AsyncLM15Engine
from dspy.clients.model_metadata import _snapshot
from dspy.lm15 import Message, Response, Usage
from dspy.utils.callback import BaseCallback
from dspy.utils.usage_tracker import track_usage


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    state = {"active": 0, "peak": 0, "calls": 0, "status": 200, "delay": 0.01, "retry_after": "9"}
    lock = threading.Lock()
    entered = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["active"] += 1
                state["calls"] += 1
                state["peak"] = max(state["peak"], state["active"])
                status = state["status"]
                if state.get("recover"):
                    state["status"] = 200
            entered.set()
            time.sleep(state["delay"])
            answer = "A" if "alpha" in json.dumps(data) else "B"
            body = {"model": "fake", "choices": [{"index": 0, "message": {"role": "assistant",
                    "content": f"[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
            if status != 200:
                body = {"error": {"message": "try later", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
            encoded = json.dumps(body).encode()
            streaming = data.get("stream") and status == 200
            if streaming:
                frames = [{"choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
                          for t in ("[[", " ## answer ## ]]\n", answer, "\n\n[[ ## completed ## ]]")]
                frames.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                frames.append({"choices": [], "usage": body["usage"]})
                encoded = "".join("data: " + json.dumps(f) + "\n\n" for f in frames).encode() + b"data: [DONE]\n\n"
            try:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
                if state["retry_after"] is not None:
                    self.send_header("rEtRy-AfTeR", state["retry_after"])
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    state["active"] -= 1

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{http.server_port}/v1", state, entered
    http.shutdown()
    http.server_close()
    worker.join()


def native_lm(endpoint, **kwargs):
    return dspy.LM("openai/fake", engine="lm15", api_base=endpoint[0], api_key="fake",
                   cache=False, num_retries=0, **kwargs)


@pytest.mark.asyncio
async def test_concurrent_programs_and_usage_are_isolated(endpoint):
    lm = native_lm(endpoint)
    endpoint[1]["delay"] = 0.1
    program = dspy.Predict("question -> answer")

    async def call(question):
        with dspy.context(lm=lm, track_usage=True, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)):
            return await program.acall(question=question)

    try:
        results = await asyncio.wait_for(asyncio.gather(call("alpha"), call("beta"), call("alpha")), 10)
        assert [r.answer for r in results] == ["A", "B", "A"]
        assert endpoint[1]["peak"] > 1
        assert all(r.get_lm_usage()[lm.model]["total_tokens"] == 10 for r in results)
        assert len(lm.history) == 3
    finally:
        await lm.aclose()


@pytest.mark.asyncio
async def test_native_async_program_streaming(endpoint):
    lm = native_lm(endpoint)
    program = dspy.Predict("question -> answer")
    stream = dspy.streamify(program, is_async_program=True,
                           stream_listeners=[dspy.streaming.StreamListener("answer")])
    try:
        with dspy.context(lm=lm, track_usage=True):
            async def collect():
                return [item async for item in stream(question="alpha")]
            items = await asyncio.wait_for(collect(), 10)
        assert items[-1].answer == "A"
        assert "".join(i.chunk for i in items if isinstance(i, dspy.streaming.StreamResponse)) == "A"
        assert items[-1].get_lm_usage()[lm.model]["total_tokens"] == 10
    finally:
        await lm.aclose()


@pytest.mark.asyncio
async def test_http_cancellation_releases_pool_without_history(endpoint):
    lm = native_lm(endpoint)
    endpoint[1]["delay"] = 0.3
    task = asyncio.create_task(lm.acall("alpha"))
    try:
        assert await asyncio.to_thread(endpoint[2].wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        backend = next(iter(lm._engine_store.values()))
        provider = next(iter(backend._providers.values()))
        assert provider.transport.pool_stats()["in_use"] == 0
        assert not lm.history
        assert endpoint[1]["calls"] == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await lm.aclose()


def test_same_lm_on_successive_loops(endpoint):
    lm = native_lm(endpoint)

    async def once():
        try:
            return await lm.acall("alpha")
        finally:
            await lm.aclose()

    assert asyncio.run(once()) == asyncio.run(once())
    assert endpoint[1]["calls"] == 2
    assert lm._engine_store == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("header,expected", [("9", 9), ("invalid", None), (None, None)])
async def test_http_retry_header_reaches_dspy_error(endpoint, asynchronous, streaming, header, expected):
    lm = native_lm(endpoint)
    endpoint[1].update(status=429, retry_after=header)

    class Sink:
        async def send(self, chunk):
            raise AssertionError("Rejected stream must not emit chunks")

    try:
        with dspy.context(send_stream=Sink() if streaming else None):
            with pytest.raises(dspy.LMRateLimitError) as caught:
                if asynchronous:
                    await lm.acall("alpha")
                else:
                    await anyio.to_thread.run_sync(lambda: lm("alpha"))
        assert caught.value.retry_after == expected
        assert caught.value.status == 429
        assert endpoint[1]["calls"] == 1
    finally:
        await lm.aclose()


@pytest.mark.asyncio
async def test_retry_loop_uses_preserved_header(endpoint, monkeypatch):
    import dspy.clients.execution as execution

    lm = native_lm(endpoint)
    lm.num_retries = 1
    endpoint[1].update(status=429, recover=True)
    delays = []
    original = execution._delay

    def delay(exc, attempt):
        delays.append(original(exc, attempt))
        return 0  # verify the selected delay without waiting nine seconds

    monkeypatch.setattr(execution, "_delay", delay)
    try:
        await lm.acall("alpha")
        assert delays == [9]
        assert endpoint[1]["calls"] == 2
        assert len(lm.history) == 1
    finally:
        await lm.aclose()


@pytest.mark.asyncio
async def test_async_model_listing_preserves_retry_header():
    from dspy._vendor.lm15.providers.async_base import AsyncOpenAIChatLM
    from dspy._vendor.lm15.providers.base import HttpResponse
    from dspy.lm15 import RateLimitError

    async def send(self, request):
        return HttpResponse(429, "Too Many Requests", [("Retry-After", "9")],
                            b'{"error":{"type":"rate_limit_error","message":"wait"}}')

    provider = AsyncOpenAIChatLM(api_key="fake")
    try:
        with patch.object(AsyncOpenAIChatLM, "_send", send):
            with pytest.raises(RateLimitError) as caught:
                await provider.list_models()
        assert caught.value.retry_after == 9
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [dspy.ChatAdapter, dspy.JSONAdapter])
async def test_adapter_metadata_load_runs_off_loop_once(adapter, monkeypatch):
    import io

    import dspy.clients.model_metadata as metadata

    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    monkeypatch.setattr(metadata, "_data", None)
    monkeypatch.setattr(metadata, "_source", {})
    entered, release = threading.Event(), threading.Event()
    threads = []
    payload = json.dumps(_snapshot()).encode()
    loop_thread = threading.get_ident()

    def download(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(5)
        return io.BytesIO(payload)

    monkeypatch.setattr(metadata.urllib.request, "urlopen", download)

    async def complete(self, request):
        text = '{"answer":"A"}' if adapter is dspy.JSONAdapter else "[[ ## answer ## ]]\nA"
        return Response(None, request.model, Message.assistant(text), "stop", Usage(input_tokens=1, output_tokens=1))

    monkeypatch.setattr(AsyncLM15Engine, "complete", complete)
    lm = dspy.LM("openai/gpt-4o", engine="lm15", cache=False)
    program = dspy.Predict("question -> answer")

    async def call():
        with dspy.context(lm=lm, adapter=adapter()):
            return await program.acall(question="alpha")

    tasks = [asyncio.create_task(call()), asyncio.create_task(call())]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert len(threads) == 1 and threads[0] != loop_thread
        release.set()
        assert [r.answer for r in await asyncio.gather(*tasks)] == ["A", "A"]
        assert len(threads) == 1
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await lm.aclose()


@pytest.mark.asyncio
async def test_cost_lookup_runs_off_loop(endpoint, monkeypatch):
    import dspy.clients.costs as costs

    lm = native_lm(endpoint)
    threads = []
    loop_thread = threading.get_ident()

    def info(*args):
        threads.append(threading.get_ident())
        return {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}

    monkeypatch.setattr(costs, "model_info", info)
    try:
        await lm.acall("alpha")
        assert threads and all(t != loop_thread for t in threads)
        assert lm.history[-1]["cost"] == pytest.approx(0.013)
    finally:
        await lm.aclose()


class Candidates:
    def __init__(self, phase):
        self.calls = 0
        self.phase = phase
        self.waiting = asyncio.Event()

    async def complete(self, request):
        self.calls += 1
        if self.calls == 2:
            self.waiting.set()
            if self.phase == "backoff":
                raise dspy.LMRateLimitError("wait", retry_after=60)
            await asyncio.Event().wait()
        return Response(None, request.model, Message.assistant("ok"), "stop", Usage(input_tokens=7, output_tokens=3))

    async def stream(self, request):
        from dspy.lm15 import response_to_events

        for event in response_to_events(await self.complete(request)):
            yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("phase", ["provider", "backoff", "pricing", "cache"])
async def test_cancellation_accounts_completed_candidates_once(phase, streaming, monkeypatch):
    import dspy.clients.execution as execution

    engine = Candidates(phase)
    lm = dspy.LM("custom", engine=object_engine(), async_engine=engine, cache=phase == "cache", num_retries=3)
    worker_entered, release = threading.Event(), threading.Event()
    worker_threads = []
    loop_thread = threading.get_ident()

    def block(*args):
        worker_threads.append(threading.get_ident())
        worker_entered.set()
        release.wait(5)

    if phase == "pricing":
        monkeypatch.setattr(execution, "_price_result", block)
    if phase == "cache":
        monkeypatch.setattr(execution, "_store", block)
    calls_stored = []
    if phase != "cache":
        monkeypatch.setattr(execution, "_store", lambda *args: calls_stored.append(args))
    errors = []

    class Trace(BaseCallback):
        def on_lm_end(self, call_id, outputs, exception):
            errors.append(exception)

    class Sink:
        async def send(self, chunk):
            pass

    lm.callbacks = [Trace()]
    with dspy.context(send_stream=Sink() if streaming else None), track_usage() as tracker:
        task = asyncio.create_task(lm.acall("hello", n=1 if phase == "cache" else 2))
        try:
            if phase in ("provider", "backoff"):
                await asyncio.wait_for(engine.waiting.wait(), 5)
                await asyncio.sleep(0)  # let the caller enter its retry backoff
            else:
                assert await asyncio.to_thread(worker_entered.wait, 5)
                assert worker_threads[0] != loop_thread
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 10
            assert not lm.history
            assert len(errors) == 1 and isinstance(errors[0], asyncio.CancelledError)
            assert not calls_stored
            assert engine.calls == (2 if phase in ("provider", "backoff") else 1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_during_final_stream_delivery_preserves_usage():
    engine = Candidates("provider")
    lm = dspy.LM("custom", engine=object_engine(), async_engine=engine, cache=False)
    waiting = asyncio.Event()

    class Sink:
        async def send(self, chunk):
            if chunk.choices[0].finish_reason is not None:
                waiting.set()
                await asyncio.Event().wait()

    with dspy.context(send_stream=Sink()), track_usage() as tracker:
        task = asyncio.create_task(lm.acall("hello"))
        try:
            await asyncio.wait_for(waiting.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 10
            assert not lm.history
            assert engine.calls == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def object_engine():
    class Engine:
        def complete(self, request):
            raise AssertionError("Async call must not use sync engine")
    return Engine()


@pytest.mark.asyncio
async def test_async_cache_and_preparation_use_workers(endpoint, monkeypatch):
    import dspy.clients.execution as execution

    lm = native_lm(endpoint)
    lm.cache = True
    threads = {name: [] for name in ("prepare", "_cached", "_store")}
    loop_thread = threading.get_ident()
    for name in threads:
        original = getattr(execution, name)

        def record(*args, _name=name, _original=original, **kwargs):
            threads[_name].append(threading.get_ident())
            return _original(*args, **kwargs)

        monkeypatch.setattr(execution, name, record)
    try:
        with track_usage() as tracker:
            first = await lm.acall("alpha")
            assert await lm.acall("alpha") == first
        assert tracker.get_total_tokens()[lm.model]["total_tokens"] == 10
        assert endpoint[1]["calls"] == 1
        assert all(values and all(t != loop_thread for t in values) for values in threads.values())
        assert lm.history[-1]["usage"] == {}
    finally:
        await lm.aclose()
