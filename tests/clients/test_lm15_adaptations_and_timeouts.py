"""The native engine on lm15 1.0.0rc2: timeouts, plan()-before-I/O, adaptations, remedies.

Found by the DSPy gauntlet (cmpnd-ai/breaka-your-lm, 2026-09-13): `timeout=`
moved every call to LiteLLM; a slow local model died at lm15's fixed 60 s;
a refusal surfaced only after a failed attempt; the missing-key message
named a RouterConfig a DSPy user cannot pass.
"""

import asyncio
import json

import pytest

import dspy
from dspy._vendor.lm15.testing import FakeResponse, FakeTransport
from dspy._vendor.lm15.transports import StdlibAsyncTransport, StdlibTransport
from dspy.clients.backend_selection import select_backend
from dspy.clients.engines import LiteLLMEngine, LM15Engine
from dspy.clients.engines.lm15_engine import timeouts_for
from dspy.clients.execution import _engine, prepare
from dspy.lm15 import Message, Request, Timeouts


def _wire(text="hello"):
    body = {"id": "c", "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    return FakeResponse(status=200, body=json.dumps(body).encode())


# ─── timeout= stays native and reaches the transport ─────────────────

def test_timeout_is_a_native_setting_now():
    lm = dspy.LM("openai/gpt-4o", timeout=30, api_key="k", cache=False)
    assert select_backend(lm).native


@pytest.mark.parametrize("timeout,expected", [
    (30, Timeouts(read=30.0, write=30.0, pool=30.0)),
    (0.5, Timeouts(read=0.5, write=0.5, pool=0.5)),
    (None, None),
])
def test_timeouts_for_a_number_bounds_every_wait_but_connect(timeout, expected):
    assert timeouts_for(timeout) == expected


def test_timeouts_for_httpx_timeout_maps_each_component():
    httpx = pytest.importorskip("httpx")
    assert timeouts_for(httpx.Timeout(120.0)) == Timeouts(connect=120.0, read=120.0, write=120.0, pool=120.0)
    assert timeouts_for(httpx.Timeout(connect=5.0, read=900.0, write=None, pool=None)) == Timeouts(connect=5.0, read=900.0)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, "10"])
def test_timeouts_for_rejects_what_it_cannot_honour(bad):
    with pytest.raises((TypeError, ValueError)):
        timeouts_for(bad)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_timeout_reaches_the_engines_shared_transport(asynchronous):
    lm = dspy.LM("openai/gpt-4o", timeout=1800, api_key="k", cache=False)
    call = prepare(lm, "hello", None, {})

    def check(backend):
        transport = backend.router._shared_transport()
        assert isinstance(transport, StdlibAsyncTransport if asynchronous else StdlibTransport)
        assert transport._read_timeout == 1800.0 and transport._pool_timeout == 1800.0
        assert transport._connect_timeout == 10.0  # lm15's own connect default stays
        return transport

    async def run_async():
        # Async pools belong to their loop: select and close on the same one.
        backend, _, _ = _engine(lm, call, True)
        transport = check(backend)
        await lm.aclose()
        return transport

    if asynchronous:
        transport = asyncio.run(run_async())
    else:
        backend, _, _ = _engine(lm, call, False)
        transport = check(backend)
        lm.close()
    assert transport._closed


def test_different_timeouts_get_different_engines_and_close_releases_them():
    lm = dspy.LM("openai/gpt-4o", api_key="k", cache=False)
    fast, _, _ = _engine(lm, prepare(lm, "hello", None, {"timeout": 5}), False)
    slow, _, _ = _engine(lm, prepare(lm, "hello", None, {"timeout": 500}), False)
    again, _, _ = _engine(lm, prepare(lm, "hello", None, {"timeout": 5}), False)
    assert fast is again and fast is not slow
    assert fast.router._shared_transport()._read_timeout == 5.0
    assert slow.router._shared_transport()._read_timeout == 500.0
    lm.close()
    assert fast.router._transport is None and slow.router._transport is None


# ─── plan() before I/O: the fallback promise, kept offline ───────────

def test_auto_engine_falls_back_before_io_on_a_refusal(monkeypatch):
    calls = []

    def compatible(engine, lm, request, **context):
        calls.append(request)
        from dspy.clients.call_result import CallResult

        return CallResult(outputs=["compatible"], response_model=lm.model)

    monkeypatch.setattr(LiteLLMEngine, "complete_legacy", compatible)
    lm = dspy.LM("openai/gpt-4o", api_key="k", cache=False)
    # An image inside a tool result has no slot on the Chat Completions wire:
    # lm15 refuses at build time (MAP-13 rule 4b). The refusal is known from
    # plan() with no network, so the call goes to LiteLLM before any I/O.
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    messages = [{"role": "user", "content": "look"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]
    backend, _, _ = _engine(lm, prepare(lm, None, messages, {}), False)
    assert isinstance(backend, LiteLLMEngine)


def test_plan_raises_the_refusal_with_its_field():
    from dspy.lm15 import ImagePart, ToolResultPart

    engine = LM15Engine(model_type="chat")
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    request = Request(model="openai/gpt-4o", messages=(
        Message.user("look"),
        Message.tool((ToolResultPart(id="c1", content=(ImagePart(media_type="image/png", data=png),)),)),
    ))
    with pytest.raises(dspy.lm15.UnsupportedFeatureError) as err:
        engine.plan(request)
    assert err.value.feature == "messages[*].tool_result[c1].content[image]"


def test_plan_needs_no_key():
    engine = LM15Engine(model_type="chat")
    request = Request(model="openai/gpt-4o", messages=(Message.user("hi"),), config=dspy.lm15.Config(top_k=3))
    plan = engine.plan(request)
    assert [(a.field, a.action) for a in plan] == [("config.top_k", "dropped")]


# ─── adaptations ride the history entry ──────────────────────────────

def test_history_entry_records_adaptations(monkeypatch):
    import dspy.clients.execution as execution
    from dspy.lm15 import RouterConfig

    transport = FakeTransport([_wire()])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**{k: v for k, v in kwargs.items() if k != "timeouts"}, "api_keys": {"openai-chat": "fake"}, "transport": transport},
    ))
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    assert lm("hello", top_k=3) == ["hello"]
    entry = lm.history[-1]
    assert [(a.field, a.action, a.asked) for a in entry["adaptations"]] == [("config.top_k", "dropped", 3)]
    assert "top_k" not in json.loads(transport.requests[0].body)
    assert entry["response"].adaptations == entry["adaptations"]


# ─── the remedy is said in DSPy's words ──────────────────────────────

def test_missing_key_remedy_names_dspy_lm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    lm = dspy.LM("openai/gpt-4o", cache=False)
    with pytest.raises(dspy.LMNotConfiguredError) as err:
        lm("hello")
    text = str(err.value)
    assert 'pass api_key="..." to dspy.LM(...)' in text and "RouterConfig" not in text


# ─── review findings, through dspy.LM ────────────────────────────────

def test_simultaneous_first_calls_share_one_pool_and_close_releases_it(monkeypatch):
    import threading

    from dspy.clients.engines import LM15Engine as Engine

    calls = []
    monkeypatch.setattr(Engine, "complete", lambda self, request: calls.append(request) or dspy.lm15.Response(
        None, request.model, dspy.lm15.Message.assistant("ok"), "stop", dspy.lm15.Usage()))
    lm = dspy.LM("openai/gpt-4o", api_key="k", cache=False, timeout=30)
    barrier = threading.Barrier(8)
    transports: set[int] = set()

    def go():
        barrier.wait()
        assert lm("hello") == ["ok"]
        backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
        transports.add(id(backend.router._shared_transport()))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(transports) == 1
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    transport = backend.router._shared_transport()
    lm.close()
    assert transport._closed


def test_stop_word_split_across_stream_chunks_is_honoured(monkeypatch):
    import dspy.clients.execution as execution
    from dspy.lm15 import RouterConfig

    frames = [
        'event: response.created\ndata: {"type":"response.created","response":{"id":"r","model":"gpt-5"}}\n\n',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","content":[]}}\n\n',
    ]
    for piece in ("alpha ", "S", "T", "O", "P", " beta"):
        frames.append("event: response.output_text.delta\ndata: " + json.dumps({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": piece}) + "\n\n")
    frames.append('event: response.completed\ndata: {"type":"response.completed","response":{"id":"r","model":"gpt-5","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":9}}}\n\n')
    transport = FakeTransport([FakeResponse(status=200, body="".join(frames).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**{k: v for k, v in kwargs.items() if k != "timeouts"}, "api_keys": {"openai": "fake"}, "transport": transport},
    ))
    lm = dspy.LM("openai/gpt-5", model_type="responses", cache=False)
    outputs = lm("hello", stop=["STOP"])
    assert outputs[0]["text"] == "alpha "
    assert [a.action for a in lm.history[-1]["adaptations"]] == ["client_side"]


def test_plan_reads_no_stored_login(monkeypatch):
    from dspy._vendor.lm15 import access

    class Forbidden(dict):
        def get(self, key, default=None):
            raise AssertionError("engine selection must not read a stored login")

    monkeypatch.setattr(access, "_CREDENTIAL_LOADERS", Forbidden())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lm = dspy.LM("anthropic/claude-sonnet-4-5", cache=False)
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {"seed": 7}), False)
    assert isinstance(backend, LM15Engine)  # selected without a key, without a login


def test_stop_word_spanning_two_text_parts_is_honoured(monkeypatch):
    import dspy.clients.execution as execution
    from dspy.lm15 import RouterConfig

    frames = [
        'event: response.created\ndata: {"type":"response.created","response":{"id":"r","model":"gpt-5"}}\n\n',
        'event: response.output_item.added\ndata: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","content":[]}}\n\n',
        "event: response.output_text.delta\ndata: " + json.dumps({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "alpha S"}) + "\n\n",
        "event: response.output_text.delta\ndata: " + json.dumps({"type": "response.output_text.delta", "output_index": 0, "content_index": 1, "delta": "TOP beta"}) + "\n\n",
        'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r","model":"gpt-5","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":9}}}\n\n',
    ]
    transport = FakeTransport([FakeResponse(status=200, body="".join(frames).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**{k: v for k, v in kwargs.items() if k != "timeouts"}, "api_keys": {"openai": "fake"}, "transport": transport},
    ))
    lm = dspy.LM("openai/gpt-5", model_type="responses", cache=False)
    assert lm("hello", stop=["STOP"])[0]["text"] == "alpha "
