"""The gauntlet findings (cmpnd-ai/breaka-your-lm results/findings.md,
2026-09-13) still open on main after #10409, #10441 and #10442, each pinned
from a reproduction against the current code."""

import asyncio
import gc
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pydantic
import pytest

import dspy
from dspy.clients.legacy_requests import _strict_json_schema
from dspy.lm15 import AuthError
from dspy.utils.exceptions import LMInvalidRequestError


@pytest.fixture
def server():
    state = {"mode": "plain"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = {"id": "x", "object": "chat.completion", "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            if state["mode"] == "responses-only":
                data = json.dumps({"error": {"message": "Function tools with reasoning_effort are not supported for gpt-5.6-luna "
                                             "in /v1/chat/completions. To use function tools, use /v1/responses",
                                             "type": "invalid_request_error", "code": None}}).encode()
                self.send_response(400)
            else:
                data = json.dumps(body).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/v1", state
    finally:
        http.shutdown()
        http.server_close()
        worker.join()


def _native(base, model="openai/m", **kwargs):
    return dspy.LM(model, engine="lm15", api_key="k", api_base=base, cache=False, num_retries=0, **kwargs)


# ─── 5. errors inside streamify are the program's own exception ───────


class Failing:
    def complete(self, request):
        raise AuthError("bad key", provider="x")

    def stream(self, request):
        raise AuthError("bad key", provider="x")
        yield  # a generator that fails before its first event


def test_streamify_raises_the_public_error_itself():
    lm = dspy.LM("custom/failing", engine=Failing(), cache=False, num_retries=0)
    stream = dspy.streamify(dspy.Predict("q -> a"), async_streaming=False)
    with dspy.context(lm=lm), pytest.raises(dspy.LMAuthError) as info:
        for _ in stream(q="x"):
            pass
    assert not hasattr(info.value, "exceptions")  # not a group


@pytest.mark.asyncio
async def test_async_streamify_raises_the_public_error_itself():
    lm = dspy.LM("custom/failing", engine=Failing(), cache=False, num_retries=0)
    stream = dspy.streamify(dspy.Predict("q -> a"), async_streaming=True)
    with dspy.context(lm=lm), pytest.raises(dspy.LMAuthError):
        async for _ in stream(q="x"):
            pass


def test_streamify_keeps_a_group_of_several_failures():
    import builtins

    from dspy.streaming.streamify import _single_failure

    group_class = getattr(builtins, "BaseExceptionGroup", None)
    if group_class is None:  # Python 3.10: anyio's backport
        import exceptiongroup

        group_class = exceptiongroup.BaseExceptionGroup

    group = group_class("two", [ValueError("a"), ValueError("b")])
    assert _single_failure(group) is None
    nested = group_class("outer", [group_class("inner", [KeyError("k")])])
    assert isinstance(_single_failure(nested), KeyError)
    assert _single_failure(ValueError("plain")) is None


# ─── 4. async pools of finished loops are released ────────────────────


def _fds():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None


def test_close_releases_pools_of_closed_loops(server):
    base, _ = server
    lm = _native(base)
    for _ in range(3):
        asyncio.run(lm.acall("hi"))
    lm("hi")
    async_pools = [key for key in lm._engine_store if key[0] is not None]
    assert len(async_pools) == 1 and async_pools[0][0].is_closed()  # earlier loops were evicted as each new one came
    before = _fds()
    lm.close()
    gc.collect()
    assert lm._engine_store == {}
    if before is not None:
        assert _fds() < before  # the dead loops' sockets are closed, not only the sync pool's


def test_a_new_loop_evicts_pools_of_closed_loops(server):
    # A long-lived LM used from many asyncio.run() calls does not keep one
    # pool per finished loop; the next loop's pool replaces them.
    base, _ = server
    lm = _native(base)
    for _ in range(4):
        asyncio.run(lm.acall("hi"))
    assert len([key for key in lm._engine_store if key[0] is not None]) == 1
    lm.close()


@pytest.mark.asyncio
async def test_aclose_still_closes_the_live_loop_and_reaps_dead_ones(server):
    base, _ = server
    lm = _native(base)
    await lm.acall("hi")
    await lm.aclose()
    assert lm._engine_store == {}


# ─── 8. a generated schema is shaped for OpenAI strict mode ──────────


class Inner(pydantic.BaseModel):
    note: str = "n/a"


class Record(pydantic.BaseModel):
    name: str
    count: int = 0
    tag: str | None = None
    inner: Inner | None = None
    items: list[Inner] = []


def test_strict_schema_requires_every_property_and_drops_none_defaults():
    schema = _strict_json_schema(Record.model_json_schema())
    assert schema["required"] == ["name", "count", "tag", "inner", "items"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["tag"]  # None default dropped
    assert schema["properties"]["count"]["default"] == 0  # a value default is kept, as the OpenAI SDK keeps it
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["note"] and inner["additionalProperties"] is False
    # An optional submodel is anyOf[$ref, null]; the ref stays a bare ref.
    assert {"$ref": "#/$defs/Inner"} in schema["properties"]["inner"]["anyOf"]
    json.dumps(schema)


def test_strict_schema_unravels_a_ref_with_siblings_and_a_lone_allof():
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/A", "description": "d"},
                                               "b": {"allOf": [{"$ref": "#/$defs/A"}]}},
              "$defs": {"A": {"type": "string"}}}
    out = _strict_json_schema(schema)
    assert out["properties"]["a"] == {"type": "string", "description": "d"}
    assert out["properties"]["b"] == {"$ref": "#/$defs/A"}  # a bare ref stays a ref
    assert out["required"] == ["a", "b"]


def test_response_format_model_is_sent_strict_and_complete(server, monkeypatch):
    base, _ = server
    sent = {}
    from dspy.clients import execution

    original = execution._canonical

    def capture(call, **kwargs):
        request = original(call, **kwargs)
        sent["format"] = request.config.response_format
        return request

    monkeypatch.setattr(execution, "_canonical", capture)
    _native(base)("hi", response_format=Record)
    fmt = sent["format"]
    assert fmt["type"] == "json_schema" and fmt.get("strict") is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]


def test_forced_litellm_responses_path_sends_the_same_strict_schema():
    # engine="litellm" with model_type="responses" converts the legacy body
    # itself; a pydantic response_format must mean the same contract there
    # (greptile on dspy#10451).
    from dspy.clients.legacy_requests import chat_to_responses

    data = chat_to_responses({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}],
                              "response_format": Record})
    fmt = data["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["required"] == ["name", "count", "tag", "inner", "items"]
    assert "default" not in fmt["schema"]["properties"]["tag"]
    assert fmt["schema"]["$defs"]["Inner"]["additionalProperties"] is False


# ─── 14b. an empty model id is refused at construction ────────────────


@pytest.mark.parametrize("model", ["openai/", "anthropic/", ""])
def test_empty_model_id_is_refused(model):
    with pytest.raises(ValueError, match="model"):
        dspy.LM(model, api_key="k")


# ─── 6b. OpenAI's "use /v1/responses" refusal names DSPy's switch ─────


def test_responses_only_refusal_names_model_type(server):
    base, state = server
    state["mode"] = "responses-only"
    lm = _native(base, model="openai/gpt-5.6-luna", max_tokens=16000)
    with pytest.raises(LMInvalidRequestError, match=r"model_type='responses'") as info:
        lm("hi", tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}])
    assert "/v1/responses" in str(info.value)  # the provider's own words are kept
    # A Responses-API LM gets no hint: it is already there.
    state["mode"] = "plain"


def test_hint_is_added_once_and_only_for_chat():
    from dspy.clients.execution import _hint_responses_api

    exc = LMInvalidRequestError("use /v1/responses", model="m")
    lm = dspy.LM("openai/gpt-5.6-luna", api_key="k", model_type="responses", max_tokens=16000)
    _hint_responses_api(lm, exc)
    assert "DSPy:" not in str(exc)
    lm = dspy.LM("openai/gpt-5.6-luna", api_key="k", max_tokens=16000)
    _hint_responses_api(lm, exc)
    _hint_responses_api(lm, exc)
    assert str(exc).count("DSPy:") == 1


# ─── 13. the Flex shim is package data read on first use ──────────────


def test_flex_shim_is_read_lazily_as_package_data():
    import importlib

    from dspy.predict.flex import bridge

    assert not hasattr(bridge, "SHIM_SETUP")  # nothing read at import
    source = bridge._shim_source()
    assert "_DspyPending" in source and bridge._shim_source() is source  # cached
    hook_dirs = importlib.import_module("dspy.__pyinstaller").get_hook_dirs()
    assert os.path.isfile(os.path.join(hook_dirs[0], "hook-dspy.py"))


