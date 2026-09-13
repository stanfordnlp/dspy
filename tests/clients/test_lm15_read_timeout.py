"""Native completions wait for slow first bytes and honor a caller's timeout."""

import asyncio
import http.server
import json
import threading

import pytest

import dspy
from dspy._vendor.lm15.router import LMRouter
from dspy._vendor.lm15.testing import FakeTransport
from dspy._vendor.lm15.transports import StdlibTransport
from dspy._vendor.lm15.transports._async import StdlibAsyncTransport
from dspy.clients.backend_selection import select_backend
from dspy.clients.engines import AsyncLM15Engine, LiteLLMEngine, LM15Engine
from dspy.clients.engines.lm15_engine import UNSET, read_timeout_seconds
from dspy.clients.execution import _engine, prepare
from dspy.lm15 import Message, Request, RouterConfig

DIALECTS = ["openai-chat", "openai", "anthropic", "gemini"]
MODELS = {"openai-chat": "gpt-5", "openai": "gpt-5", "anthropic": "claude-opus-5", "gemini": "gemini-2.5-pro"}


def provider_for(dialect, transport):
    config = RouterConfig(env={}, api_keys={dialect: "fake"}, transport=transport)
    return LMRouter(config).lm(f"{dialect}:{MODELS[dialect]}")


def request_for(dialect):
    return Request(model=MODELS[dialect], messages=(Message.user("hello"),))


@pytest.mark.parametrize("dialect", DIALECTS)
def test_completion_requests_defer_to_the_transport_read_timeout(dialect):
    provider = provider_for(dialect, FakeTransport())
    wire = provider.build_request(request_for(dialect), stream=False)
    assert wire.read_timeout is None


@pytest.mark.parametrize("dialect", DIALECTS)
def test_streaming_requests_defer_to_the_transport_read_timeout(dialect):
    provider = provider_for(dialect, FakeTransport())
    wire = provider.build_request(request_for(dialect), stream=True)
    assert wire.read_timeout is None


@pytest.mark.parametrize("transport_cls", [StdlibTransport, StdlibAsyncTransport])
def test_transport_default_read_timeout_matches_litellm(transport_cls):
    assert transport_cls()._read_timeout == 600.0


class _ChatServer(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"id": "r", "model": "gpt-5", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _RecordingTransport(StdlibTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.header_waits = []

    def _read_head(self, conn, *, read_timeout):
        self.header_waits.append(read_timeout)
        return super()._read_head(conn, read_timeout=read_timeout)


@pytest.fixture
def chat_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _ChatServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("read_timeout", [900, None])
def test_router_transport_read_timeout_governs_completion_header_wait(chat_server, read_timeout):
    transport = _RecordingTransport(read_timeout=read_timeout)
    config = RouterConfig(env={}, api_keys={"openai-chat": "fake"}, base_urls={"openai-chat": chat_server},
                          transport=transport)
    provider = LMRouter(config).lm("openai-chat:gpt-5")
    try:
        response = provider.complete(request_for("openai-chat"))
    finally:
        transport.close()
    assert response.message.text == "hi"
    assert transport.header_waits == [read_timeout]


@pytest.fixture
def no_gateway_env(monkeypatch):
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("timeout", [600, 600.0])
def test_lm_timeout_stays_native_and_configures_the_transport(timeout, no_gateway_env):
    lm = dspy.LM("openai/gpt-5", api_key="fake", timeout=timeout)
    assert select_backend(lm).native is True
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    try:
        assert isinstance(backend, LM15Engine)
        assert backend.config.transport._read_timeout == 600.0
    finally:
        backend.close()


def test_lm_httpx_timeout_uses_its_read_component(no_gateway_env):
    httpx = pytest.importorskip("httpx")
    lm = dspy.LM("openai/gpt-5", api_key="fake", timeout=httpx.Timeout(10, read=450))
    assert select_backend(lm).native is True
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    try:
        assert backend.config.transport._read_timeout == 450.0
    finally:
        backend.close()


def test_lm_httpx_timeout_without_read_bound_waits_without_limit(no_gateway_env):
    httpx = pytest.importorskip("httpx")
    lm = dspy.LM("openai/gpt-5", api_key="fake", timeout=httpx.Timeout(None))
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    try:
        assert backend.config.transport is not None
        assert backend.config.transport._read_timeout is None
    finally:
        backend.close()


def test_equal_httpx_timeouts_share_one_engine(no_gateway_env):
    httpx = pytest.importorskip("httpx")
    lm = dspy.LM("openai/gpt-5", api_key="fake")
    backends = []
    for _ in range(2):
        call = prepare(lm, "hello", None, {"timeout": httpx.Timeout(10, read=450)})
        backends.append(_engine(lm, call, False)[0])
    try:
        assert backends[0] is backends[1]
        assert len(lm._engine_store) == 1
    finally:
        backends[0].close()


def test_lm_without_timeout_keeps_the_router_default_transport(no_gateway_env):
    lm = dspy.LM("openai/gpt-5", api_key="fake")
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    try:
        assert backend.config.transport is None
    finally:
        backend.close()


def test_lm_timeout_on_litellm_engine_is_unchanged(no_gateway_env):
    lm = dspy.LM("openai/gpt-5", engine="litellm", timeout=600)
    assert select_backend(lm).native is False
    backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), False)
    assert isinstance(backend, LiteLLMEngine)
    assert backend.client_options["timeout"] == 600


def test_async_lm_timeout_configures_an_async_transport(no_gateway_env):
    async def build():
        lm = dspy.LM("openai/gpt-5", api_key="fake", timeout=600)
        backend, _, _ = _engine(lm, prepare(lm, "hello", None, {}), True)
        try:
            assert isinstance(backend, AsyncLM15Engine)
            assert isinstance(backend.config.transport, StdlibAsyncTransport)
            assert backend.config.transport._read_timeout == 600.0
        finally:
            await backend.aclose()

    asyncio.run(build())


@pytest.mark.parametrize("timeout", [0, -5, "600", True, float("nan"), float("inf")])
def test_read_timeout_seconds_rejects_unusable_values(timeout):
    with pytest.raises((TypeError, ValueError)):
        read_timeout_seconds(timeout)


def test_read_timeout_seconds_keeps_the_default_for_none():
    assert read_timeout_seconds(None) is UNSET
