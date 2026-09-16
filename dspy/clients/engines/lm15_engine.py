"""Native lm15 routing, with canonical errors and no retry policy."""

import math
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, ExitStack
from dataclasses import replace

from dspy._vendor.lm15.router import LITELLM_PROVIDER_PREFIXES
from dspy._vendor.lm15.transports import StdlibAsyncTransport, StdlibTransport
from dspy._vendor.lm15.types import StreamEvent
from dspy.clients.engines.base import validate_request
from dspy.clients.engines.lifecycle import aclosing_stream, closing_stream
from dspy.clients.engines.stream_guard import achecked_stream, checked_stream
from dspy.lm15 import (
    AsyncLMRouter,
    ConfigurationError,
    LMRouter,
    Request,
    Response,
    RouterConfig,
    UnsupportedFeatureError,
)


def _model_string(model: str, model_type: str) -> str:
    """Keep endpoint selection deliberate, including model ids containing ':' ."""
    head, separator, rest = model.partition("/")
    if separator and head in LITELLM_PROVIDER_PREFIXES:
        provider = LITELLM_PROVIDER_PREFIXES[head]
        if model_type == "responses":
            provider = {"openai-chat": "openai", "azure-chat": "azure"}.get(provider, provider)
        return f"{provider}:{rest}"
    return model


# No timeout setting. Distinct from None, which asks the transport for an
# unbounded wait.
UNSET = object()


def read_timeout_seconds(timeout):
    """Seconds to wait for each read, from an LM ``timeout`` setting.

    Accepts a number of seconds or an ``httpx.Timeout``, whose ``read``
    component is the equivalent bound. No setting returns ``UNSET``; an
    ``httpx.Timeout`` with ``read=None`` returns ``None``, an unbounded wait.
    """
    if timeout is None:
        return UNSET
    read = getattr(timeout, "read", timeout)
    if read is None:
        return None
    if isinstance(read, bool) or not isinstance(read, (int, float)):
        raise TypeError(f"timeout must be a number of seconds or an httpx.Timeout, not {type(timeout).__name__}")
    if not math.isfinite(read) or read <= 0:
        raise ValueError("timeout must be a positive finite number of seconds")
    return float(read)


class _Routing:
    def _init_routing(self, config, model_type, read_timeout, transport_cls):
        if model_type not in {"chat", "responses"}:
            raise UnsupportedFeatureError("lm15 engines support chat and Responses APIs, not text completions.")
        self.model_type = model_type
        self.config = config if config is not None else RouterConfig()
        # A caller-supplied transport keeps its own timeouts. Otherwise the
        # engine owns one transport, shared by every provider it builds.
        self._owned_transport = None
        if read_timeout is not UNSET and self.config.transport is None:
            self._owned_transport = transport_cls(read_timeout=read_timeout)
            self.config = replace(self.config, transport=self._owned_transport)
        self._closed = False
        self._lock = threading.RLock()
        self._providers = {}

    def resolve(self, model: str):
        """Offline routing. Callers inspect canonical errors, not a wrapped cause."""
        routed = _model_string(model, self.model_type)
        if self.model_type == "chat":
            return self.router.resolve_openai_chat(routed)
        resolution = self.router.resolve(routed)
        from dspy._vendor.lm15.registry import lookup

        definition = lookup(resolution.provider)
        if definition is None or definition.dialect != "openai-responses":
            raise UnsupportedFeatureError(f"{model!r} does not select a Responses API endpoint.")
        return resolution

    def _target(self, request, *, streaming=False):
        validate_request(request)
        resolution = self.resolve(request.model)
        with self._lock:
            if self._closed:
                raise ConfigurationError("Engine is closed")
            # Serialize lazy construction so parallel first calls do not leak
            # duplicate owned transports. Borrowed transports stay caller-owned.
            provider = self._providers.get(resolution.provider)
            if provider is None:
                provider = self.router.lm(f"{resolution.provider}:{resolution.model}")
                self._providers[resolution.provider] = provider
        surface = "stream" if streaming else "complete"
        if not getattr(provider.supports, surface):
            raise UnsupportedFeatureError(
                f"{resolution.provider} does not support {surface}.", provider=resolution.provider,
            )
        return provider, replace(request, model=resolution.model)


class LM15Engine(_Routing):
    """One synchronous native attempt. Close after its active calls finish."""

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat", read_timeout=UNSET):
        self._init_routing(config, model_type, read_timeout, StdlibTransport)
        self.router = LMRouter(self.config)

    def complete(self, request: Request) -> Response:
        provider, routed = self._target(request)
        return provider.complete(routed)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        provider, routed = self._target(request, streaming=True)
        events = checked_stream(provider.stream(routed), provider=provider.provider)
        with closing_stream(events):
            yield from events

    def close(self):
        with self._lock:
            self._closed = True
            providers = list(self._providers.values())
            self._providers.clear()
        self.router._lms.clear()
        if self._owned_transport is not None:
            self._owned_transport.close()
        elif self.config.transport is None:
            with ExitStack() as cleanup:
                for provider in providers:
                    cleanup.callback(provider.close)


class AsyncLM15Engine(_Routing):
    """One async attempt; owned pools are confined to their event loop."""

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat", read_timeout=UNSET):
        self._init_routing(config, model_type, read_timeout, StdlibAsyncTransport)
        self.router = AsyncLMRouter(self.config)

    async def complete(self, request: Request) -> Response:
        import asyncio

        # Construction may load or refresh credentials synchronously.
        provider, routed = await asyncio.to_thread(self._target, request)
        return await provider.complete(routed)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        import asyncio

        provider, routed = await asyncio.to_thread(self._target, request, streaming=True)
        events = achecked_stream(provider.stream(routed), provider=provider.provider)
        async with aclosing_stream(events):
            async for event in events:
                yield event

    async def aclose(self):
        with self._lock:
            self._closed = True
            providers = list(self._providers.values())
            self._providers.clear()
        self.router._lms.clear()
        if self._owned_transport is not None:
            await self._owned_transport.aclose()
        elif self.config.transport is None:
            async with AsyncExitStack() as cleanup:
                for provider in providers:
                    cleanup.push_async_callback(provider.aclose)
