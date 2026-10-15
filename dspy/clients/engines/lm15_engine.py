"""Native lm15 routing, with canonical errors and no retry policy."""

import math
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace

from dspy._vendor.lm15.registry import canonical_provider
from dspy._vendor.lm15.router import LITELLM_PROVIDER_PREFIXES
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
    Timeouts,
    UnsupportedFeatureError,
)


def timeouts_for(timeout) -> Timeouts | None:
    """lm15 ``Timeouts`` from an LM ``timeout`` setting, or None for lm15's own defaults.

    A number of seconds bounds every wait the way LiteLLM's ``timeout`` did:
    the next byte (read), the send (write), and a free connection (pool).
    Connecting keeps lm15's short default. An ``httpx.Timeout`` maps each of
    its components. In httpx a component set to ``None`` means "wait
    forever"; lm15 bounds every wait and has no "forever", so such a value
    is refused as unsupported rather than silently replaced by a default.
    """
    if timeout is None:
        return None
    names = ("connect", "read", "write", "pool")
    if any(hasattr(timeout, name) for name in names):
        parts = {name: getattr(timeout, name, None) for name in names}
        disabled = sorted(name for name, value in parts.items() if value is None)
        if disabled:
            raise UnsupportedFeatureError(
                f"timeout disables {', '.join(disabled)} (None means wait forever); the native engine "
                "bounds every wait, so pass a number of seconds for each component or use engine='litellm'",
                feature="timeout",
            )
    else:
        parts = {"connect": None, "read": timeout, "write": timeout, "pool": timeout}
    kwargs = {}
    for name, value in parts.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"timeout must be a number of seconds or an httpx.Timeout, not {type(timeout).__name__}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")
        kwargs[name] = float(value)
    return Timeouts(**kwargs) if kwargs else None


def _model_string(model: str, model_type: str, providers=()) -> str:
    """Keep endpoint selection deliberate, including model ids containing ':' .

    A ``provider/model`` prefix is a LiteLLM spelling lm15 has a door for, or
    a provider declared on this engine's config, by id or alias in either
    spelling.
    """
    head, separator, rest = model.partition("/")
    if separator and rest:
        if head in LITELLM_PROVIDER_PREFIXES:
            provider = LITELLM_PROVIDER_PREFIXES[head]
            if model_type == "responses":
                provider = {"openai-chat": "openai", "azure-chat": "azure"}.get(provider, provider)
            return f"{provider}:{rest}"
        spelling = canonical_provider(head)
        for definition in providers:
            if spelling in definition.spellings:
                return f"{definition.id}:{rest}"
    return model


class _Routing:
    def _init_routing(self, config, model_type):
        if model_type not in {"chat", "responses"}:
            raise UnsupportedFeatureError("lm15 engines support chat and Responses APIs, not text completions.")
        self.model_type = model_type
        self.config = config if config is not None else RouterConfig()
        self._closed = False
        self._lock = threading.RLock()
        self._providers = {}

    def resolve(self, model: str):
        """Offline routing. Callers inspect canonical errors, not a wrapped cause."""
        routed = _model_string(model, self.model_type, self.config.providers)
        if self.model_type == "chat":
            return self.router.resolve_openai_chat(routed)
        resolution = self.router.resolve(routed)
        from dspy._vendor.lm15.registry import lookup

        declared = {d.id: d for d in self.config.providers}
        definition = declared.get(resolution.provider) or lookup(resolution.provider)
        if definition is None or definition.dialect != "openai-responses":
            raise UnsupportedFeatureError(f"{model!r} does not select a Responses API endpoint.")
        return resolution

    def _target(self, request, *, streaming=False):
        validate_request(request)
        resolution = self.resolve(request.model)
        with self._lock:
            if self._closed:
                raise ConfigurationError("Engine is closed")
            # Serialize lazy construction so parallel first calls do not race
            # the router's one shared transport. Borrowed transports stay caller-owned.
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

    def plan(self, request: Request):
        """What this request WOULD adapt on its route (lm15 MAP-13), with no network.

        Offline like ``resolve``: no credential is read or invoked, so a
        model with no key configured still plans. Raises the refusal the
        call would raise, so engine selection can fall back before any I/O
        instead of after a failed attempt.
        """
        validate_request(request)
        resolution = self.resolve(request.model)
        with self._lock:
            if self._closed:
                raise ConfigurationError("Engine is closed")
            # Under the engine's lock like _target: the router is thread-safe
            # itself (lm15 rc2), and this keeps "closed" and "planning"
            # from interleaving.
            return self.router.plan(replace(request, model=f"{resolution.provider}:{resolution.model}"))


class LM15Engine(_Routing):
    """One synchronous native attempt. Close after its active calls finish."""

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat"):
        self._init_routing(config, model_type)
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
            self._providers.clear()
        # The router owns the one transport its providers share (lm15 rc2);
        # a caller-supplied transport stays the caller's to close.
        if self.config.transport is None:
            self.router.close()
        else:
            self.router._lms.clear()


class AsyncLM15Engine(_Routing):
    """One async attempt; owned pools are confined to their event loop."""

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat"):
        self._init_routing(config, model_type)
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
            self._providers.clear()
        if self.config.transport is None:
            await self.router.aclose()
        else:
            self.router._lms.clear()
