"""Native lm15 routing, with no retries or cross-backend fallback."""

import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, ExitStack
from dataclasses import replace

from dspy._vendor.lm15.router import LITELLM_PROVIDER_PREFIXES
from dspy._vendor.lm15.types import StreamEvent
from dspy.clients.engines.base import validate_request
from dspy.clients.engines.errors import wrap_error
from dspy.lm15 import AsyncLMRouter, LM15Error, LMRouter, Request, Response, RouterConfig
from dspy.utils.exceptions import LMUnsupportedFeatureError


def _model_string(model: str, model_type: str) -> str:
    """Keep endpoint selection deliberate, including model ids containing ':' ."""
    head, separator, rest = model.partition("/")
    if separator and head in LITELLM_PROVIDER_PREFIXES:
        provider = LITELLM_PROVIDER_PREFIXES[head]
        if model_type == "responses":
            provider = {"openai-chat": "openai", "azure-chat": "azure"}.get(provider, provider)
        return f"{provider}:{rest}"
    return model


class _Routing:
    def _init_routing(self, config, model_type):
        if model_type not in {"chat", "responses"}:
            raise LMUnsupportedFeatureError("lm15 engines support chat and Responses APIs, not text completions.")
        self.model_type = model_type
        self.config = config if config is not None else RouterConfig()
        self._closed = False
        self._lock = threading.RLock()
        self._providers = {}

    def resolve(self, model: str):
        """Offline routing only. A missing route does not initiate a fallback."""
        try:
            routed = _model_string(model, self.model_type)
            if self.model_type == "chat":
                # Already normalized slash prefixes must not be read again;
                # bare OpenAI names retain the legacy Chat endpoint.
                resolution = self.router.resolve_openai_chat(routed)
            else:
                resolution = self.router.resolve(routed)
                from dspy._vendor.lm15.registry import lookup

                definition = lookup(resolution.provider)
                if definition is None or definition.dialect != "openai-responses":
                    raise LMUnsupportedFeatureError(
                        f"{model!r} does not select a Responses API endpoint.", model=model,
                    )
            return resolution
        except LM15Error as exc:
            raise wrap_error(exc, model=model) from exc

    def _target(self, request, *, streaming=False):
        validate_request(request)
        resolution = self.resolve(request.model)
        with self._lock:
            if self._closed:
                raise RuntimeError("Engine is closed")
            # Router construction is lazy. Serialize that step so parallel
            # first calls do not construct and leak duplicate transports.
            provider = self._providers.get(resolution.provider)
            if provider is None:
                provider = self.router.lm(f"{resolution.provider}:{resolution.model}")
                self._providers[resolution.provider] = provider
        surface = "stream" if streaming else "complete"
        if not getattr(provider.supports, surface):
            raise LMUnsupportedFeatureError(
                f"{resolution.provider} does not support {surface}.", model=request.model,
                provider=resolution.provider, features=[surface],
            )
        return provider, replace(request, model=resolution.model)


class LM15Engine(_Routing):
    """Synchronous native engine. Close only after its active calls finish.

    A caller-supplied transport is borrowed and is never closed by the engine.
    Default provider transports are owned and closed here.
    """

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat"):
        self._init_routing(config, model_type)
        self.router = LMRouter(self.config)

    def complete(self, request: Request) -> Response:
        validate_request(request)
        try:
            provider, routed = self._target(request)
            return provider.complete(routed)
        except LM15Error as exc:
            raise wrap_error(exc, model=request.model) from exc

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        validate_request(request)
        source = None
        try:
            provider, routed = self._target(request, streaming=True)
            source = provider.stream(routed)
            ended = False
            for event in source:
                if event.type == "error":
                    from dspy._vendor.lm15.errors import error_class_for_code

                    raise error_class_for_code(event.error.code)(
                        event.error.message, provider=provider.provider, provider_code=event.error.provider_code,
                    )
                if event.type == "end":
                    ended = True
                yield event
            if not ended:
                from dspy.lm15 import StreamAssemblyError

                raise StreamAssemblyError("Provider stream ended without a completion event")
        except LM15Error as exc:
            raise wrap_error(exc, model=request.model) from exc
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    def close(self):
        with self._lock:
            self._closed = True
            providers = list(self._providers.values())
            self._providers.clear()
        self.router._lms.clear()
        if self.config.transport is None:
            with ExitStack() as cleanup:
                for provider in providers:
                    cleanup.callback(provider.close)


class AsyncLM15Engine(_Routing):
    """Async native engine; use within one event loop and close after calls finish."""

    def __init__(self, config: RouterConfig | None = None, *, model_type="chat"):
        self._init_routing(config, model_type)
        self.router = AsyncLMRouter(self.config)

    async def complete(self, request: Request) -> Response:
        validate_request(request)
        try:
            # Provider construction can load/refresh subscription credentials.
            # Do not block the event loop on that synchronous work.
            import asyncio

            provider, routed = await asyncio.to_thread(self._target, request)
            return await provider.complete(routed)
        except LM15Error as exc:
            raise wrap_error(exc, model=request.model) from exc

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        import asyncio

        validate_request(request)
        source = None
        try:
            provider, routed = await asyncio.to_thread(self._target, request, streaming=True)
            source = provider.stream(routed)
            ended = False
            async for event in source:
                if event.type == "error":
                    from dspy._vendor.lm15.errors import error_class_for_code

                    raise error_class_for_code(event.error.code)(
                        event.error.message, provider=provider.provider, provider_code=event.error.provider_code,
                    )
                if event.type == "end":
                    ended = True
                yield event
            if not ended:
                from dspy.lm15 import StreamAssemblyError

                raise StreamAssemblyError("Provider stream ended without a completion event")
        except LM15Error as exc:
            raise wrap_error(exc, model=request.model) from exc
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()

    async def aclose(self):
        with self._lock:
            self._closed = True
            providers = list(self._providers.values())
            self._providers.clear()
        self.router._lms.clear()
        if self.config.transport is None:
            async with AsyncExitStack() as cleanup:
                for provider in providers:
                    cleanup.push_async_callback(provider.aclose)
