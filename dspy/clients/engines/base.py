"""One-attempt execution interfaces for DSPy's LM layer.

Engines consume one canonical request and produce one response. They do not
own DSPy's cache, retries, candidate fan-out, history or callbacks. A stream
emits one leading start event and one final end event only on successful completion.
Expected failures raise dspy.lm15 errors. Canonical StreamErrorEvents are also
accepted as terminal failures by DSPy's shared stream guard, never as Responses.
SDK-specific errors are translated by the owning engine; engines do not import
DSPy's public error hierarchy or implement a second retry loop.

DSPy 3.5 uses this contract throughout adapters and engines. Legacy forward()
plugins, LegacyEngine/AsyncLegacyEngine, and complete_legacy() shortcuts are
3.4 transition interfaces, deprecated for removal in 3.5.
"""

import inspect
from typing import Protocol

from dspy.lm15 import Request, Response


class Engine(Protocol):
    """Required sync interface; streaming and cleanup are optional extensions."""

    def complete(self, request: Request) -> Response: ...


class AsyncEngine(Protocol):
    """Required async interface; streaming and cleanup are optional extensions."""

    async def complete(self, request: Request) -> Response: ...


def _is_coroutine_function(fn) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    # A callable object whose __call__ (own or inherited) is async: resolve
    # it through the MRO as Python does. A plain function's type resolves
    # __call__ to a slot wrapper, which is not a coroutine function.
    call = next((vars(cls)["__call__"] for cls in type(fn).__mro__ if "__call__" in vars(cls)), None)
    return inspect.iscoroutinefunction(call)


def validate_engines(engine: Engine | None, async_engine: AsyncEngine | None) -> None:
    """Validate supplied engines without requiring inheritance or optional methods."""
    if engine is not None:
        complete = getattr(engine, "complete", None)
        if not callable(complete):
            raise TypeError("A custom engine must implement complete(Request) -> Response")
        if _is_coroutine_function(complete):
            raise TypeError(
                "engine.complete is a coroutine function; the sync engine returns a Response directly. "
                "Pass an async engine as async_engine="
            )
    if async_engine is not None:
        acomplete = getattr(async_engine, "complete", None)
        if not callable(acomplete):
            raise TypeError("A custom async engine must implement async complete(Request) -> Response")
        if not _is_coroutine_function(acomplete):
            raise TypeError(
                "async_engine.complete is not a coroutine function; the async engine is awaited "
                "(async def complete). Pass a sync engine as engine="
            )


def validate_request(request: Request) -> None:
    if not isinstance(request, Request):
        raise TypeError("An engine requires a dspy.lm15.Request")
    # Execution controls may not bypass the single-response contract through
    # provider passthrough fields. They are owned by the outer DSPy LM layer.
    forbidden = {
        "n", "num_generations", "stream", "stream_options", "num_retries", "max_retries",
        "retry_strategy", "cache", "caching", "rollout_id", "model", "messages", "input",
        "api_key", "api_base", "base_url", "headers", "extra_headers", "timeout",
    }
    extra = forbidden.intersection(request.config.extensions or {})
    if extra:
        raise ValueError(f"Engine execution controls do not belong in Config.extensions: {sorted(extra)}")
