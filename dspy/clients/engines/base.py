"""One-attempt execution interfaces for DSPy's LM layer.

Engines consume one canonical request and produce one response. They do not
own DSPy's cache, retries, candidate fan-out, history or callbacks. A stream
raises on failure and emits one final end event only on successful completion.

DSPy 3.5 uses this contract throughout adapters and engines. Legacy forward()
plugins, LegacyEngine/AsyncLegacyEngine, and complete_legacy() shortcuts are
3.4 transition interfaces, deprecated for removal in 3.5.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from dspy._vendor.lm15.types import StreamEvent
from dspy.lm15 import Request, Response


class Engine(Protocol):
    def complete(self, request: Request) -> Response: ...

    def stream(self, request: Request) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


class AsyncEngine(Protocol):
    async def complete(self, request: Request) -> Response: ...

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...


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
