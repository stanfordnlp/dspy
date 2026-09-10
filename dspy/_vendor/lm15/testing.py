"""
lm15.testing — shipped test doubles for code built on lm15.

Two seams, pick by what you are testing:

- :class:`FakeLM` — a canonical-level provider double.  Script it with
  ``Response`` objects (or plain strings, or exceptions) and it answers
  ``complete()`` / ``stream()`` with them, streaming via
  ``response_to_events``.  No wire format involved, works wherever a
  provider LM does — the right seam for testing tool loops, retries,
  and everything above the wire.
- :class:`FakeTransport` / :class:`FakeResponse` — a wire-level double.
  Inject it as any adapter's ``transport`` and the REAL adapter serde
  runs both directions against your scripted HTTP bytes; the fake
  records every ``TransportRequest`` so one fixture asserts behavior
  AND wire format.  The right seam for testing provider dialects.

Both record the requests they serve.  Neither performs any I/O.

    from lm15.testing import FakeLM
    lm = FakeLM(responses=["hello"])
    assert lm.complete(request).text == "hello"

The router accepts a transport too — ``RouterConfig(transport=...)`` —
so routed code paths are testable without touching the LM cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterator, Sequence

from .features import EndpointSupport, ProviderManifest
from .result import response_to_events
from .types import (
    Message,
    Request,
    Response,
    StreamEvent,
    TextPart,
    Usage,
)

__all__ = ["FakeLM", "FakeTransport", "FakeResponse"]


@dataclass
class FakeResponse:
    """A scripted wire response served by :class:`FakeTransport`.

    ``body`` is the exact bytes the adapter will parse; ``chunks`` (when
    given) is what streaming iteration yields instead — SSE tests script
    the frames there.
    """

    status: int
    body: bytes
    headers: list[tuple[str, str]] | None = None
    reason: str = "OK"
    http_version: str = "HTTP/1.1"
    chunks: list[bytes] | None = None

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = [("content-type", "application/json")]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield from (self.chunks if self.chunks is not None else [self.body])

    def read(self) -> bytes:
        return b"".join(iter(self))

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for key, value in self.headers or []:
            if key.lower() == lname:
                return value
        return None


class FakeTransport:
    """Wire-level double: serves scripted :class:`FakeResponse` objects
    in order and records every request in ``.requests``.

    Script an ``Exception`` instance instead of a response and it is
    raised at request time (dropped-socket tests: script
    ``lm15.transports.TransportError``).
    """

    def __init__(self, responses: Sequence[FakeResponse | Exception] | None = None) -> None:
        self.responses: list[FakeResponse | Exception] = list(responses or [])
        self.requests: list[Any] = []

    def stream(self, request: Any) -> FakeResponse:
        self.requests.append(request)
        assert self.responses, (
            "FakeTransport ran out of scripted responses "
            f"(served {len(self.requests) - 1}, then got another request)"
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _as_response(item: "Response | str") -> Response:
    if isinstance(item, Response):
        return item
    return Response(
        id="fake",
        model="fake",
        message=Message.assistant(TextPart(str(item))),
        finish_reason="stop",
        usage=Usage(),
    )


class FakeLM:
    """Canonical-level provider double.

    Scripted with ``Response`` objects, plain strings (shorthand for a
    text response), or ``Exception`` instances (raised in order).
    ``stream()`` replays the same scripted response through
    ``response_to_events``, so streaming code paths see exactly the
    events a real adapter would emit for that response.
    """

    provider: str = "fake"
    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="fake", supports=supports, auth_modes=(), env_keys=(),
    )

    def __init__(
        self,
        responses: Sequence["Response | str | Exception"] = (),
        *,
        provider: str = "fake",
    ) -> None:
        self.provider = provider
        self._script: list[Any] = list(responses)
        self.requests: list[Request] = []

    def _next(self, request: Request) -> Response:
        self.requests.append(request)
        assert self._script, (
            "FakeLM ran out of scripted responses "
            f"(served {len(self.requests) - 1}, then got another request)"
        )
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _as_response(item)

    def complete(self, request: Request) -> Response:
        return self._next(request)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        return response_to_events(self._next(request))

    def close(self) -> None:  # symmetric with real adapters
        return None
