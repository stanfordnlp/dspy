"""
lm15.sse — Server-Sent Events parser.

Parses a byte-line iterator (from a streaming HTTP response) into
typed SSEEvent objects.

No size limit by default (lm15-contract INV-056). A provider sends whole
objects as single lines: OpenAI Responses repeats the full response,
system prompt included, in ``response.completed``; Gemini sends a 4K image
as one 29.7 MB line. A non-streamed reply has no limit either, and a stream
is accumulated into the whole reply anyway, so a per-line cap refused real
answers without bounding memory. Callers who want a cap on this parser set
``max_line_bytes`` / ``max_event_bytes``; going over one raises
``TransportError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from .errors import TransportError


@dataclass(frozen=True, slots=True)
class SSEEvent:
    event: str | None
    data: str


def parse_sse(
    lines: Iterator[bytes],
    *,
    max_line_bytes: int | None = None,
    max_event_bytes: int | None = None,
) -> Iterator[SSEEvent]:
    """Parse SSE byte lines into events.

    ``max_line_bytes`` / ``max_event_bytes`` are opt-in caps (``None``: no
    cap, INV-056); a line or event over a cap raises ``TransportError``.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0

    for raw in lines:
        if max_line_bytes is not None and len(raw) > max_line_bytes:
            raise TransportError(f"SSE line exceeds limit ({len(raw)} > {max_line_bytes})")

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        event_bytes += len(raw)
        if max_event_bytes is not None and event_bytes > max_event_bytes:
            raise TransportError(f"SSE event exceeds limit ({event_bytes} > {max_event_bytes})")

        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
            event_bytes = 0
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
            continue

    if data_lines:
        yield SSEEvent(event=event_name, data="\n".join(data_lines))


async def aparse_sse(
    lines: AsyncIterator[bytes],
    *,
    max_line_bytes: int | None = None,
    max_event_bytes: int | None = None,
) -> AsyncIterator[SSEEvent]:
    """Async mirror of :func:`parse_sse` over an async byte-line iterator.

    Same field grammar and opt-in caps; the only difference is ``async for`` over
    the line source.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0

    async for raw in lines:
        if max_line_bytes is not None and len(raw) > max_line_bytes:
            raise TransportError(f"SSE line exceeds limit ({len(raw)} > {max_line_bytes})")

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        event_bytes += len(raw)
        if max_event_bytes is not None and event_bytes > max_event_bytes:
            raise TransportError(f"SSE event exceeds limit ({event_bytes} > {max_event_bytes})")

        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
            event_bytes = 0
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
            continue

    if data_lines:
        yield SSEEvent(event=event_name, data="\n".join(data_lines))
