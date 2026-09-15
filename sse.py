"""
lm15.sse — Server-Sent Events parser.

Parses a byte-line iterator (from a streaming HTTP response) into
typed SSEEvent objects.
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
    max_line_bytes: int = 64 * 1024,
    max_event_bytes: int = 1024 * 1024,
) -> Iterator[SSEEvent]:
    """Parse SSE byte lines into events."""
    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0

    for raw in lines:
        if len(raw) > max_line_bytes:
            raise TransportError(f"SSE line exceeds limit ({len(raw)} > {max_line_bytes})")

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        event_bytes += len(raw)
        if event_bytes > max_event_bytes:
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
    max_line_bytes: int = 64 * 1024,
    max_event_bytes: int = 1024 * 1024,
) -> AsyncIterator[SSEEvent]:
    """Async mirror of :func:`parse_sse` over an async byte-line iterator.

    Same field grammar and limits; the only difference is ``async for`` over
    the line source.
    """
    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0

    async for raw in lines:
        if len(raw) > max_line_bytes:
            raise TransportError(f"SSE line exceeds limit ({len(raw)} > {max_line_bytes})")

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        event_bytes += len(raw)
        if event_bytes > max_event_bytes:
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
