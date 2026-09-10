"""
Transport-level request/response models.

These are intentionally minimal — they're the bytes-in/bytes-out interface
between the LM layer (which speaks the lm15 type system) and the
HTTP transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator


@dataclass(slots=True)
class TransportRequest:
    method: str
    url: str = field(repr=False)
    headers: list[tuple[str, str]] = field(default_factory=list, repr=False)
    body: bytes = field(default=b"", repr=False)
    # Per-request timeout overrides (None = use transport default)
    connect_timeout: float | None = None
    read_timeout: float | None = None
    write_timeout: float | None = None


class TransportResponse:
    """Sync streaming response.

    Iterating yields body chunks as bytes.  Must be used as a context manager
    so the connection is properly returned to the pool (or closed) even on
    early exit.
    """

    status: int
    reason: str
    headers: list[tuple[str, str]]
    http_version: str

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        http_version: str,
        chunks: Iterator[bytes],
        release: "callable",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.http_version = http_version
        self._chunks = chunks
        self._release = release
        self._released = False
        self._complete = False

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [v for k, v in self.headers if k.lower() == lname]

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._chunks:
                if chunk:
                    yield chunk
            self._complete = True
        finally:
            self._release_once(body_consumed=self._complete)

    def read(self) -> bytes:
        return b"".join(self)

    def iter_lines(self) -> Iterator[bytes]:
        """Yield newline-terminated byte lines from arbitrary body chunks."""
        buf = bytearray()
        for chunk in self:
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                idx = buf.find(b"\n")
                if idx < 0:
                    break
                line = bytes(buf[: idx + 1])
                del buf[: idx + 1]
                yield line
        if buf:
            yield bytes(buf)

    def close(self) -> None:
        self._release_once(body_consumed=self._complete)

    def _release_once(self, *, body_consumed: bool) -> None:
        if self._released:
            return
        self._released = True
        try:
            self._release(body_consumed)
        except Exception:
            pass

    def __enter__(self) -> "TransportResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AsyncTransportResponse:
    """Async streaming response.  Async-iterate to get body chunks."""

    status: int
    reason: str
    headers: list[tuple[str, str]]
    http_version: str

    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: list[tuple[str, str]],
        http_version: str,
        chunks: AsyncIterator[bytes],
        release: "callable",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.http_version = http_version
        self._chunks = chunks
        self._release = release  # async callable(body_consumed: bool)
        self._released = False
        self._complete = False

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [v for k, v in self.headers if k.lower() == lname]

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._chunks:
                if chunk:
                    yield chunk
            self._complete = True
        finally:
            await self._release_once(body_consumed=self._complete)

    async def read(self) -> bytes:
        buf = bytearray()
        async for c in self:
            buf.extend(c)
        return bytes(buf)

    async def aiter_lines(self) -> AsyncIterator[bytes]:
        """Yield newline-terminated byte lines from arbitrary body chunks."""
        buf = bytearray()
        async for chunk in self:
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                idx = buf.find(b"\n")
                if idx < 0:
                    break
                line = bytes(buf[: idx + 1])
                del buf[: idx + 1]
                yield line
        if buf:
            yield bytes(buf)

    async def aclose(self) -> None:
        await self._release_once(body_consumed=self._complete)

    async def _release_once(self, *, body_consumed: bool) -> None:
        if self._released:
            return
        self._released = True
        try:
            await self._release(body_consumed)
        except Exception:
            pass

    async def __aenter__(self) -> "AsyncTransportResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
