"""
Transport-level request/response models.

These are intentionally minimal — they're the bytes-in/bytes-out interface
between the LM layer (which speaks the lm15 type system) and the
HTTP transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator


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
    _admit: Callable[[], None] | None = field(default=None, repr=False, compare=False)


class LineSplitter:
    """Split body chunks into newline-terminated lines in linear time.

    A provider can send one line of tens of megabytes (Gemini streams a 4K
    image as a single 29.7 MB SSE line, lm15-contract INV-056) in thousands
    of reads.  Searching the whole buffer for ``\\n`` after every read is
    quadratic in that line's length; this searches only the bytes that have
    not been searched yet, and copies a line once, when it is complete.
    Lines keep their ``\\n``; a last line without one is yielded at the end.
    """

    __slots__ = ("_buf", "_searched")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._searched = 0

    def feed(self, chunk: bytes) -> list[bytes]:
        """Take one chunk; return the lines it completed, in order."""
        buf = self._buf
        buf += chunk
        lines: list[bytes] = []
        start = 0
        idx = buf.find(b"\n", self._searched)
        while idx >= 0:
            lines.append(bytes(buf[start : idx + 1]))
            start = idx + 1
            idx = buf.find(b"\n", start)
        if start:
            del buf[:start]
        self._searched = len(buf)
        return lines

    def finish(self) -> list[bytes]:
        """End of body: the unterminated last line, if any."""
        rest = bytes(self._buf)
        self._buf.clear()
        self._searched = 0
        return [rest] if rest else []

    @classmethod
    def iterate(cls, chunks: Iterable[bytes]) -> Iterator[bytes]:
        splitter = cls()
        for chunk in chunks:
            if chunk:
                yield from splitter.feed(chunk)
        yield from splitter.finish()

    @classmethod
    async def aiterate(cls, chunks: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        splitter = cls()
        async for chunk in chunks:
            if chunk:
                for line in splitter.feed(chunk):
                    yield line
        for line in splitter.finish():
            yield line


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
        release: Callable[[bool], None],
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
        """Yield newline-terminated byte lines from arbitrary body chunks.

        Linear in the bytes received, however long a line is (see
        :class:`LineSplitter`)."""
        return LineSplitter.iterate(self)

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
        release: Callable[[bool], Awaitable[None]],
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

    def aiter_lines(self) -> AsyncIterator[bytes]:
        """Yield newline-terminated byte lines from arbitrary body chunks.

        Linear in the bytes received, however long a line is (see
        :class:`LineSplitter`)."""
        return LineSplitter.aiterate(self)

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
