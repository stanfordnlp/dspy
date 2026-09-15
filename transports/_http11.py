"""
HTTP/1.1 codec — pure functions over bytes.

Used by both the sync and async transports.  All state is in small helper
objects (ResponseHeadParser, ContentLengthDecoder, ChunkedDecoder, EOFDecoder);
none of them do any I/O.  The transport feeds them bytes and reads back
decoded body chunks + a completion flag.

This is deliberately minimal:
    - HTTP/1.1 only (no HTTP/2, no HTTP/1.0 write side)
    - No Content-Encoding handling (gzip/deflate/br/zstd) — LLM APIs don't
      compress responses in practice, and we'd rather not pull in zlib paths
      until we need them.  Set Accept-Encoding: identity to be explicit.
    - No trailers, no pipelining, no 100-continue.
    - Chunk extensions (RFC 7230 §4.1.1) are parsed and discarded.
"""
from __future__ import annotations

from typing import Iterator

from ._exceptions import ProtocolError


# ─── Request serialization ───────────────────────────────────────────


_FORBIDDEN_BYTES = {ord("\r"), ord("\n"), 0}


def _validate_field(kind: str, value: str) -> None:
    """Reject header values / targets containing CR/LF/NUL (CRLF injection)."""
    for ch in value.encode("iso-8859-1", errors="replace"):
        if ch in _FORBIDDEN_BYTES:
            raise ProtocolError(f"forbidden byte in HTTP {kind}: {value!r}")


def build_request_head(
    *,
    method: str,
    target: str,
    host: str,
    port: int,
    is_tls: bool,
    headers: list[tuple[str, str]],
    body_length: int | None,
    user_agent: str = "lm15/stdlib",
) -> bytes:
    """Serialize the request line + headers (NOT the body).

    `body_length=None` means no body at all (e.g. GET).  `body_length=0` means
    the request has an empty body (POST with no payload).  Any other value
    becomes the Content-Length header.
    """
    _validate_field("method", method)
    _validate_field("target", target)
    _validate_field("host", host)
    for k, v in headers:
        _validate_field(f"header name ({k})", k)
        _validate_field(f"header value ({k})", v)

    lines = [f"{method} {target} HTTP/1.1"]

    # Track which headers the caller provided so we don't duplicate them
    lower_names = {k.lower() for k, _ in headers}

    if "host" not in lower_names:
        is_default = (is_tls and port == 443) or (not is_tls and port == 80)
        host_hdr = _bracketed_if_v6(host) if is_default else f"{_bracketed_if_v6(host)}:{port}"
        lines.append(f"Host: {host_hdr}")

    if "user-agent" not in lower_names:
        lines.append(f"User-Agent: {user_agent}")

    if "accept" not in lower_names:
        lines.append("Accept: */*")

    # Be explicit about no compression — we don't implement Content-Encoding.
    if "accept-encoding" not in lower_names:
        lines.append("Accept-Encoding: identity")

    if body_length is not None and "content-length" not in lower_names:
        lines.append(f"Content-Length: {body_length}")

    for k, v in headers:
        lines.append(f"{k}: {v}")

    lines.append("")  # end of headers
    lines.append("")  # blank line terminator

    return "\r\n".join(lines).encode("iso-8859-1")


def _bracketed_if_v6(host: str) -> str:
    return f"[{host}]" if ":" in host else host


# ─── Response head parser ────────────────────────────────────────────


class ResponseHeadParser:
    """Incrementally parses status line + headers up to the blank line.

    Feed bytes via `.feed(data)`.  Once `.complete` is True the status,
    reason, http_version, and headers are populated, and `.leftover` holds
    any bytes that arrived past the \\r\\n\\r\\n terminator (start of body).
    """

    def __init__(self, max_head_bytes: int = 64 * 1024) -> None:
        self._buf = bytearray()
        self._max = max_head_bytes
        self.complete = False
        self.http_version: str = ""
        self.status: int = 0
        self.reason: str = ""
        # Preserve order + duplicates
        self.headers: list[tuple[str, str]] = []
        self.leftover: bytes = b""

    def feed(self, data: bytes) -> None:
        if self.complete:
            # Anything after completion belongs to the body; we shouldn't be called.
            self.leftover += bytes(data)
            return
        self._buf.extend(data)
        if len(self._buf) > self._max:
            raise ProtocolError(
                f"response head exceeded limit ({len(self._buf)} > {self._max} bytes)"
            )
        end = self._buf.find(b"\r\n\r\n")
        if end == -1:
            # Also accept bare \n\n (spec requires \r\n\r\n but be lenient)
            return
        head = bytes(self._buf[:end])
        self.leftover = bytes(self._buf[end + 4:])
        self._buf.clear()
        self._parse_head(head)
        self.complete = True

    def _parse_head(self, head: bytes) -> None:
        # Accept \r\n or \n line terminators inside head
        lines = head.replace(b"\r\n", b"\n").split(b"\n")
        if not lines:
            raise ProtocolError("empty response head")
        status_line = lines[0].decode("iso-8859-1")
        parts = status_line.split(" ", 2)
        if len(parts) < 2:
            raise ProtocolError(f"malformed status line: {status_line!r}")
        version = parts[0]
        if not version.startswith("HTTP/"):
            raise ProtocolError(f"malformed status line: {status_line!r}")
        self.http_version = version
        try:
            self.status = int(parts[1])
        except ValueError:
            raise ProtocolError(f"malformed status code: {status_line!r}")
        self.reason = parts[2] if len(parts) == 3 else ""

        for line in lines[1:]:
            if not line:
                continue
            s = line.decode("iso-8859-1")
            if ":" not in s:
                raise ProtocolError(f"malformed header line: {s!r}")
            name, _, value = s.partition(":")
            self.headers.append((name.strip(), value.strip()))

    # ─── Convenience accessors ───

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for k, v in self.headers:
            if k.lower() == lname:
                return v
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [v for k, v in self.headers if k.lower() == lname]

    # ─── Framing decision ───

    def body_decoder(self, request_method: str) -> "_BodyDecoder":
        """Pick the right decoder based on headers (RFC 7230 §3.3.3)."""
        # No body on 1xx, 204, 304, or HEAD
        if (
            self.status // 100 == 1
            or self.status in (204, 304)
            or request_method.upper() == "HEAD"
        ):
            return NoBodyDecoder()

        te = self.header("transfer-encoding")
        if te:
            # "chunked" may be the last (or only) coding
            codings = [c.strip().lower() for c in te.split(",")]
            if codings and codings[-1] == "chunked":
                # We don't support inner codings (gzip,chunked etc.)
                if len(codings) > 1:
                    raise ProtocolError(
                        f"unsupported Transfer-Encoding stacking: {te!r}"
                    )
                return ChunkedDecoder()
            # Non-chunked TE (deprecated) — treat as identity-to-EOF
            return EOFDecoder()

        cl = self.header("content-length")
        if cl is not None:
            try:
                n = int(cl)
            except ValueError:
                raise ProtocolError(f"malformed Content-Length: {cl!r}")
            if n < 0:
                raise ProtocolError(f"negative Content-Length: {cl!r}")
            return ContentLengthDecoder(n)

        # No framing headers — HTTP/1.1 should have used Content-Length,
        # but in practice responses without either exist; read until close.
        return EOFDecoder()

    def keep_alive(self) -> bool:
        """Whether the connection may be reused after this response."""
        conn = (self.header("connection") or "").lower()
        # HTTP/1.1 default is keep-alive; HTTP/1.0 default is close.
        if self.http_version == "HTTP/1.0":
            return "keep-alive" in conn
        return "close" not in conn


# ─── Body decoders ───────────────────────────────────────────────────


class _BodyDecoder:
    """Interface: feed(data) -> iter of decoded chunks, eof() to signal close."""
    complete: bool
    leftover: bytes

    def feed(self, data: bytes) -> Iterator[bytes]:
        raise NotImplementedError

    def eof(self) -> None:
        raise NotImplementedError


class NoBodyDecoder(_BodyDecoder):
    def __init__(self) -> None:
        self.complete = True
        self.leftover = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        if data:
            self.leftover += bytes(data)
        return iter(())

    def eof(self) -> None:
        self.complete = True


class ContentLengthDecoder(_BodyDecoder):
    def __init__(self, length: int) -> None:
        self._remaining = length
        self.complete = length == 0
        self.leftover = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        if self.complete:
            if data:
                raise ProtocolError("received data after Content-Length reached")
            return
        if not data:
            return
        take = min(len(data), self._remaining)
        out = bytes(data[:take])
        self._remaining -= take
        if take < len(data):
            self.leftover = bytes(data[take:])
        if self._remaining == 0:
            self.complete = True
        yield out

    def eof(self) -> None:
        if not self.complete:
            raise ProtocolError(
                f"connection closed with {self._remaining} body bytes outstanding"
            )


class ChunkedDecoder(_BodyDecoder):
    """Decodes Transfer-Encoding: chunked.

    States:
      SIZE     — reading the chunk size line (hex + optional extensions)
      DATA     — reading chunk data
      DATA_CRLF — reading the CRLF after chunk data
      TRAILER  — reading trailer headers + terminating blank line
    """

    SIZE = 0
    DATA = 1
    DATA_CRLF = 2
    TRAILER = 3

    def __init__(self, max_line_bytes: int = 16 * 1024) -> None:
        self._buf = bytearray()
        self._state = self.SIZE
        self._remaining = 0
        self._max_line = max_line_bytes
        self.complete = False
        self.leftover = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        if data:
            self._buf.extend(data)
        while True:
            if self._state == self.SIZE:
                line, rest = self._extract_line()
                if line is None:
                    return
                assert rest is not None
                self._buf = rest
                size = self._parse_chunk_size(line)
                if size == 0:
                    self._state = self.TRAILER
                    continue
                self._remaining = size
                self._state = self.DATA
                continue

            if self._state == self.DATA:
                if not self._buf:
                    return
                take = min(len(self._buf), self._remaining)
                out = bytes(self._buf[:take])
                self._buf = self._buf[take:]
                self._remaining -= take
                if out:
                    yield out
                if self._remaining == 0:
                    self._state = self.DATA_CRLF
                    continue
                return

            if self._state == self.DATA_CRLF:
                if len(self._buf) < 2:
                    return
                if bytes(self._buf[:2]) != b"\r\n":
                    # Be lenient: accept bare \n as terminator
                    if self._buf[:1] == b"\n":
                        self._buf = self._buf[1:]
                    else:
                        raise ProtocolError("missing CRLF after chunk data")
                else:
                    self._buf = self._buf[2:]
                self._state = self.SIZE
                continue

            if self._state == self.TRAILER:
                # Lines until blank line
                line, rest = self._extract_line()
                if line is None:
                    return
                assert rest is not None
                self._buf = rest
                if line == b"":
                    self.complete = True
                    self.leftover = bytes(self._buf)
                    self._buf.clear()
                    return
                # Trailer header — ignored (we don't surface them)
                continue
            return  # unreachable

    def _extract_line(self) -> tuple[bytes, bytearray] | tuple[None, None]:
        # Find \r\n or bare \n
        idx = self._buf.find(b"\r\n")
        if idx == -1:
            bare = self._buf.find(b"\n")
            if bare == -1:
                if len(self._buf) > self._max_line:
                    raise ProtocolError("chunk line exceeds limit")
                return None, None
            line = bytes(self._buf[:bare])
            return line, self._buf[bare + 1:]
        line = bytes(self._buf[:idx])
        if idx > self._max_line:
            raise ProtocolError("chunk line exceeds limit")
        return line, self._buf[idx + 2:]

    @staticmethod
    def _parse_chunk_size(line: bytes) -> int:
        # Discard chunk extensions after ';'
        size_part = line.split(b";", 1)[0].strip()
        if not size_part:
            raise ProtocolError(f"empty chunk size line: {line!r}")
        try:
            return int(size_part, 16)
        except ValueError:
            raise ProtocolError(f"malformed chunk size: {line!r}")

    def eof(self) -> None:
        if not self.complete:
            raise ProtocolError("connection closed mid-chunk")


class EOFDecoder(_BodyDecoder):
    """Body ends when the connection closes (HTTP/1.0 style)."""

    def __init__(self) -> None:
        self.complete = False
        self.leftover = b""

    def feed(self, data: bytes) -> Iterator[bytes]:
        if data:
            yield bytes(data)

    def eof(self) -> None:
        self.complete = True
