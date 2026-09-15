"""
Sync transport built on the stdlib `socket` + `ssl` modules.

Design:

- A single transport owns a connection pool keyed by origin (scheme, host,
  port).  Up to `max_connections` concurrent connections per pool total (not
  per origin — keeping it simple for lm15's workload).
- `stream(request)` takes a connection (new or reused), writes the request,
  reads the response head, and returns a `TransportResponse` whose iteration yields
  body bytes.  The response's `__exit__`/generator teardown releases the
  connection (to the pool if reusable, otherwise closed).
- Staleness detection: before reusing an idle connection, we peek with
  `select` to see if the socket has readable bytes.  For an idle HTTP keepalive
  socket, readable means the server sent EOF — so we drop it and open fresh.
- If the server sent `Connection: close`, or the body is still outstanding
  when the response closes, the connection is closed rather than reused.
- Proxies: `proxy=` pins one explicitly; otherwise `trust_env=True` (default)
  honors HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY.  Plain-HTTP targets
  are forwarded absolute-URI; TLS targets are tunneled with CONNECT and the
  handshake runs end-to-end to the origin (see `_proxy.py`).
"""
from __future__ import annotations

import select
import socket
import ssl
import threading
from typing import Iterator

from ._exceptions import (
    ConnectError,
    ConnectTimeout,
    ProtocolError,
    ReadError,
    ReadTimeout,
    TransportError,
    WriteError,
    WriteTimeout,
)
from ._http11 import (
    ChunkedDecoder,
    ContentLengthDecoder,
    EOFDecoder,
    ResponseHeadParser,
    build_request_head,
)
from ._proxy import ProxyRoute, connect_payload, proxy_route_for, route_origin
from ._ssl import make_ssl_context
from ._types import TransportRequest, TransportResponse
from ._url import ParsedURL, parse_url


_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 60.0
_DEFAULT_WRITE_TIMEOUT = 60.0
_READ_CHUNK = 64 * 1024


# ─── Connection wrapper ──────────────────────────────────────────────


class _SyncConnection:
    """Owns a single TCP/TLS socket keyed by an origin."""

    __slots__ = ("origin", "sock", "in_use", "closed")

    def __init__(self, origin: tuple[str, str, int], sock: socket.socket) -> None:
        self.origin = origin
        self.sock = sock
        self.in_use = False
        self.closed = False

    def is_stale(self) -> bool:
        """Return True if the peer has closed the connection while it was idle."""
        if self.closed:
            return True
        try:
            r, _, _ = select.select([self.sock], [], [], 0)
            if not r:
                return False
            # Readable on an idle socket means the peer sent EOF/close_notify,
            # or that unexpected bytes were left unread.  Either way this
            # connection is not safe to reuse.  Do not probe with MSG_PEEK here:
            # ssl.SSLSocket.recv(..., flags) raises ValueError for non-zero
            # flags, and peeking is unnecessary once select() says an idle
            # socket is readable.
            return True
        except (OSError, ValueError):
            return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.close()
        except Exception:
            pass


# ─── Connection pool ─────────────────────────────────────────────────


class _ConnectionPool:
    """Small LIFO pool.  LIFO (not FIFO) minimizes chance of reusing stale
    keepalive connections, since the most recently used is the most likely
    to still be alive."""

    def __init__(self, max_connections: int) -> None:
        self._max = max_connections
        self._idle: dict[tuple[str, str, int], list[_SyncConnection]] = {}
        self._in_use: set[_SyncConnection] = set()
        self._lock = threading.Lock()
        self._slot = threading.Semaphore(max_connections)
        self._total_opened = 0
        self._closed = False

    def acquire_slot(self, timeout: float | None = None) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        ok = self._slot.acquire(timeout=timeout)
        if not ok:
            raise TransportError("timed out waiting for connection pool slot")

    def release_slot(self) -> None:
        self._slot.release()

    def checkout(self, origin: tuple[str, str, int]) -> _SyncConnection | None:
        """Return an idle connection for this origin, or None if none usable."""
        with self._lock:
            q = self._idle.get(origin)
            while q:
                conn = q.pop()
                if conn.is_stale():
                    conn.close()
                    continue
                conn.in_use = True
                self._in_use.add(conn)
                return conn
            return None

    def checkin(self, conn: _SyncConnection) -> None:
        """Return a connection to the idle pool."""
        with self._lock:
            self._in_use.discard(conn)
            if self._closed or conn.closed:
                conn.close()
                return
            conn.in_use = False
            self._idle.setdefault(conn.origin, []).append(conn)

    def discard(self, conn: _SyncConnection) -> None:
        """Close a connection and remove it from all tracking."""
        with self._lock:
            self._in_use.discard(conn)
        conn.close()

    def register_new(self, conn: _SyncConnection) -> None:
        with self._lock:
            self._in_use.add(conn)
            self._total_opened += 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "idle": sum(len(q) for q in self._idle.values()),
                "in_use": len(self._in_use),
                "total_opened": self._total_opened,
            }

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            for q in self._idle.values():
                for conn in q:
                    conn.close()
            self._idle.clear()
            for conn in list(self._in_use):
                conn.close()
            self._in_use.clear()


# ─── Transport ───────────────────────────────────────────────────────


class StdlibTransport:
    """Sync HTTP/1.1 transport using only the Python standard library."""

    def __init__(
        self,
        *,
        connect_timeout: float = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = _DEFAULT_READ_TIMEOUT,
        write_timeout: float = _DEFAULT_WRITE_TIMEOUT,
        max_connections: int = 10,
        verify: bool = True,
        ca_bundle: str | None = None,
        user_agent: str = "lm15/stdlib",
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._user_agent = user_agent
        self._verify = verify
        self._ca_bundle = ca_bundle
        self._proxy = proxy
        self._trust_env = trust_env
        self._ssl_ctx: ssl.SSLContext | None = None
        self._ssl_lock = threading.Lock()
        self._pool = _ConnectionPool(max_connections)
        self._closed = False

    def _get_ssl_ctx(self) -> ssl.SSLContext:
        with self._ssl_lock:
            if self._ssl_ctx is None:
                self._ssl_ctx = make_ssl_context(
                    verify=self._verify, ca_bundle=self._ca_bundle
                )
            return self._ssl_ctx

    def pool_stats(self) -> dict:
        return self._pool.stats()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.close_all()

    def __enter__(self) -> "StdlibTransport":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ─── Main entry point ───

    def stream(self, request: TransportRequest) -> TransportResponse:
        """Open a stream to the URL, returning a TransportResponse whose iteration
        yields body bytes.

        The response MUST be used as a context manager (or have `.close()`
        called) — otherwise the underlying connection leaks.
        """
        if self._closed:
            raise TransportError("transport is closed")

        parsed = parse_url(request.url)
        proxy = proxy_route_for(parsed, proxy=self._proxy, trust_env=self._trust_env)
        origin = route_origin(parsed, proxy)
        connect_timeout = request.connect_timeout or self._connect_timeout
        read_timeout = request.read_timeout or self._read_timeout
        write_timeout = request.write_timeout or self._write_timeout

        # Acquire a pool slot FIRST (may block if we're at max_connections)
        self._pool.acquire_slot(timeout=connect_timeout * 5)
        slot_released = False

        def release_slot_once() -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                self._pool.release_slot()

        conn: _SyncConnection | None = None
        try:
            # Try to reuse an idle connection.  If the write fails because the
            # connection went stale between stale-check and write, open fresh
            # and retry once.
            attempt = 0
            while True:
                conn = self._pool.checkout(origin) if attempt == 0 else None
                reused = conn is not None
                if conn is None:
                    conn = self._open(
                        parsed, proxy=proxy, origin=origin, connect_timeout=connect_timeout
                    )
                    self._pool.register_new(conn)

                try:
                    self._send_request(
                        conn, request, parsed, proxy=proxy, write_timeout=write_timeout
                    )
                    break
                except (WriteError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                    if not reused or attempt > 0:
                        self._pool.discard(conn)
                        if isinstance(exc, WriteError):
                            raise
                        raise WriteError(str(exc)) from exc
                    # Reused connection was stale — close it and retry fresh
                    self._pool.discard(conn)
                    conn = None
                    attempt += 1
                    continue

            # Receive response head
            head = self._read_head(conn, read_timeout=read_timeout)
            decoder = head.body_decoder(request.method)
            keep_alive = head.keep_alive()

            release_conn = self._make_release(conn, keep_alive)
            # Transfer ownership of the pool slot to the TransportResponse
            # (TransportResponse.close will trigger release_slot via _release)
            chunks = self._iter_body(
                conn, decoder, initial=head.leftover,
                read_timeout=read_timeout, on_finish=release_slot_once,
                on_abort=release_slot_once,
            )
            return TransportResponse(
                status=head.status,
                reason=head.reason,
                headers=head.headers,
                http_version=head.http_version,
                chunks=chunks,
                release=release_conn,
            )
        except BaseException:
            release_slot_once()
            if conn is not None and conn not in self._pool._in_use:
                # never happens — register_new or checkout put it in_use
                pass
            if conn is not None:
                self._pool.discard(conn)
            raise

    # ─── Connection I/O ───

    def _open(
        self,
        parsed: ParsedURL,
        *,
        proxy: ProxyRoute | None,
        origin: tuple[str, str, int],
        connect_timeout: float,
    ) -> _SyncConnection:
        connect_host = proxy.host if proxy is not None else parsed.host
        connect_port = proxy.port if proxy is not None else parsed.port
        try:
            sock = socket.create_connection(
                (connect_host, connect_port), timeout=connect_timeout,
            )
        except socket.timeout as exc:
            raise ConnectTimeout(
                f"timed out connecting to {connect_host}:{connect_port}"
            ) from exc
        except OSError as exc:
            raise ConnectError(
                f"failed to connect to {connect_host}:{connect_port}: {exc}"
            ) from exc

        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        if proxy is not None and parsed.is_tls:
            try:
                self._connect_tunnel(sock, parsed, proxy, timeout=connect_timeout)
            except BaseException:
                try:
                    sock.close()
                except Exception:
                    pass
                raise

        if parsed.is_tls:
            try:
                ctx = self._get_ssl_ctx()
                sock.settimeout(connect_timeout)
                sock = ctx.wrap_socket(sock, server_hostname=parsed.host)
            except (ssl.SSLError, OSError) as exc:
                try:
                    sock.close()
                except Exception:
                    pass
                raise ConnectError(f"TLS handshake failed: {exc}") from exc

        return _SyncConnection(origin, sock)

    def _connect_tunnel(
        self, sock: socket.socket, parsed: ParsedURL, proxy: ProxyRoute, *, timeout: float
    ) -> None:
        """Ask the proxy to open a CONNECT tunnel to the TLS target."""
        sock.settimeout(timeout)
        try:
            sock.sendall(connect_payload(parsed, proxy))
        except socket.timeout as exc:
            raise ConnectTimeout(f"CONNECT to proxy timed out: {exc}") from exc
        except OSError as exc:
            raise ConnectError(f"CONNECT write to proxy failed: {exc}") from exc

        parser = ResponseHeadParser()
        while not parser.complete:
            try:
                data = sock.recv(_READ_CHUNK)
            except socket.timeout as exc:
                raise ConnectTimeout(f"timed out waiting for CONNECT response: {exc}") from exc
            except OSError as exc:
                raise ConnectError(f"read of CONNECT response failed: {exc}") from exc
            if not data:
                raise ConnectError("proxy closed connection during CONNECT")
            parser.feed(data)
        if not (200 <= parser.status < 300):
            raise ConnectError(
                f"proxy refused CONNECT to {proxy.authority(parsed)}: "
                f"{parser.status} {parser.reason}"
            )
        if parser.leftover:
            raise ProtocolError("proxy sent unexpected bytes after the CONNECT response")

    def _send_request(
        self,
        conn: _SyncConnection,
        request: TransportRequest,
        parsed: ParsedURL,
        *,
        proxy: ProxyRoute | None,
        write_timeout: float,
    ) -> None:
        body = request.body or b""
        has_body = request.method.upper() not in ("GET", "HEAD", "DELETE") or bool(body)
        body_length = len(body) if has_body else None
        target = parsed.target
        headers = request.headers
        if proxy is not None and not parsed.is_tls:
            # Forward-proxy plain HTTP: absolute-URI request line; the
            # Host header still names the target (build_request_head).
            target = f"http://{parsed.host_header()}{parsed.target}"
            if proxy.basic_auth is not None:
                headers = [*headers, ("Proxy-Authorization", proxy.basic_auth)]
        head = build_request_head(
            method=request.method,
            target=target,
            host=parsed.host,
            port=parsed.port,
            is_tls=parsed.is_tls,
            headers=headers,
            body_length=body_length,
            user_agent=self._user_agent,
        )
        conn.sock.settimeout(write_timeout)
        try:
            conn.sock.sendall(head)
            if body_length is not None and body:
                conn.sock.sendall(body)
        except socket.timeout as exc:
            raise WriteTimeout(f"write timed out: {exc}") from exc
        except (BrokenPipeError, ConnectionResetError):
            raise  # caller retries reused-but-stale connections
        except OSError as exc:
            raise WriteError(f"write failed: {exc}") from exc

    def _read_head(
        self, conn: _SyncConnection, *, read_timeout: float
    ) -> ResponseHeadParser:
        parser = ResponseHeadParser()
        conn.sock.settimeout(read_timeout)
        while not parser.complete:
            try:
                data = conn.sock.recv(_READ_CHUNK)
            except socket.timeout as exc:
                raise ReadTimeout(f"read timed out waiting for headers: {exc}") from exc
            except OSError as exc:
                raise ReadError(f"read failed: {exc}") from exc
            if not data:
                raise ReadError("server closed connection before sending response")
            parser.feed(data)
        return parser

    def _iter_body(
        self,
        conn: _SyncConnection,
        decoder,
        *,
        initial: bytes,
        read_timeout: float,
        on_finish,
        on_abort,
    ) -> Iterator[bytes]:
        """Generator that yields decoded body chunks.

        Calls on_finish() on clean completion, on_abort() on exception.
        """
        aborted = False
        try:
            if initial:
                for out in decoder.feed(initial):
                    if out:
                        yield out
                if decoder.complete:
                    return
            conn.sock.settimeout(read_timeout)
            while not decoder.complete:
                try:
                    data = conn.sock.recv(_READ_CHUNK)
                except socket.timeout as exc:
                    aborted = True
                    raise ReadTimeout(f"read timed out mid-body: {exc}") from exc
                except OSError as exc:
                    aborted = True
                    raise ReadError(f"read failed: {exc}") from exc
                if not data:
                    # EOF
                    try:
                        decoder.eof()
                    except ProtocolError:
                        aborted = True
                        raise
                    break
                try:
                    for out in decoder.feed(data):
                        if out:
                            yield out
                except ProtocolError:
                    aborted = True
                    raise
        finally:
            if aborted:
                on_abort()
            else:
                on_finish()

    def _make_release(self, conn: _SyncConnection, keep_alive: bool):
        """Create the release callback the TransportResponse invokes on close."""
        def release(body_consumed: bool) -> None:
            if body_consumed and keep_alive and not conn.closed:
                self._pool.checkin(conn)
            else:
                self._pool.discard(conn)
        return release
