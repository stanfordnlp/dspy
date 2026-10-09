"""
Async transport built on `asyncio.open_connection`.

Mirror of _sync.py but async end-to-end.  Key differences:

- `asyncio.open_connection` gives us a (reader, writer) pair, not a raw socket.
- Stale-connection detection: we peek at the reader's transport socket with
  select (fd-level), same trick as the sync side.
- Cancellation correctness: if the caller's task is cancelled mid-stream,
  we must close the writer (not return it to the pool), then re-raise
  CancelledError without awaiting anything that could itself be cancelled.
- Timeouts use a cancellation-safe wait helper, including on Python 3.10/3.11
  where asyncio.wait_for can swallow simultaneous caller cancellation.
- Lifetime: `aclose()` (or `async with`) closes every connection.  A
  transport collected without it closes its idle connections from a
  finalizer; when their event loop is already gone the socket underneath
  is closed directly, since the asyncio transport can no longer do it.
- TLS lives in `_ssl.py`, which this module imports on the first https request
  and not before.  That module needs the stdlib `ssl`; this one does not, so a
  CPython build without `ssl` still carries plain HTTP through here.
"""
from __future__ import annotations

import asyncio
import select
import socket
import weakref
from typing import TYPE_CHECKING, AsyncIterator

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
    ResponseHeadParser,
    build_request_head,
)
from ._limits import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_POOL_TIMEOUT,
    DEFAULT_READ_TIMEOUT,
    DEFAULT_WRITE_TIMEOUT,
    check_max_connections,
    pool_timeout_hint,
    read_timeout_hint,
)
from ._proxy import ProxyRoute, connect_payload, proxy_route_for, route_origin
from ._timeouts import wait_for
from ._types import AsyncTransportResponse, TransportRequest
from ._url import ParsedURL, parse_url

if TYPE_CHECKING:
    from ._ssl import TLS


_READ_CHUNK = 64 * 1024


# ─── Async connection wrapper ────────────────────────────────────────


class _AsyncConnection:
    __slots__ = ("origin", "reader", "writer", "closed")

    def __init__(
        self,
        origin: tuple[str, str, int],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.origin = origin
        self.reader = reader
        self.writer = writer
        self.closed = False

    def is_stale(self) -> bool:
        if self.closed:
            return True
        # Look at the underlying socket's FD for readability
        try:
            sock = self.writer.get_extra_info("socket")
        except Exception:
            return True
        if sock is None:
            return True
        try:
            r, _, _ = select.select([sock], [], [], 0)
            if not r:
                return False
            # Readable on an idle keepalive means EOF/close_notify or
            # unexpected unread bytes, so the connection is not safe to reuse.
            # Avoid MSG_PEEK: TLS sockets/proxies may reject non-zero recv()
            # flags, and peeking is unnecessary once select() reports readable.
            return True
        except (OSError, ValueError):
            return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        _close_writer(self.writer)


def _close_writer(writer: asyncio.StreamWriter) -> None:
    """Close a StreamWriter; when its loop is already closed (the owner was
    collected after ``asyncio.run`` returned) asyncio cannot schedule the
    close, so the socket underneath is closed directly.

    The fallback reaches asyncio's private ``_sock`` (through the TLS
    protocol's inner transport when there is one) and does what the
    loop's own ``_call_connection_lost`` would have done: close it and
    forget it, so the transport's finalizer neither warns nor double-
    closes.  Best effort by construction; a future asyncio that renames
    the field degrades to the warning, never to an exception."""
    try:
        writer.close()
        return
    except Exception:
        pass
    transport = getattr(writer, "transport", None)
    ssl_protocol = getattr(transport, "_ssl_protocol", None)
    if ssl_protocol is not None:
        transport = getattr(ssl_protocol, "_transport", None)
    sock = getattr(transport, "_sock", None)
    if sock is None:
        return
    try:
        sock.close()
    except Exception:
        pass
    try:
        transport._sock = None
    except Exception:
        pass


# ─── Async pool ──────────────────────────────────────────────────────


class _AsyncConnectionPool:
    def __init__(self, max_connections: int) -> None:
        self._max = max_connections
        self._idle: dict[tuple[str, str, int], list[_AsyncConnection]] = {}
        self._in_use: set[_AsyncConnection] = set()
        self._lock = asyncio.Lock()
        self._slot = asyncio.Semaphore(max_connections)
        self._total_opened = 0
        self._closed = False
        weakref.finalize(self, _close_idle, self._idle)

    async def acquire_slot(self, timeout: float | None = None) -> None:
        if self._closed:
            raise TransportError("transport is closed")
        if timeout is None:
            await self._slot.acquire()
            return
        try:
            await wait_for(self._slot.acquire(), timeout=timeout, cancel_result=lambda _: self._slot.release())
        except asyncio.TimeoutError:
            raise TransportError(pool_timeout_hint(timeout, self._max))

    def release_slot(self) -> None:
        self._slot.release()

    async def checkout(self, origin: tuple[str, str, int]) -> _AsyncConnection | None:
        async with self._lock:
            q = self._idle.get(origin)
            while q:
                conn = q.pop()
                if conn.is_stale():
                    conn.close()
                    continue
                self._in_use.add(conn)
                return conn
            return None

    async def checkin(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.discard(conn)
            if self._closed or conn.closed:
                conn.close()
                return
            self._idle.setdefault(conn.origin, []).append(conn)

    async def discard(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.discard(conn)
        conn.close()

    async def register_new(self, conn: _AsyncConnection) -> None:
        async with self._lock:
            self._in_use.add(conn)
            self._total_opened += 1

    def stats(self) -> dict:
        return {
            "idle": sum(len(q) for q in self._idle.values()),
            "in_use": len(self._in_use),
            "total_opened": self._total_opened,
        }

    async def close_all(self) -> None:
        async with self._lock:
            self._closed = True
            for q in self._idle.values():
                for conn in q:
                    conn.close()
            self._idle.clear()
            for conn in list(self._in_use):
                conn.close()
            self._in_use.clear()


def _close_idle(idle: dict) -> None:
    for q in list(idle.values()):
        for conn in q:
            conn.close()
    idle.clear()


# ─── Transport ───────────────────────────────────────────────────────


class StdlibAsyncTransport:
    """Async HTTP/1.1 transport on asyncio streams.  Same knobs and per-
    operation timeout semantics as :class:`StdlibTransport`.  Its pool
    belongs to the event loop that first used it; one transport per loop.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float | None = DEFAULT_POOL_TIMEOUT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        verify: bool = True,
        ca_bundle: str | None = None,
        user_agent: str = "lm15/stdlib",
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._pool_timeout = pool_timeout
        self._max_connections = check_max_connections(max_connections)
        self._user_agent = user_agent
        self._verify = verify
        self._ca_bundle = ca_bundle
        self._proxy = proxy
        self._trust_env = trust_env
        self._tls: TLS | None = None
        self._pool = _AsyncConnectionPool(self._max_connections)
        self._closed = False

    @property
    def max_connections(self) -> int:
        return self._max_connections

    def copy(self) -> "StdlibAsyncTransport":
        """A fresh, open transport with this one's configuration and none of
        its connections (an interactive runner closed the original)."""
        return type(self)(
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
            write_timeout=self._write_timeout,
            pool_timeout=self._pool_timeout,
            max_connections=self._max_connections,
            verify=self._verify,
            ca_bundle=self._ca_bundle,
            user_agent=self._user_agent,
            proxy=self._proxy,
            trust_env=self._trust_env,
        )

    def _tls_half(self) -> "TLS":
        """Build the TLS half on first https use. `._ssl` needs the stdlib `ssl` module,
        which a reduced build may not have — and a plain-HTTP caller never asks for it."""
        from ._ssl import TLS

        if self._tls is None:
            self._tls = TLS(verify=self._verify, ca_bundle=self._ca_bundle)
        return self._tls

    def pool_stats(self) -> dict:
        return self._pool.stats()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._pool.close_all()

    async def __aenter__(self) -> "StdlibAsyncTransport":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ─── Main entry point ───

    def stream(self, request: TransportRequest) -> "_AsyncStreamCM":
        """Return an async context manager that produces an AsyncTransportResponse."""
        return _AsyncStreamCM(self, request)

    async def _do_stream(self, request: TransportRequest) -> AsyncTransportResponse:
        if self._closed:
            raise TransportError("transport is closed")

        parsed = parse_url(request.url)
        proxy = proxy_route_for(parsed, proxy=self._proxy, trust_env=self._trust_env)
        origin = route_origin(parsed, proxy)
        connect_timeout = request.connect_timeout or self._connect_timeout
        read_timeout = request.read_timeout or self._read_timeout
        write_timeout = request.write_timeout or self._write_timeout

        await self._pool.acquire_slot(timeout=self._pool_timeout)
        slot_released = {"done": False}

        def release_slot_once() -> None:
            if not slot_released["done"]:
                slot_released["done"] = True
                self._pool.release_slot()

        conn: _AsyncConnection | None = None
        try:
            attempt = 0
            while True:
                conn = (
                    await self._pool.checkout(origin) if attempt == 0 else None
                )
                reused = conn is not None
                if conn is None:
                    conn = await self._open(
                        parsed, proxy=proxy, origin=origin, connect_timeout=connect_timeout
                    )
                    await self._pool.register_new(conn)

                try:
                    await self._send_request(
                        conn, request, parsed, proxy=proxy, write_timeout=write_timeout
                    )
                    break
                except (WriteError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                    if not reused or attempt > 0:
                        await self._pool.discard(conn)
                        if isinstance(exc, WriteError):
                            raise
                        raise WriteError(str(exc)) from exc
                    await self._pool.discard(conn)
                    conn = None
                    attempt += 1
                    continue

            head = await self._read_head(conn, read_timeout=read_timeout)
            decoder = head.body_decoder(request.method)
            keep_alive = head.keep_alive()

            release_conn = self._make_release(conn, keep_alive)

            async def chunks_gen() -> AsyncIterator[bytes]:
                aborted = False
                try:
                    if head.leftover:
                        for out in decoder.feed(head.leftover):
                            if out:
                                yield out
                        if decoder.complete:
                            return
                    while not decoder.complete:
                        try:
                            data = await wait_for(
                                conn.reader.read(_READ_CHUNK),
                                timeout=read_timeout,
                            )
                        except asyncio.TimeoutError as exc:
                            aborted = True
                            raise ReadTimeout(
                                f"read timed out mid-body: {read_timeout_hint(read_timeout)}"
                            ) from exc
                        except asyncio.CancelledError:
                            aborted = True
                            raise
                        except OSError as exc:
                            aborted = True
                            raise ReadError(f"read failed: {exc}") from exc
                        if not data:
                            try:
                                decoder.eof()
                            except ProtocolError:
                                aborted = True
                                raise
                            tail = decoder.drain()
                            if tail:
                                yield tail
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
                        release_slot_once()

            async def release_with_slot(body_consumed: bool) -> None:
                try:
                    await release_conn(body_consumed)
                finally:
                    release_slot_once()

            return AsyncTransportResponse(
                status=head.status,
                reason=head.reason,
                headers=head.headers,
                http_version=head.http_version,
                chunks=chunks_gen(),
                release=release_with_slot,
            )
        except BaseException:
            if conn is not None:
                await self._pool.discard(conn)
            release_slot_once()
            raise

    # ─── Connection I/O ───

    async def _open(
        self,
        parsed: ParsedURL,
        *,
        proxy: ProxyRoute | None,
        origin: tuple[str, str, int],
        connect_timeout: float,
    ) -> _AsyncConnection:
        tls = self._tls_half() if parsed.is_tls else None

        if proxy is not None and tls is not None:
            # CONNECT over a raw socket, then hand it to open_connection
            # for the end-to-end TLS handshake (works on 3.10; StreamWriter
            # gained start_tls only in 3.11).
            tunnel = await self._connect_tunnel(parsed, proxy, timeout=connect_timeout)
            reader, writer = await tls.connect_over(
                tunnel, server_hostname=parsed.host, timeout=connect_timeout
            )
            return _AsyncConnection(origin, reader, writer)

        connect_host = proxy.host if proxy is not None else parsed.host
        connect_port = proxy.port if proxy is not None else parsed.port
        if tls is not None:
            reader, writer = await tls.connect(
                host=connect_host,
                port=connect_port,
                server_hostname=parsed.host,
                timeout=connect_timeout,
            )
        else:
            try:
                reader, writer = await wait_for(
                    asyncio.open_connection(host=connect_host, port=connect_port),
                    timeout=connect_timeout,
                    cancel_result=lambda pair: pair[1].close(),
                )
            except asyncio.TimeoutError as exc:
                raise ConnectTimeout(
                    f"timed out connecting to {connect_host}:{connect_port}"
                ) from exc
            except OSError as exc:
                raise ConnectError(
                    f"failed to connect to {connect_host}:{connect_port}: {exc}"
                ) from exc

        # TCP_NODELAY on the underlying socket
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        return _AsyncConnection(origin, reader, writer)

    async def _connect_tunnel(
        self, parsed: ParsedURL, proxy: ProxyRoute, *, timeout: float
    ) -> socket.socket:
        """Open a raw socket to the proxy and CONNECT it to the TLS target."""
        loop = asyncio.get_running_loop()
        try:
            sock = await wait_for(
                asyncio.to_thread(
                    socket.create_connection, (proxy.host, proxy.port), timeout
                ),
                timeout=timeout,
                cancel_result=lambda sock: sock.close(),
            )
        except asyncio.TimeoutError as exc:
            raise ConnectTimeout(
                f"timed out connecting to proxy {proxy.host}:{proxy.port}"
            ) from exc
        except OSError as exc:
            raise ConnectError(
                f"failed to connect to proxy {proxy.host}:{proxy.port}: {exc}"
            ) from exc
        sock.setblocking(False)
        try:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            await wait_for(
                loop.sock_sendall(sock, connect_payload(parsed, proxy)), timeout=timeout
            )
            parser = ResponseHeadParser()
            while not parser.complete:
                data = await wait_for(
                    loop.sock_recv(sock, _READ_CHUNK), timeout=timeout
                )
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
        except asyncio.TimeoutError as exc:
            sock.close()
            raise ConnectTimeout(
                f"CONNECT via {proxy.host}:{proxy.port} timed out"
            ) from exc
        except (ConnectError, ProtocolError):
            sock.close()
            raise
        except OSError as exc:
            sock.close()
            raise ConnectError(f"CONNECT via {proxy.host}:{proxy.port} failed: {exc}") from exc
        return sock

    async def _send_request(
        self,
        conn: _AsyncConnection,
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
        try:
            conn.writer.write(head)
            if body_length is not None and body:
                conn.writer.write(body)
            await wait_for(conn.writer.drain(), timeout=write_timeout)
        except asyncio.TimeoutError as exc:
            raise WriteTimeout(f"write timed out: {exc}") from exc
        except (BrokenPipeError, ConnectionResetError):
            raise
        except OSError as exc:
            raise WriteError(f"write failed: {exc}") from exc

    async def _read_head(
        self, conn: _AsyncConnection, *, read_timeout: float
    ) -> ResponseHeadParser:
        parser = ResponseHeadParser()
        while not parser.complete:
            try:
                data = await wait_for(
                    conn.reader.read(_READ_CHUNK), timeout=read_timeout
                )
            except asyncio.TimeoutError as exc:
                raise ReadTimeout(
                    f"read timed out waiting for headers: {read_timeout_hint(read_timeout)}"
                ) from exc
            except OSError as exc:
                raise ReadError(f"read failed: {exc}") from exc
            if not data:
                raise ReadError("server closed connection before sending response")
            parser.feed(data)
        return parser

    def _make_release(self, conn: _AsyncConnection, keep_alive: bool):
        async def release(body_consumed: bool) -> None:
            if body_consumed and keep_alive and not conn.closed:
                await self._pool.checkin(conn)
            else:
                await self._pool.discard(conn)
        return release


class _AsyncStreamCM:
    """Async context manager wrapper so callers can write:

        async with transport.stream(req) as resp:
            async for chunk in resp: ...
    """

    def __init__(self, transport: StdlibAsyncTransport, request: TransportRequest) -> None:
        self._transport = transport
        self._request = request
        self._response: AsyncTransportResponse | None = None

    def __await__(self):
        return self._transport._do_stream(self._request).__await__()

    async def __aenter__(self) -> AsyncTransportResponse:
        self._response = await self._transport._do_stream(self._request)
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._response is not None:
            await self._response.aclose()
