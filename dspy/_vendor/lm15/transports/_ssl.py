"""TLS for the stdlib transports: one context, and the ways to put it on a socket.

The socket transports load this module only when a request names an HTTPS URL.
This keeps plain HTTP usable on Python builds without ``ssl``. If such a build
does request HTTPS, :class:`TLS` raises a transport error that names the host's
fetch transport as the Pyodide alternative.

We rely on the stdlib ``create_default_context``, which on Python 3.10+ loads
the system trust store correctly on Linux, macOS, and Windows. No certifi bundle
is shipped: set ``SSL_CERT_FILE`` if the system store is broken, or pass an
explicit ``ca_bundle=`` to the transport.
"""
from __future__ import annotations

import asyncio
import socket

try:
    import ssl
except ImportError:  # pragma: no cover - Pyodide and reduced Python builds
    ssl = None  # type: ignore[assignment]

from ._exceptions import ConnectError, ConnectTimeout
from ._timeouts import wait_for


def _close(sock: socket.socket) -> None:
    try:
        sock.close()
    except Exception:
        pass


def _is_ssl_error(exc: OSError) -> bool:
    return ssl is not None and isinstance(exc, ssl.SSLError)


class TLS:
    """The TLS half of a transport, holding the context its connections share."""

    __slots__ = ("_ctx",)

    def __init__(self, *, verify: bool = True, ca_bundle: str | None = None) -> None:
        if ssl is None:  # pragma: no cover - Pyodide
            raise ConnectError(
                "this Python has no ssl module (Pyodide?), so the socket transports "
                "cannot open TLS; use lm15.transports.FetchTransport, the host's fetch"
            )
        if not verify:
            self._ctx = ssl._create_unverified_context()
            return
        self._ctx = ssl.create_default_context()
        if ca_bundle:
            self._ctx.load_verify_locations(cafile=ca_bundle)

    def wrap(
        self, sock: socket.socket, *, server_hostname: str, timeout: float
    ) -> socket.socket:
        """Return ``sock`` with TLS, or close it and describe the handshake failure."""
        try:
            sock.settimeout(timeout)
            return self._ctx.wrap_socket(sock, server_hostname=server_hostname)
        except OSError as exc:
            _close(sock)
            raise ConnectError(f"TLS handshake failed: {exc}") from exc

    async def connect(
        self, *, host: str, port: int, server_hostname: str, timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a TCP connection and run its TLS handshake in one asyncio call."""
        try:
            return await wait_for(
                asyncio.open_connection(
                    host=host,
                    port=port,
                    ssl=self._ctx,
                    server_hostname=server_hostname,
                ),
                timeout=timeout,
                cancel_result=lambda pair: pair[1].close(),
            )
        except asyncio.TimeoutError as exc:
            raise ConnectTimeout(f"timed out connecting to {host}:{port}") from exc
        except OSError as exc:
            if _is_ssl_error(exc):
                raise ConnectError(f"TLS handshake failed: {exc}") from exc
            raise ConnectError(f"failed to connect to {host}:{port}: {exc}") from exc

    async def connect_over(
        self, sock: socket.socket, *, server_hostname: str, timeout: float
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Run TLS over a socket that a proxy has already tunneled to the origin."""
        try:
            return await wait_for(
                asyncio.open_connection(
                    sock=sock,
                    ssl=self._ctx,
                    server_hostname=server_hostname,
                ),
                timeout=timeout,
                cancel_result=lambda pair: pair[1].close(),
            )
        except asyncio.TimeoutError as exc:
            _close(sock)
            raise ConnectTimeout("TLS handshake through proxy timed out") from exc
        except OSError as exc:
            _close(sock)
            if _is_ssl_error(exc):
                raise ConnectError(f"TLS handshake failed: {exc}") from exc
            raise ConnectError(f"TLS handshake through proxy failed: {exc}") from exc
