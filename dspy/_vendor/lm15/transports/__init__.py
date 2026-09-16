"""
lm15.transports — Minimal stdlib-only HTTP/1.1 transports.

Public API:
    TransportRequest, TransportResponse         — transport-level request/response models
    StdlibTransport           — sync transport (blocking, socket-based)
    StdlibAsyncTransport      — async transport (asyncio-native)
    FetchTransport            — async transport over the host's fetch (Pyodide: a page, a worker)
    Timeouts                  — per-operation timeouts (connect/read/write/pool)
    TransportError + subclasses
"""

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
from ._limits import DEFAULT_MAX_CONNECTIONS, Timeouts
from ._types import TransportRequest, TransportResponse, AsyncTransportResponse
from ._sync import StdlibTransport
from ._async import StdlibAsyncTransport
from ._fetch import FetchTransport

__all__ = [
    "Timeouts",
    "DEFAULT_MAX_CONNECTIONS",
    "TransportRequest",
    "TransportResponse",
    "AsyncTransportResponse",
    "StdlibTransport",
    "StdlibAsyncTransport",
    "FetchTransport",
    "TransportError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "WriteError",
    "WriteTimeout",
    "ProtocolError",
]
