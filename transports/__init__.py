"""
lm15.transports — Minimal stdlib-only HTTP/1.1 transports.

Public API:
    TransportRequest, TransportResponse         — transport-level request/response models
    StdlibTransport           — sync transport (blocking, socket-based)
    StdlibAsyncTransport      — async transport (asyncio-native)
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
from ._types import TransportRequest, TransportResponse, AsyncTransportResponse
from ._sync import StdlibTransport
from ._async import StdlibAsyncTransport

__all__ = [
    "TransportRequest",
    "TransportResponse",
    "AsyncTransportResponse",
    "StdlibTransport",
    "StdlibAsyncTransport",
    "TransportError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "WriteError",
    "WriteTimeout",
    "ProtocolError",
]
