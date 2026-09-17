"""The async transport over the host's ``fetch``: lm15 in a page.

Under Pyodide — CPython compiled to WebAssembly, in a browser page, a
worker, or Node — there is no socket, so :class:`StdlibAsyncTransport`
cannot open one. The host has ``fetch``. :class:`FetchTransport` is the
same :class:`AsyncTransport` protocol over it: the same
``AsyncTransportResponse``, chunk by chunk as the body arrives, cancelled
by dropping the response before its end. Every adapter's async mirror
(``AsyncOpenAIChatLM``, ``AsyncAnthropicLM``, …) takes it as
``transport=`` and is otherwise unchanged: the request bytes are the
ones the stdlib transport would send.

What a page cannot do is not hidden here: a cross-origin request goes
only where the server's CORS headers allow it (the browser refuses
before any bytes leave, and ``fetch`` reports a bare ``TypeError`` with
no status — surfaced as :class:`TransportError` naming that
possibility); a separate connect timeout does not exist (``fetch`` has
one signal); there is no proxy configuration. The TypeScript port's
``docs/browser.md`` states the same line.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ._exceptions import TransportError
from ._types import AsyncTransportResponse, TransportRequest

try:  # the host bridge exists only under Pyodide
    import js as _js  # type: ignore[import-not-found]
    from pyodide.ffi import JsProxy, to_js  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised under CPython only by the import guard test
    _js = None
    JsProxy = None  # type: ignore[assignment]
    to_js = None  # type: ignore[assignment]

from ._limits import DEFAULT_READ_TIMEOUT, read_timeout_hint


def _require_pyodide() -> None:
    if _js is None:
        raise TransportError(
            "FetchTransport runs under Pyodide (a page, a worker, or Node hosting Pyodide); "
            "on CPython use StdlibAsyncTransport"
        )


class FetchTransport:
    """The :class:`AsyncTransport` over the host's ``fetch``.

    ``fetch`` may be given (a test double, a wrapped ``fetch``); by default
    it is the host's global. ``read_timeout`` bounds the wait for each body
    chunk, as the stdlib transport's does; a request's own
    ``read_timeout`` overrides it.
    """

    def __init__(self, *, fetch: Any = None, read_timeout: float = DEFAULT_READ_TIMEOUT) -> None:
        _require_pyodide()
        self._fetch = fetch if fetch is not None else _js.fetch
        self._read_timeout = read_timeout
        self._closed = False

    def stream(self, request: TransportRequest) -> "_FetchStreamCM":
        return _FetchStreamCM(self, request)

    async def aclose(self) -> None:
        self._closed = True

    async def _do_stream(self, request: TransportRequest) -> AsyncTransportResponse:
        if self._closed:
            raise TransportError("transport is closed")
        if request.connect_timeout is not None:
            raise TransportError("fetch has no separate connect timeout; leave connect_timeout unset")
        controller = _js.AbortController.new()
        headers = _js.Headers.new()
        for name, value in request.headers:
            headers.append(name, value)
        options: dict[str, Any] = {"method": request.method, "headers": headers, "signal": controller.signal}
        if request.body:
            options["body"] = to_js(request.body)
        try:
            response = await self._fetch(request.url, to_js(options, dict_converter=_js.Object.fromEntries))
        except Exception as exc:  # a JS TypeError: network, or the browser's CORS refusal — fetch does not say which
            raise TransportError(
                f"{request.method} {request.url.split('?', 1)[0]}: fetch failed ({_message(exc)}); "
                "a browser reports a CORS refusal and a network failure the same way"
            ) from exc

        header_pairs: list[tuple[str, str]] = []
        for pair in response.headers.entries():
            header_pairs.append((str(pair[0]), str(pair[1])))
        read_timeout = request.read_timeout or self._read_timeout
        reader = response.body.getReader() if response.body is not None else None
        state = {"consumed": False}

        async def chunks() -> AsyncIterator[bytes]:
            if reader is None:
                state["consumed"] = True
                return
            while True:
                try:
                    result = await asyncio.wait_for(reader.read(), read_timeout)
                except asyncio.TimeoutError as exc:
                    controller.abort()
                    raise TransportError(f"response body read timed out: {read_timeout_hint(read_timeout)}") from exc
                except Exception as exc:
                    raise TransportError(f"response body read failed ({_message(exc)})") from exc
                if result.done:
                    state["consumed"] = True
                    return
                yield bytes(result.value.to_py())

        async def release(body_consumed: bool) -> None:
            # Dropping the response before its end cancels the request: the
            # socket closes, the server sees it (the browser smoke checks this).
            if not (body_consumed or state["consumed"]):
                controller.abort()
                if reader is not None:
                    try:
                        await reader.cancel()
                    except Exception:
                        pass

        return AsyncTransportResponse(
            status=int(response.status),
            reason=str(response.statusText or ""),
            headers=header_pairs,
            http_version="HTTP/1.1",  # fetch does not expose the negotiated version
            chunks=chunks(),
            release=release,
        )


class _FetchStreamCM:
    def __init__(self, transport: FetchTransport, request: TransportRequest) -> None:
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


def _message(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
