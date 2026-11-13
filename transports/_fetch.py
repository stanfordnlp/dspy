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
only where the server's CORS headers allow it (a refusal can happen
AFTER the server served the request; ``fetch`` reports a bare
``TypeError`` with no status — surfaced as :class:`TransportError`
naming that possibility). Separate socket connect/write deadlines,
pool limits, and proxy configuration are controlled by the host, not us.
The TypeScript port's ``docs/browser.md`` describes these host limitations.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ._exceptions import ReadTimeout, TransportError
from ._http11 import content_codings
from ._timeouts import wait_for
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
    it is the host's global. ``read_timeout`` bounds the initial fetch
    (through response headers) and each body chunk, not the whole response.
    A request's explicit ``read_timeout`` overrides it. The host decodes
    compression; visible Content-Encoding is checked against INV-053, never
    inflated again. CORS-hidden headers cannot be checked.
    """

    def __init__(self, *, fetch: Any = None, read_timeout: float = DEFAULT_READ_TIMEOUT) -> None:
        _require_pyodide()
        self._fetch = fetch if fetch is not None else _js.fetch
        self._read_timeout = read_timeout
        self._closed = False
        self._controllers: dict[int, Any] = {}
        self._responses: dict[int, AsyncTransportResponse] = {}

    def stream(self, request: TransportRequest) -> "_FetchStreamCM":
        return _FetchStreamCM(self, request)

    async def aclose(self) -> None:
        self._closed = True
        for controller in tuple(self._controllers.values()):
            controller.abort()
        self._controllers.clear()
        for response in tuple(self._responses.values()):
            await response.aclose()

    async def _do_stream(self, request: TransportRequest) -> AsyncTransportResponse:
        if self._closed:
            raise TransportError("transport is closed")
        for name in ("connect_timeout", "write_timeout"):
            if getattr(request, name) is not None:
                raise TransportError(f"fetch has no separate {name}; leave {name} unset")
        read_timeout = self._read_timeout if request.read_timeout is None else request.read_timeout
        controller = _js.AbortController.new()
        self._controllers[id(controller)] = controller
        reader = None

        async def host_wait(awaitable):
            # Abort BEFORE draining cancellation. asyncio.wait distinguishes
            # caller cancellation even when the host promise resolves at once.
            task = asyncio.ensure_future(awaitable)
            try:
                done, _ = await asyncio.wait((task,), timeout=read_timeout)
                if not done:
                    raise asyncio.TimeoutError()
                return task.result()
            except BaseException:
                controller.abort()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise

        try:
            headers = _js.Headers.new()
            for name, value in request.headers:
                headers.append(name, value)
            if not any(name.lower() == "accept-encoding" for name, _ in request.headers):
                # Browsers may silently strip this forbidden request header.
                headers.append("Accept-Encoding", "identity")
            options: dict[str, Any] = {"method": request.method, "headers": headers, "signal": controller.signal}
            if request.body:
                options["body"] = to_js(request.body)
            try:
                response = await host_wait(self._fetch(request.url, to_js(options, dict_converter=_js.Object.fromEntries)))
            except asyncio.TimeoutError as exc:
                raise ReadTimeout(f"response headers timed out: {read_timeout_hint(read_timeout)}") from exc
            except Exception as exc:
                raise TransportError(
                    f"{request.method} {request.url.split('?', 1)[0]}: fetch failed ({_message(exc)}); "
                    "a browser reports a CORS refusal and a network failure the same way"
                ) from exc

            header_pairs = [(str(pair[0]), str(pair[1])) for pair in response.headers.entries()]
            # Fetch already decoded the body, even when it retains the coding
            # header. Enforce the allowed vocabulary only; do not double inflate.
            content_codings([v for k, v in header_pairs if k.lower() == "content-encoding"])
            status = int(response.status)
            reason = str(response.statusText or "")
            reader = response.body.getReader() if response.body is not None else None
        except BaseException:
            controller.abort()
            self._controllers.pop(id(controller), None)
            raise
        state = {"consumed": reader is None}

        async def chunks() -> AsyncIterator[bytes]:
            if reader is None:
                state["consumed"] = True
                return
            while True:
                try:
                    result = await host_wait(reader.read())
                except asyncio.TimeoutError as exc:
                    raise ReadTimeout(f"response body read timed out: {read_timeout_hint(read_timeout)}") from exc
                except Exception as exc:
                    raise TransportError(f"response body read failed ({_message(exc)})") from exc
                if result.done:
                    state["consumed"] = True
                    return
                yield bytes(result.value.to_py())

        async def release(body_consumed: bool) -> None:
            # Abort cancels host work, not a guarantee the server stops billing.
            # Release the reader lock on success too, and forget ownership once.
            try:
                if not (body_consumed or state["consumed"]):
                    controller.abort()
                    if reader is not None:
                        try:
                            await wait_for(reader.cancel(), read_timeout)
                        except Exception:
                            pass
            finally:
                self._controllers.pop(id(controller), None)
                self._responses.pop(id(controller), None)
                if reader is not None:
                    try:
                        reader.releaseLock()
                    except Exception:
                        pass

        result = AsyncTransportResponse(
            status=status,
            reason=reason,
            headers=header_pairs,
            http_version="HTTP/1.1",  # fetch does not expose the negotiated version
            chunks=chunks(),
            release=release,
        )
        self._responses[id(controller)] = result
        return result


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
