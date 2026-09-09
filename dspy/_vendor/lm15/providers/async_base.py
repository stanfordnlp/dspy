"""Async mirror provider LMs.

Each Async* class is a perfect mirror of its sync sibling: same constructor
fields, same canonical Request in, same canonical Response / StreamEvents
out — ``await`` is the only user-visible difference.

Design (see docs/design-rationale.md, "Async"): composition, not
inheritance.  Subclassing the sync adapter and overriding sync methods with
async ones would be a typing violation (``complete`` would no longer be
substitutable).  Instead each async class owns the async transport and
delegates ALL pure mapping — build_request, parse_response,
parse_stream_events, normalize_error, payload/header helpers — to an inner
instance of the sync adapter class, constructed with a transport that raises
if it is ever used: the inner adapter must never touch the network.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, ClassVar, Mapping, Protocol, TypeVar

from ..errors import (
    ProviderError,
    TransportError as LM15TransportError,
    UnsupportedFeatureError,
)
from ..features import EndpointSupport, ProviderManifest
from ..sse import aparse_sse
from ..transports import (
    AsyncTransportResponse,
    StdlibAsyncTransport,
    TransportRequest,
    TransportError as NetworkTransportError,
)
from ..types import (
    VideoGenerationRequest,
    SpeechGenerationRequest,
    BatchRequest,
    FileUploadRequest,
    ImageGenerationRequest,
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)
from .anthropic import AnthropicLM
from .base import BaseProviderLM, Credential, HttpResponse
from .claude_code import DEFAULT_CLAUDE_CODE_VERSION, ClaudeCodeLM
from .gemini import GeminiLM
from .openai import OpenAILM
from .openai_chat import OpenAIChatLM
from .openai_codex import (
    DEFAULT_CODEX_BASE_URL,
    DEFAULT_CODEX_CLIENT_VERSION,
    DEFAULT_CODEX_ORIGINATOR,
    OpenAICodexLM,
)
from .xai import DEFAULT_XAI_BASE_URL, XaiLM

_T = TypeVar("_T")


class AsyncTransport(Protocol):
    """Minimal async transport surface used by async provider LMs.

    ``stream`` returns an async context manager producing an
    :class:`AsyncTransportResponse` (StdlibAsyncTransport's shape).
    """

    def stream(self, request: TransportRequest) -> Any: ...


def default_async_transport() -> AsyncTransport:
    """Create the default async transport for standalone async provider LMs."""
    return StdlibAsyncTransport()


def _mirror_default(cls, name: str):
    """Default value of a sync sibling's dataclass field (slots-safe)."""
    return cls.__dataclass_fields__[name].default


class _ForbiddenTransport:
    """Transport for the inner sync adapter: pure mapping only, no I/O."""

    def stream(self, request: TransportRequest) -> Any:
        raise RuntimeError(
            "inner sync adapter of an Async* provider LM must never touch the "
            "network; all I/O goes through the async transport"
        )


class AsyncBaseProviderLM:
    """Shared asynchronous provider LM implementation.

    Mirrors :class:`BaseProviderLM` faithfully: build (delegated, pure) ->
    await async transport -> parse (delegated, pure); status>=400 raises the
    delegated normalize_error; transport errors are wrapped in
    lm15.TransportError.  Streaming applies MAP-3 via
    :func:`lm15.result.acoalesce_stream`.
    """

    transport: AsyncTransport
    _inner: BaseProviderLM  # set by subclass __post_init__

    # Mirrored metadata (subclasses override like their sync siblings).
    provider: str = "unknown"
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="unknown")

    @property
    def access(self) -> ProviderManifest:
        """The bound access policy — the inner sync adapter's."""
        return self._inner.access

    @property
    def supports(self) -> EndpointSupport:
        return self._inner.access.supports

    @classmethod
    def has_stored_credential(cls) -> bool:
        from ..access import has_stored_credential

        return has_stored_credential(cls.manifest)

    def _mirror_binding(self) -> None:
        """Copy what the inner adapter's access binding resolved: the
        provider string, the credential (a per-request provider for stored
        logins, repr-suppressed), and the base URL the policy chose."""
        self.provider = self._inner.provider
        self.api_key = self._inner.api_key
        self.base_url = self._inner.base_url

    async def _build(self, build: "Callable[..., _T]", *args: Any, **kwargs: Any) -> "_T":
        """Run one of the inner adapter's request builders.

        A builder invokes the credential provider (AUTH-2) inside ``_emit``.
        A cloud-chain provider may refresh over the network, synchronously;
        run it in a worker thread so that refresh does not block the event
        loop (D16, 2026-09-06).  A static credential needs no thread.
        """
        if callable(self._inner.api_key):
            return await asyncio.to_thread(build, *args, **kwargs)
        return build(*args, **kwargs)

    async def complete(self, request: Request) -> Response:
        req = await self._build(self._inner.build_request, request, stream=False)
        resp = await self._send(req)
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner.parse_response(request, resp)

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        # MAP-3 (docs/mapping-rules.md): adapters may emit one end event per
        # provider terminal frame; the coalescer merges them so the public
        # stream yields exactly one final StreamEndEvent.
        from ..result import acoalesce_stream

        return acoalesce_stream(self._stream_raw(request), model=request.model)

    async def _stream_raw(self, request: Request) -> AsyncIterator[StreamEvent]:
        req = await self._build(self._inner.build_request, request, stream=True)
        try:
            async with self.transport.stream(req) as resp:
                if resp.status >= 400:
                    body = await resp.read()
                    raise self._inner.normalize_error(
                        resp.status, body.decode("utf-8", errors="replace")
                    )
                async for raw in aparse_sse(_aiter_lines(resp)):
                    for event in self._inner.parse_stream_events(request, raw):
                        if event is not None:
                            yield event
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    async def _send(self, request: TransportRequest) -> HttpResponse:
        try:
            async with self.transport.stream(request) as resp:
                body = await resp.read()
                return HttpResponse(
                    status=resp.status,
                    reason=resp.reason,
                    headers=resp.headers,
                    body=body,
                    http_version=resp.http_version,
                )
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    def normalize_error(self, status: int, body: str) -> ProviderError:
        return self._inner.normalize_error(status, body)

    async def list_models(self):
        """Async mirror of BaseProviderLM.list_models (canonical ModelInfo)."""
        resp = await self._send(await self._build(self._inner._models_request))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._models_from_body(resp.text())

    # ── Batch: async drivers over the sync adapter's pure hooks ──────

    async def batch_submit(self, request: "BatchRequest"):
        upload_body = None
        upload_req = await self._build(self._inner._batch_upload_request, request)
        if upload_req is not None:
            resp = await self._send(upload_req)
            if resp.status >= 400:
                raise self._inner.normalize_error(resp.status, resp.text())
            upload_body = resp.json()
        resp = await self._send(await self._build(self._inner._batch_submit_request, request, upload_body))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._batch_job_from_body(resp.text())

    async def batch_status(self, batch_id: str):
        resp = await self._send(await self._build(self._inner._batch_status_request, batch_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._batch_job_from_body(resp.text())

    async def batch_results(self, batch_id: str):
        from ..types import BATCH_TERMINAL_STATUSES

        resp = await self._send(await self._build(self._inner._batch_status_request, batch_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        job = self._inner._batch_job_from_body(resp.text())
        if job.status not in BATCH_TERMINAL_STATUSES:
            raise ValueError(
                f"batch {batch_id} is not finished (status={job.status!r}); "
                f"await wait() or poll batch_status() until done"
            )
        status_body = resp.json()
        texts = []
        for fetch in await self._build(self._inner._batch_result_fetches, status_body):
            fetched = await self._send(fetch)
            if fetched.status >= 400:
                raise self._inner.normalize_error(fetched.status, fetched.text())
            texts.append(fetched.text())
        return self._inner._batch_entries(status_body, tuple(texts))

    async def batch_cancel(self, batch_id: str):
        resp = await self._send(await self._build(self._inner._batch_cancel_request, batch_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._batch_job_from_body(resp.text())

    async def batch_list(self, limit: int = 20):
        resp = await self._send(await self._build(self._inner._batch_list_request, limit))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._batch_jobs_from_list_body(resp.text())

    async def batch(self, requests, *, model: str | None = None, label: str | None = None,
                    extensions=None):
        """Third execution mode, async twin: returns an AsyncBatchJob."""
        from ..batch import AsyncBatchJob

        if isinstance(requests, BatchRequest):
            request = requests
        else:
            request = BatchRequest(model=model, requests=tuple(requests), label=label, extensions=extensions)
        return AsyncBatchJob(self, await self.batch_submit(request))

    async def batch_job(self, batch_id: str):
        from ..batch import AsyncBatchJob

        return AsyncBatchJob(self, await self.batch_status(batch_id))

    async def batches(self, limit: int = 20):
        from ..batch import AsyncBatchJob

        return tuple(AsyncBatchJob(self, info) for info in await self.batch_list(limit))

    # ── Cache resources: async drivers over the sync adapter's pure hooks ──

    async def cache_create(self, prefix: Request, *, ttl_seconds: int | None = None, label: str | None = None):
        self._inner._check_cache_prefix(prefix, ttl_seconds)
        resp = await self._send(await self._build(self._inner._cache_create_request, prefix, ttl_seconds, label))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._cache_info_from_body(resp.text())

    async def cache_get(self, cache_id: str):
        resp = await self._send(await self._build(self._inner._cache_get_request, cache_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._cache_info_from_body(resp.text())

    async def cache_list(self, limit: int = 20, cursor: str | None = None):
        resp = await self._send(await self._build(self._inner._cache_list_request, limit, cursor))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._cache_page_from_list_body(resp.text())

    async def cache_delete(self, cache_id: str) -> None:
        resp = await self._send(await self._build(self._inner._cache_delete_request, cache_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())

    async def cache_update(self, cache_id: str, *, ttl_seconds: int):
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive int")
        resp = await self._send(await self._build(self._inner._cache_update_request, cache_id, ttl_seconds))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._cache_info_from_body(resp.text())

    async def cache(self, prefix: Request, *, ttl_seconds: int | None = None, label: str | None = None):
        from ..types import CachedPrefix

        if self.supports.caches:
            return CachedPrefix(prefix, await self.cache_create(prefix, ttl_seconds=ttl_seconds, label=label))
        self._inner._check_cache_prefix(prefix, ttl_seconds)
        return CachedPrefix(prefix)

    # ── Files: async drivers over the sync adapter's pure hooks ──────

    async def file_upload(self, request: "FileUploadRequest"):
        resp = await self._send(await self._build(self._inner._file_upload_request, request))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._file_info_from_body(resp.text())

    async def file_get(self, file_id: str):
        resp = await self._send(await self._build(self._inner._file_get_request, file_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._file_info_from_body(resp.text())

    async def file_list(self, limit: int = 20, cursor: str | None = None):
        resp = await self._send(await self._build(self._inner._file_list_request, limit, cursor))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._file_page_from_list_body(resp.text())

    async def file_delete(self, file_id: str) -> None:
        resp = await self._send(await self._build(self._inner._file_delete_request, file_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())

    async def file_download(self, file_id: str) -> bytes:
        resp = await self._send(await self._build(self._inner._file_download_request, file_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return resp.body

    async def file_wait_ready(self, file_id: str, poll_every: float = 2.0, timeout: float | None = None):
        import asyncio
        import time as _time

        deadline = None if timeout is None else _time.monotonic() + timeout
        info = await self.file_get(file_id)
        while info.readiness == "pending":
            if deadline is not None and _time.monotonic() >= deadline:
                raise TimeoutError(f"file {file_id} still pending after {timeout}s")
            await asyncio.sleep(poll_every)
            info = await self.file_get(file_id)
        return info

    async def aclose(self) -> None:
        aclose = getattr(self.transport, "aclose", None)
        if callable(aclose):
            await aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # ── Non-chat endpoints: sync-only for now (honest surface) ──────

    def _async_unsupported(self, endpoint: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(
            f"{self.provider}: {endpoint}: use the sync adapter for this "
            "endpoint (async endpoints planned)",
            provider=self.provider,
        )

    def live(self, config: LiveConfig):
        raise self._async_unsupported("live")

    # ── Video generation: async drivers over the sync adapter's pure hooks ──

    async def video_submit(self, request: "VideoGenerationRequest"):
        resp = await self._send(await self._build(self._inner._video_submit_request, request))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._video_job_from_body(resp.text())

    async def video_status(self, video_id: str):
        resp = await self._send(await self._build(self._inner._video_status_request, video_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._video_job_from_body(resp.text(), video_id)

    async def video_result(self, video_id: str):
        from ..types import VIDEO_TERMINAL_STATUSES

        resp = await self._send(await self._build(self._inner._video_status_request, video_id))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        job = self._inner._video_job_from_body(resp.text(), video_id)
        if job.status not in VIDEO_TERMINAL_STATUSES:
            raise ValueError(
                f"video {video_id} is not finished (status={job.status!r}); "
                f"wait() or poll video_status() until done"
            )
        status_body = resp.json()
        fetch = await self._build(self._inner._video_result_fetch, status_body)
        fetched = None
        if fetch is not None:
            fetched = await self._send(fetch)
            if fetched.status >= 400:
                raise self._inner.normalize_error(fetched.status, fetched.text())
        return self._inner._video_part(status_body, fetched)

    async def video_list(self, limit: int = 20, model: str | None = None):
        resp = await self._send(await self._build(self._inner._video_list_request, limit, model))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._video_jobs_from_list_body(resp.text())

    async def video_generate(self, request: "VideoGenerationRequest"):
        from ..video_jobs import AsyncVideoJob

        return AsyncVideoJob(self, await self.video_submit(request))

    async def video_job(self, video_id: str):
        from ..video_jobs import AsyncVideoJob

        return AsyncVideoJob(self, await self.video_status(video_id))

    async def video_jobs(self, limit: int = 20, model: str | None = None):
        from ..video_jobs import AsyncVideoJob

        return tuple(AsyncVideoJob(self, info) for info in await self.video_list(limit, model))

    # ── Media generation: async drivers over the sync adapter's pure hooks ──

    async def image_generate(self, request: ImageGenerationRequest):
        resp = await self._send(await self._build(self._inner._image_generate_request, request))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._image_generation_from_response(request, resp)

    async def speech_generate(self, request: SpeechGenerationRequest):
        resp = await self._send(await self._build(self._inner._speech_generate_request, request))
        if resp.status >= 400:
            raise self._inner.normalize_error(resp.status, resp.text())
        return self._inner._speech_generation_from_response(request, resp)


async def _aiter_lines(resp: AsyncTransportResponse) -> AsyncIterator[bytes]:
    aiter_lines = getattr(resp, "aiter_lines", None)
    if aiter_lines is not None:
        async for line in aiter_lines():
            yield line
        return
    buf = bytearray()
    async for chunk in resp:
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\n")
            if idx < 0:
                break
            yield bytes(buf[: idx + 1])
            del buf[: idx + 1]
    if buf:
        yield bytes(buf)


# ─── Mirror classes ──────────────────────────────────────────────────


@dataclass(slots=True)
class AsyncOpenAILM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.openai.com/v1"
    profile: Any | None = None
    compat: Any | None = field(default=None, kw_only=True)
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = field(default=None, kw_only=True)
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False, kw_only=True)
    account_id: str | None = None

    async def live(self, config: LiveConfig):
        self._inner._require("live")
        # Native async websocket (websockets.asyncio) — not a thread
        # wrapper: a blocked sync recv in a worker thread cannot be
        # cancelled from the event loop, and cancellation is the heart
        # of realtime. Codecs are the inner adapter's pure functions.
        import json as _json

        from ..live import AsyncWebSocketLiveSession, require_websocket_async_connect

        connect = require_websocket_async_connect()
        inner = self._inner
        ws = await connect(inner._live_url(config.model), additional_headers=inner._live_headers())
        for frame in inner._live_setup_frames(config):
            await ws.send(_json.dumps(frame))
        return AsyncWebSocketLiveSession(
            ws=ws,
            encode_event=inner._live_encoder(config),
            decode_event=inner._decode_live_server_event,
        )

    provider: str = field(default="openai", init=False)
    manifest: ClassVar[ProviderManifest] = OpenAILM.manifest

    _inner: OpenAILM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAILM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            profile=self.profile,
            compat=self.compat,
            access=self.access,
            credentials_path=self.credentials_path,
            settings=self.settings,
            clock=self.clock,
            account_id=self.account_id,
        )
        self._mirror_binding()
        # The sync sibling resolves a preset name (and may supply that
        # server's default base_url); mirror the resolved values.
        self.base_url = self._inner.base_url


@dataclass(slots=True)
class AsyncAnthropicLM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    compat: Any | None = None
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="anthropic", init=False)
    manifest: ClassVar[ProviderManifest] = AnthropicLM.manifest

    _inner: AnthropicLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = AnthropicLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            api_version=self.api_version,
            compat=self.compat,
            access=self.access,
            credentials_path=self.credentials_path,
            settings=self.settings,
            clock=self.clock,
        )
        self._mirror_binding()
        # The sync sibling resolves compat presets (and that server's default
        # base_url); mirror the resolved values, as AsyncOpenAIChatLM does.
        self.base_url = self._inner.base_url
        self.compat = self._inner.compat


@dataclass(slots=True)
class AsyncGeminiLM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="gemini", init=False)
    manifest: ClassVar[ProviderManifest] = GeminiLM.manifest

    _inner: GeminiLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = GeminiLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            upload_base_url=self.upload_base_url,
            access=self.access,
            credentials_path=self.credentials_path,
            settings=self.settings,
            clock=self.clock,
        )
        self._mirror_binding()

    async def live(self, config: LiveConfig):
        self._inner._require("live")
        # Native async twin of GeminiLM.live: same pure setup frame,
        # encoder, and setup-status classifier; only the socket awaits.
        import json as _json

        from ..live import AsyncWebSocketLiveSession, require_websocket_async_connect

        connect = require_websocket_async_connect()
        inner = self._inner
        ws = await connect(inner._live_url())
        for frame in inner._live_setup_frames(config):
            await ws.send(_json.dumps(frame))
        while not inner._live_setup_status(await ws.recv()):
            pass
        return AsyncWebSocketLiveSession(
            ws=ws,
            encode_event=inner._live_encoder(config),
            decode_event=inner._decode_live_server_event,
        )


@dataclass(slots=True)
class AsyncOpenAIChatLM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = _mirror_default(OpenAIChatLM, "base_url")
    compat: Any | None = None
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="openai_chat", init=False)
    manifest: ClassVar[ProviderManifest] = OpenAIChatLM.manifest

    _inner: OpenAIChatLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAIChatLM(
            api_key=self.api_key,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            compat=self.compat,
            access=self.access,
            credentials_path=self.credentials_path,
            settings=self.settings,
            clock=self.clock,
        )
        self._mirror_binding()
        # The sync sibling's __post_init__ resolves compat presets (and may
        # supply that server's default base_url); mirror the resolved values.
        self.base_url = self._inner.base_url
        self.compat = self._inner.compat


# ─── Subscription mirrors (Claude Code / Codex CLI OAuth) ────────────
#
# Same composition pattern: the inner sync adapter validates the local OAuth
# credential at construction time (typed, re-login-guided errors), then
# re-resolves it per request so long-lived clients stay fresh.  Per-request
# resolution is a local file read; a refresh, when the token has expired, is
# one blocking token-endpoint call inside the sync mapping layer.  Credential
# fields are repr-suppressed so secrets never leak.


@dataclass(slots=True)
class AsyncClaudeCodeLM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = None
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = "https://api.anthropic.com/v1"
    api_version: str = "2023-06-01"
    claude_code_version: str = DEFAULT_CLAUDE_CODE_VERSION

    # Not constructor params on the sync sibling either (it is not a dataclass).
    provider: str = field(default="claude-code", init=False)
    manifest: ClassVar[ProviderManifest] = ClaudeCodeLM.manifest

    _inner: ClaudeCodeLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = ClaudeCodeLM(
            api_key=self.api_key,
            credentials_path=self.credentials_path,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            api_version=self.api_version,
            claude_code_version=self.claude_code_version,
        )
        self.api_key = self._inner.api_key  # static key or per-request credential provider (repr-suppressed)

    # Files are an API-key surface; the subscription credential does not
    # carry them. Block every inherited async driver, not just upload.
    def file_upload(self, request: FileUploadRequest):
        return self._inner.file_upload(request)  # raises UnsupportedFeatureError

    def file_get(self, file_id: str):
        return self._inner.file_get(file_id)  # raises UnsupportedFeatureError

    def file_list(self, limit: int = 20, cursor: str | None = None):
        return self._inner.file_list(limit, cursor)  # raises UnsupportedFeatureError

    def file_delete(self, file_id: str):
        return self._inner.file_delete(file_id)  # raises UnsupportedFeatureError

    def file_download(self, file_id: str):
        return self._inner.file_download(file_id)  # raises UnsupportedFeatureError

    # Batch is an API-key surface; the subscription credential does not
    # carry it. Block every inherited async driver, not just submit.
    def batch_submit(self, request: BatchRequest):
        return self._inner.batch_submit(request)  # raises UnsupportedFeatureError

    def batch_status(self, batch_id: str):
        return self._inner.batch_status(batch_id)  # raises UnsupportedFeatureError

    def batch_results(self, batch_id: str):
        return self._inner.batch_results(batch_id)  # raises UnsupportedFeatureError

    def batch_cancel(self, batch_id: str):
        return self._inner.batch_cancel(batch_id)  # raises UnsupportedFeatureError

    def batch_list(self, limit: int = 20):
        return self._inner.batch_list(limit)  # raises UnsupportedFeatureError

    def live(self, config: LiveConfig):
        return self._inner.live(config)  # raises UnsupportedFeatureError


@dataclass(slots=True)
class AsyncOpenAICodexLM(AsyncBaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    account_id: str | None = None
    auth_path: "str | os.PathLike[str] | None" = None
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = DEFAULT_CODEX_BASE_URL
    originator: str = DEFAULT_CODEX_ORIGINATOR
    client_version: str = DEFAULT_CODEX_CLIENT_VERSION

    # Not constructor params on the sync sibling either (it is not a dataclass).
    provider: str = field(default="openai-codex", init=False)
    manifest: ClassVar[ProviderManifest] = OpenAICodexLM.manifest

    _inner: OpenAICodexLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = OpenAICodexLM(
            api_key=self.api_key,
            account_id=self.account_id,
            auth_path=self.auth_path,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
            originator=self.originator,
            client_version=self.client_version,
        )
        self.api_key = self._inner.api_key  # static key or per-request credential provider (repr-suppressed)
        self.account_id = self._inner.account_id

    async def complete(self, request: Request) -> Response:
        # Mirror of OpenAICodexLM.complete: the Codex subscription backend is
        # streaming-first; materialize the (coalesced) stream.
        from ..result import amaterialize_response

        return await amaterialize_response(self.stream(request), request)

    def live(self, config: LiveConfig):
        return self._inner.live(config)  # raises UnsupportedFeatureError

    # Files are an API-key surface; the subscription credential does not
    # carry them. Block every inherited async driver, not just upload.
    def file_upload(self, request: FileUploadRequest):
        return self._inner.file_upload(request)  # raises UnsupportedFeatureError

    def file_get(self, file_id: str):
        return self._inner.file_get(file_id)  # raises UnsupportedFeatureError

    def file_list(self, limit: int = 20, cursor: str | None = None):
        return self._inner.file_list(limit, cursor)  # raises UnsupportedFeatureError

    def file_delete(self, file_id: str):
        return self._inner.file_delete(file_id)  # raises UnsupportedFeatureError

    def file_download(self, file_id: str):
        return self._inner.file_download(file_id)  # raises UnsupportedFeatureError

    # Batch is an API-key surface; the subscription credential does not
    # carry it. Block every inherited async driver, not just submit.
    def batch_submit(self, request: BatchRequest):
        return self._inner.batch_submit(request)  # raises UnsupportedFeatureError

    def batch_status(self, batch_id: str):
        return self._inner.batch_status(batch_id)  # raises UnsupportedFeatureError

    def batch_results(self, batch_id: str):
        return self._inner.batch_results(batch_id)  # raises UnsupportedFeatureError

    def batch_cancel(self, batch_id: str):
        return self._inner.batch_cancel(batch_id)  # raises UnsupportedFeatureError

    def batch_list(self, limit: int = 20):
        return self._inner.batch_list(limit)  # raises UnsupportedFeatureError

    def image_generate(self, request: ImageGenerationRequest):
        return self._inner.image_generate(request)  # raises UnsupportedFeatureError

    def speech_generate(self, request: SpeechGenerationRequest):
        return self._inner.speech_generate(request)  # raises UnsupportedFeatureError


@dataclass(slots=True)
class AsyncXaiLM(AsyncBaseProviderLM):
    """Async mirror of :class:`XaiLM` (subscription OAuth or bearer key)."""

    api_key: Credential | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = None
    transport: AsyncTransport = field(default_factory=default_async_transport)
    base_url: str = DEFAULT_XAI_BASE_URL

    # Not constructor params on the sync sibling either.
    provider: str = field(default="xai", init=False)
    manifest: ClassVar[ProviderManifest] = XaiLM.manifest
    # Same offline probe as the sync sibling (oauth-unless-explicit policy).
    has_stored_credential: ClassVar = XaiLM.has_stored_credential

    _inner: XaiLM = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._inner = XaiLM(
            api_key=self.api_key,
            credentials_path=self.credentials_path,
            transport=_ForbiddenTransport(),
            base_url=self.base_url,
        )
        self.api_key = self._inner.api_key  # static key or per-request credential provider (repr-suppressed)


__all__ = [
    "AsyncBaseProviderLM",
    "AsyncTransport",
    "AsyncOpenAILM",
    "AsyncAnthropicLM",
    "AsyncGeminiLM",
    "AsyncOpenAIChatLM",
    "AsyncClaudeCodeLM",
    "AsyncOpenAICodexLM",
    "AsyncXaiLM",
    "default_async_transport",
]
