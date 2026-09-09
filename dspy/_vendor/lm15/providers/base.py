from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Mapping, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for annotations
    from ..batch import BatchJob
    from ..video_jobs import VideoJob

from ..errors import (
    AuthError,
    NotConfiguredError,
    ProviderError,
    TransportError as LM15TransportError,
    UnsupportedFeatureError,
    map_http_error,
    with_credential_hint,
)
from ..credentials import AwsCredentials, CredentialLike, CredentialValue, coerce_credential
from ..features import EndpointSupport, ProviderManifest
from ..models import ModelInfo
from ..protocols import LiveSession
from ..sse import SSEEvent, parse_sse
from ..transports import TransportRequest
from ..transports import TransportResponse
from ..transports import StdlibTransport
from ..transports import TransportError as NetworkTransportError
from ..types import (
    CachedPrefix,
    CacheInfo,
    CachePage,
    Config,
    VIDEO_TERMINAL_STATUSES,
    VideoGenerationRequest,
    VideoJobInfo,
    VideoPart,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    BATCH_TERMINAL_STATUSES,
    BatchEntry,
    BatchJobInfo,
    BatchRequest,
    FileInfo,
    FilePage,
    FileUploadRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)


class SyncTransport(Protocol):
    """Minimal sync transport surface used by provider LMs."""

    def stream(self, request: TransportRequest) -> TransportResponse: ...


def default_transport() -> SyncTransport:
    """Create the default sync transport for standalone provider LMs."""
    return StdlibTransport()


# A credential is a static key string, an AUTH-2 credential value (ApiKey,
# BearerToken, AwsCredentials — lm15.credentials), or a zero-argument
# provider callable returning either.  Providers are invoked at
# request-build time, once per request, so rotating credentials (OAuth
# refresh, cloud chains) stay fresh in long-lived clients.  Caching and
# refreshing belongs to the provider callable (AUTH-2); the adapter never
# caches what it returns.
Credential = CredentialLike


def resolve_credential_value(credential: Credential) -> CredentialValue:
    """Invoke a provider callable if needed and coerce to a credential value."""
    raw = credential() if callable(credential) else credential
    return coerce_credential(raw)


def resolve_credential(credential: Credential) -> str:
    """The bearer/key STRING of a credential — the pre-2026-09-03 shape the
    dialects' header code reads.  AWS credentials have no single string;
    they travel as a SigV4 signature (``_emit``), never through here."""
    value = resolve_credential_value(credential)
    if isinstance(value, AwsCredentials):
        raise NotConfiguredError("AWS credentials cannot be sent as a header value; the door must accept sigv4")
    return value.value


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse an HTTP Retry-After header: delta-seconds or an HTTP-date."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return seconds if seconds >= 0 else None
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    try:
        when = parsedate_to_datetime(value)
    except Exception:
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _attach_retry_after(error: ProviderError, headers: "list[tuple[str, str]] | None") -> None:
    """Populate error.retry_after from a Retry-After header.

    A provider-body-derived value (set by the adapter's normalize_error)
    always wins; the header only fills the gap."""
    if getattr(error, "retry_after", None) is not None:
        return
    value = None
    for key, val in headers or []:
        if key.lower() == "retry-after":
            value = val
            break
    seconds = _retry_after_seconds(value)
    if seconds is not None:
        error.retry_after = seconds


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Buffered provider-level HTTP response.

    The stdlib transport is streaming-first.  LMs that implement ordinary
    request/response endpoints buffer the body into this small value object so
    their parsing code can stay pure and easy to test.
    """

    status: int
    reason: str
    headers: list[tuple[str, str]]
    body: bytes
    http_version: str = "HTTP/1.1"

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for key, value in self.headers:
            if key.lower() == lname:
                return value
        return None

    def headers_all(self, name: str) -> list[str]:
        lname = name.lower()
        return [value for key, value in self.headers if key.lower() == lname]

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.body)


class ProviderDialect(Protocol):
    provider: str
    supports: EndpointSupport
    manifest: ProviderManifest

    def build_request(self, request: Request, stream: bool) -> TransportRequest: ...

    def parse_response(self, request: Request, response: HttpResponse) -> Response: ...

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]: ...

    def normalize_error(self, status: int, body: str) -> ProviderError: ...


_SURFACE_WORD = {
    "files": "files",
    "batches": "batch",
    "images": "image generation",
    "speech": "speech generation",
    "video": "video generation",
    "live": "live",
    "caches": "caches",
    "models": "model listing",
}


class BaseProviderLM:
    """Shared synchronous provider LM implementation.

    An adapter is a dialect (the class) bound to an access policy (a value,
    ``lm15.access``): the policy says which credential travels in which
    header, which static headers ride along, which endpoint surfaces this
    access path carries, and which backend variant the dialect must switch
    on. ``manifest`` is the class's default policy; ``access`` is the bound
    one. Dialect code reads ``self.access`` at a small set of stated points
    and never carries a subclass per access path.
    """

    transport: SyncTransport
    provider: str = "unknown"
    manifest: ClassVar[ProviderManifest] = ProviderManifest(provider="unknown")
    access: ProviderManifest
    account_id: str | None = None
    _credential_source: str = "explicit"
    # Cloud-host state (AUTH-10): the resolved host settings and the clock
    # every time-dependent byte (SigV4 date, JWT iat/exp) is read from.  The
    # harness injects a fixed clock; users never touch it.
    host_settings: "dict[str, str]"
    clock: "Callable[[], datetime] | None"
    # The header an ApiKey travels under when the policy says x-api-key
    # (Anthropic ``x-api-key``, Gemini ``x-goog-api-key``).
    _api_key_header: ClassVar[str] = "x-api-key"

    @property
    def supports(self) -> EndpointSupport:
        """The endpoint surfaces this bound access path carries."""
        return self.access.supports

    @classmethod
    def has_stored_credential(cls) -> bool:
        """Offline probe for the router's ``oauth-unless-explicit`` chain
        (spec/auth.md AUTH-1): is a usable login stored locally?"""
        from ..access import has_stored_credential

        return has_stored_credential(cls.manifest)

    def _bind_access(
        self,
        access: ProviderManifest | None,
        *,
        credentials_path: "str | os.PathLike[str] | None" = None,
        default_base_url: str | None = None,
        settings: "Mapping[str, str] | None" = None,
    ) -> None:
        """Bind the access policy and resolve the credential it calls for.

        Called from each dialect's ``__post_init__``. Sets ``access``,
        ``provider``, ``api_key`` (a callable for stored logins, re-read per
        request so rotations and refreshes are picked up), ``account_id``
        when the credential carries one, ``host_settings`` (explicit values
        and defaults only — the router fills env fallbacks, like it does for
        the key), and ``base_url``: the policy's own, or the host template
        rendered over the settings, when the caller left the dialect's
        default.
        """
        from ..access import load_credential
        from ..cloud.hosts import render_base_url, resolve_settings

        policy = access if access is not None else type(self).manifest
        self.access = policy
        self.provider = policy.provider
        loaded = load_credential(policy, self.api_key, credentials_path=credentials_path)
        self.api_key = loaded.credential
        self._credential_source = loaded.source
        if loaded.account_id is not None and self.account_id is None:
            self.account_id = loaded.account_id
        if loaded.credential is not None and not callable(loaded.credential):
            # A static credential of the wrong kind for this door fails now,
            # not on the first request (a provider callable is checked per call).
            from ..access import select_scheme

            select_scheme(policy, coerce_credential(loaded.credential))
        self.host_settings = resolve_settings(policy.host, settings, None, provider=policy.provider)
        if getattr(self, "clock", None) is None:
            self.clock = None
        if policy.host is not None:
            if default_base_url is None or self.base_url == default_base_url:
                self.base_url = render_base_url(policy.host, self.host_settings)
        elif policy.base_url is not None and default_base_url is not None and self.base_url == default_base_url:
            self.base_url = policy.base_url

    def _registry_compat(self) -> str | None:
        """The compat preset the bound provider names in the registry, for a
        dialect constructed directly with a bound access policy and no
        ``compat=``: the policy names the provider; the provider names its
        preset (found 2026-09-03 when a direct ``OpenAIChatLM(access=
        BEDROCK_CHAT)`` skipped the bedrock per-model refusals)."""
        if self.access is type(self).manifest:
            return None
        from ..registry import lookup

        definition = lookup(self.access.provider)
        return definition.compat if definition is not None else None

    def _now(self) -> "datetime":
        from ..cloud.hosts import utc_now

        return self.clock() if self.clock is not None else utc_now()

    def _emit(
        self,
        *,
        method: str,
        url: str,
        headers: "dict[str, str] | list[tuple[str, str]] | None" = None,
        params: "dict[str, Any] | None" = None,
        payload: Any = None,
        body: bytes | None = None,
        endpoint: str | None = None,
        stream: bool = False,
        model: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        write_timeout: float | None = None,
    ) -> TransportRequest:
        """Finish a dialect-built request through the bound host (AUTH-10)
        and sign it.  With no host this is ``make_json_request`` — every
        dialect routes its wire requests through here so a host never needs
        a dialect branch."""
        from ..cloud.hosts import finish_request, sign_request
        from .common import make_json_request

        from ..access import auth_header

        credential = resolve_credential_value(self.api_key) if self.api_key is not None else None
        hdrs = dict(headers.items()) if isinstance(headers, dict) else dict(headers or [])
        if credential is not None and not isinstance(credential, AwsCredentials):
            pair = auth_header(self.access, credential, api_key_header=self._api_key_header)
            if pair is not None and pair[0].lower() not in {k.lower() for k in hdrs}:
                hdrs[pair[0]] = pair[1]
        finished = finish_request(
            self.access,
            self.host_settings,
            base_url=self.base_url,
            url=url,
            headers=hdrs,
            payload=payload,
            params=params,
            endpoint=endpoint,
            stream=stream,
            model=model,
            credential=credential,
        )
        req = make_json_request(
            method=method,
            url=finished.url,
            headers=finished.headers,
            params=finished.params or None,
            payload=finished.payload,
            body=body,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
        )
        if isinstance(credential, AwsCredentials):
            req.headers = sign_request(
                self.access,
                self.host_settings,
                method=req.method,
                url=req.url,
                headers=req.headers,
                body=req.body,
                credential=credential,
                now=self._now(),
            )
        return req

    def _require(self, surface: str) -> None:
        """Raise unless the bound access path carries ``surface``.

        A dialect may implement a surface its access path does not carry
        (Anthropic files on a Claude Code login); the policy, not the class,
        decides. Message shape: ``<provider>: <surface> not supported``.
        """
        if not self.access.supports.supports_endpoint(surface):
            raise UnsupportedFeatureError(
                f"{self.provider}: {_SURFACE_WORD.get(surface, surface)} not supported",
                provider=self.provider,
            )

    def complete(self, request: Request) -> Response:
        req = self.build_request(request, stream=False)
        resp = self._send(req)
        if resp.status >= 400:
            error = self.normalize_error(resp.status, resp.text())
            _attach_retry_after(error, resp.headers)
            raise error
        return self.parse_response(request, resp)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        # MAP-3 (docs/mapping-rules.md): adapters may emit one end event per
        # provider terminal frame; the coalescer merges them so the public
        # stream yields exactly one final StreamEndEvent.
        from ..result import coalesce_stream

        return coalesce_stream(self._stream_raw(request), model=request.model)

    def _stream_raw(self, request: Request) -> Iterator[StreamEvent]:
        req = self.build_request(request, stream=True)
        self._ensure_transport_open()
        try:
            with self.transport.stream(req) as resp:
                if resp.status >= 400:
                    body = resp.read()
                    error = self.normalize_error(
                        resp.status, body.decode("utf-8", errors="replace")
                    )
                    _attach_retry_after(error, resp.headers)
                    raise error
                lines = resp.iter_lines() if hasattr(resp, "iter_lines") else _iter_lines(resp)
                for raw in parse_sse(lines):
                    for event in self.parse_stream_events(request, raw):
                        if event is not None:
                            yield event
        except NetworkTransportError as exc:
            raise LM15TransportError(str(exc)) from exc

    def _send(self, request: TransportRequest) -> HttpResponse:
        self._ensure_transport_open()
        try:
            with self.transport.stream(request) as resp:
                body = resp.read()
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
        return self._with_login_hint(map_http_error(
            status,
            body.strip()[:500] or f"HTTP {status}",
            provider=self.provider,
            env_keys=self.access.env_keys,
        ))

    def _provider_error(
        self,
        cls: type[ProviderError],
        message: str,
        *,
        status: int | None = None,
        provider_code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> ProviderError:
        kwargs = {
            "provider": self.provider,
            "provider_code": provider_code or None,
            "status": status,
            "request_id": request_id or None,
            "retry_after": retry_after,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if issubclass(cls, AuthError):
            return self._with_login_hint(cls(message, env_keys=self.access.env_keys, **kwargs))
        return cls(message, **kwargs)

    def _with_login_hint(self, error: ProviderError) -> ProviderError:
        """Auth errors guide the user to re-login when the credential is a
        local login: always under an ``oauth`` policy (there is no env var),
        and under ``oauth-unless-explicit`` only when the stored login was
        the rung that won — an explicit key keeps the generic guidance.
        ``with_credential_hint`` is a no-op on non-auth errors."""
        hint = self.access.login_hint
        if hint and (self.access.credential_policy == "oauth" or self._credential_source == "stored"):
            return with_credential_hint(error, hint)
        return error

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    def _ensure_transport_open(self) -> None:
        """Recreate the default transport if it was closed by interactive tooling.

        Provider objects are often kept in notebook/REPL variables.  Some
        interactive runners eagerly close context-manager-like objects between
        cells; when that happens, the default StdlibTransport can be safely
        replaced before the next request.
        """
        if isinstance(self.transport, StdlibTransport) and getattr(self.transport, "_closed", False):
            self.transport = default_transport()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def live(self, config: LiveConfig) -> LiveSession:
        self._require("live")
        raise UnsupportedFeatureError(f"{self.provider}: live not supported", provider=self.provider)

    # ─── Media generation (images / speech) ────────────────────────
    #
    # Same pure-hook pattern as files and batch: adapters implement the
    # build/parse pair; the sync drivers below and the async mirrors both
    # drive the same hooks.  Parse hooks take the full HttpResponse, not
    # just the body: OpenAI speech returns raw bytes whose media type
    # lives in the content-type HEADER (captured 2026-09-01).  Media
    # types always come from the wire — hardcoding one is a lie on at
    # least one provider (OpenAI png / Gemini png / xAI jpeg captured).

    def _generation_unsupported(self, kind: str) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: {kind} generation not supported", provider=self.provider)

    def _image_generate_request(self, request: ImageGenerationRequest) -> TransportRequest:
        raise self._generation_unsupported("image")

    def _image_generation_from_response(self, request: ImageGenerationRequest, resp: HttpResponse) -> ImageGenerationResponse:
        raise self._generation_unsupported("image")

    def _speech_generate_request(self, request: SpeechGenerationRequest) -> TransportRequest:
        raise self._generation_unsupported("speech")

    def _speech_generation_from_response(self, request: SpeechGenerationRequest, resp: HttpResponse) -> SpeechGenerationResponse:
        raise self._generation_unsupported("speech")

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate (or, with ``request.images``, edit) images.

        Input images are ordinary ImageParts; adapters route them to the
        provider's real edit door (OpenAI: ``/images/edits``; Gemini: the
        same chat call; xAI: ``/images/edits``) and raise where the wire
        has none — never silently ignore them.
        """
        self._require("images")
        resp = self._send(self._image_generate_request(request))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._image_generation_from_response(request, resp)

    def speech_generate(self, request: SpeechGenerationRequest) -> SpeechGenerationResponse:
        """Text-to-speech.  Omitted ``voice``/``format`` mean the server's
        defaults, honestly reported in the returned part's media_type —
        lm15 injects no defaults of its own."""
        self._require("speech")
        resp = self._send(self._speech_generate_request(request))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._speech_generation_from_response(request, resp)

    # ─── Video generation (job-shaped on every wire: Sora / Veo / grok) ─────
    #
    # Same pure-hook pattern as batch: submit returns a ticket, status
    # polls it, result turns the terminal snapshot into a VideoPart —
    # with an optional fetch step (Sora delivers bytes from a content
    # endpoint; Veo and xAI put a URL in the terminal body).  The
    # VideoJob handle in lm15.video adds the wait/re-attach ergonomics.

    def _video_unsupported(self) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: video generation not supported", provider=self.provider)

    def _video_submit_request(self, request: VideoGenerationRequest) -> TransportRequest:
        raise self._video_unsupported()

    def _video_job_from_body(self, body: str, video_id: "str | None" = None) -> VideoJobInfo:
        """Parse a submit or status body; ``video_id`` is the known ticket
        id for wires whose status bodies do not echo it (xAI)."""
        raise self._video_unsupported()

    def _video_status_request(self, video_id: str) -> TransportRequest:
        raise self._video_unsupported()

    def _video_result_fetch(self, status_body: "dict[str, Any]") -> TransportRequest | None:
        """Optional download step; None = the terminal body carries the URL."""
        raise self._video_unsupported()

    def _video_part(self, status_body: "dict[str, Any]", fetched: "HttpResponse | None") -> VideoPart:
        raise self._video_unsupported()

    def _video_list_request(self, limit: int, model: str | None) -> TransportRequest:
        raise self._video_unsupported()

    def _video_jobs_from_list_body(self, body: str) -> "tuple[VideoJobInfo, ...]":
        raise self._video_unsupported()

    def video_submit(self, request: VideoGenerationRequest) -> VideoJobInfo:
        """Submit a video job; returns the ticket snapshot."""
        self._require("video")
        resp = self._send(self._video_submit_request(request))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._video_job_from_body(resp.text())

    def video_status(self, video_id: str) -> VideoJobInfo:
        self._require("video")
        resp = self._send(self._video_status_request(video_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._video_job_from_body(resp.text(), video_id)

    def video_result(self, video_id: str) -> VideoPart:
        """The finished video as a VideoPart, in the provider's own
        delivery mode (URL or bytes) — no silent re-hosting, no silent
        gigabyte download; ``part.bytes`` fetches URL results on demand.
        Raises ValueError while the job runs."""
        self._require("video")
        resp = self._send(self._video_status_request(video_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        job = self._video_job_from_body(resp.text(), video_id)
        if job.status not in VIDEO_TERMINAL_STATUSES:
            raise ValueError(
                f"video {video_id} is not finished (status={job.status!r}); "
                f"wait() or poll video_status() until done"
            )
        status_body = resp.json()
        fetch = self._video_result_fetch(status_body)
        fetched = None
        if fetch is not None:
            fetched = self._send(fetch)
            if fetched.status >= 400:
                raise self.normalize_error(fetched.status, fetched.text())
        return self._video_part(status_body, fetched)

    def video_list(self, limit: int = 20, model: str | None = None) -> "tuple[VideoJobInfo, ...]":
        """One page of this credential's video jobs.  ``model`` is required
        on Gemini (operations list per model), ignored on OpenAI; xAI has
        no list endpoint at all and raises."""
        self._require("video")
        resp = self._send(self._video_list_request(limit, model))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._video_jobs_from_list_body(resp.text())

    # Ergonomic verbs (ticket handles) ---------------------------------------

    def video_generate(self, request: VideoGenerationRequest) -> "VideoJob":
        """Submit and wrap the ticket in a VideoJob handle."""
        from ..video_jobs import VideoJob

        return VideoJob(self, self.video_submit(request))

    def video_job(self, video_id: str) -> "VideoJob":
        """Re-attach to an existing job by id alone."""
        from ..video_jobs import VideoJob

        return VideoJob(self, self.video_status(video_id))

    def video_jobs(self, limit: int = 20, model: str | None = None) -> "tuple[VideoJob, ...]":
        """Enumerate this credential's video jobs, where the wire lists
        them (OpenAI: account-wide; Gemini: per model, pass ``model``).
        xAI has no list endpoint and raises: the ticket you stored is
        the only copy."""
        from ..video_jobs import VideoJob

        return tuple(VideoJob(self, info) for info in self.video_list(limit, model))

    # ─── Live model listing (provisional endpoint) ──────────────────────────
    #
    # Adapters that support their provider's list-models endpoint override the
    # two hooks; list_models() itself is shared.  Each canonical ModelInfo
    # carries the usable Request.model string as `id` and the verbatim wire
    # entry under `origin.provider_data` (opaque, never cleaned).  Listing is
    # ADVISORY metadata per docs/model-hydration.md: it never changes what
    # build_request produces.

    def _models_request(self) -> TransportRequest:
        raise UnsupportedFeatureError(f"{self.provider}: model listing not supported", provider=self.provider)

    def _models_from_body(self, body: str) -> "tuple[ModelInfo, ...]":
        raise UnsupportedFeatureError(f"{self.provider}: model listing not supported", provider=self.provider)

    def list_models(self) -> "tuple[ModelInfo, ...]":
        """Fetch the models this credential can use, as canonical ModelInfo."""
        self._require("models")
        resp = self._send(self._models_request())
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._models_from_body(resp.text())

    # ─── Batch jobs (third execution mode: complete / stream / batch) ───────
    #
    # Adapters override the pure hooks; the five drivers (batch_submit,
    # batch_status, batch_results, batch_cancel, batch_list) are shared, as
    # are the ergonomic verbs (batch, batch_job, batches) returning BatchJob
    # handles.  Hooks are pure build/parse so the async twins and the future
    # harness direction drive the same code.  There is NO local fan-out
    # fallback: batch() means a provider-side queue or an honest error.

    def _batch_unsupported(self) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: batch not supported", provider=self.provider)

    def _batch_upload_request(self, request: BatchRequest) -> TransportRequest | None:
        """Optional pre-submit upload step (OpenAI's JSONL file); None = single-step."""
        return None

    def _batch_submit_request(self, request: BatchRequest, upload_body: "dict[str, Any] | None") -> TransportRequest:
        raise self._batch_unsupported()

    def _batch_job_from_body(self, body: str) -> BatchJobInfo:
        raise self._batch_unsupported()

    def _batch_status_request(self, batch_id: str) -> TransportRequest:
        raise self._batch_unsupported()

    def _batch_cancel_request(self, batch_id: str) -> TransportRequest:
        raise self._batch_unsupported()

    def _batch_result_fetches(self, status_body: "dict[str, Any]") -> "tuple[TransportRequest, ...]":
        raise self._batch_unsupported()

    def _batch_entries(self, status_body: "dict[str, Any]", fetched: "tuple[str, ...]") -> "tuple[BatchEntry, ...]":
        raise self._batch_unsupported()

    def _batch_list_request(self, limit: int) -> TransportRequest:
        raise self._batch_unsupported()

    def _batch_jobs_from_list_body(self, body: str) -> "tuple[BatchJobInfo, ...]":
        raise self._batch_unsupported()

    def batch_submit(self, request: BatchRequest) -> BatchJobInfo:
        """Submit to the provider's batch queue; returns the ticket snapshot."""
        self._require("batches")
        upload_body = None
        upload_req = self._batch_upload_request(request)
        if upload_req is not None:
            resp = self._send(upload_req)
            if resp.status >= 400:
                raise self.normalize_error(resp.status, resp.text())
            upload_body = resp.json()
        resp = self._send(self._batch_submit_request(request, upload_body))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._batch_job_from_body(resp.text())

    def batch_status(self, batch_id: str) -> BatchJobInfo:
        self._require("batches")
        resp = self._send(self._batch_status_request(batch_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._batch_job_from_body(resp.text())

    def batch_results(self, batch_id: str) -> "tuple[BatchEntry, ...]":
        """Entries in submission order; raises ValueError while the job runs."""
        self._require("batches")
        resp = self._send(self._batch_status_request(batch_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        job = self._batch_job_from_body(resp.text())
        if job.status not in BATCH_TERMINAL_STATUSES:
            raise ValueError(
                f"batch {batch_id} is not finished (status={job.status!r}); "
                f"wait() or poll batch_status() until done"
            )
        status_body = resp.json()
        texts = []
        for fetch in self._batch_result_fetches(status_body):
            fetched = self._send(fetch)
            if fetched.status >= 400:
                raise self.normalize_error(fetched.status, fetched.text())
            texts.append(fetched.text())
        return self._batch_entries(status_body, tuple(texts))

    def batch_cancel(self, batch_id: str) -> BatchJobInfo:
        """Request cancellation — a request, not a guarantee."""
        self._require("batches")
        resp = self._send(self._batch_cancel_request(batch_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._batch_job_from_body(resp.text())

    def batch_list(self, limit: int = 20) -> "tuple[BatchJobInfo, ...]":
        """Enumerate this credential's batch jobs, newest first.

        The provider is the system of record; recovery from a lost id must
        never depend on the user having been careful.
        """
        self._require("batches")
        resp = self._send(self._batch_list_request(limit))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._batch_jobs_from_list_body(resp.text())

    def batch(self, requests: "BatchRequest | Sequence[Request]", *, model: str | None = None,
              label: str | None = None, extensions: "dict[str, Any] | None" = None) -> "BatchJob":
        """Third execution mode: many requests, later, ~half price."""
        from ..batch import BatchJob

        if isinstance(requests, BatchRequest):
            request = requests
        else:
            request = BatchRequest(model=model, requests=tuple(requests), label=label, extensions=extensions)
        return BatchJob(self, self.batch_submit(request))

    def batch_job(self, batch_id: str) -> "BatchJob":
        """Re-attach to a submitted batch from its id (one status round trip)."""
        from ..batch import BatchJob

        return BatchJob(self, self.batch_status(batch_id))

    def batches(self, limit: int = 20) -> "tuple[BatchJob, ...]":
        """Lost the ticket? The queue remembers."""
        from ..batch import BatchJob

        return tuple(BatchJob(self, info) for info in self.batch_list(limit))

    # ─── Cache resources (the stored tier of MAP-6) ────────────────────
    #
    # Same pure-hook pattern as files.  A stored cache belongs to one model
    # on every provider that has the tier; the prefix Request supplies
    # model, system, tools, and messages.  Adapters without the tier leave
    # the hooks raising and `cache()` returns a CachedPrefix without a
    # resource — the marks-or-automatic path.  Nothing here names a
    # provider; id formats and TTL spellings are the adapter's.

    def _caches_unsupported(self) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: stored caches not supported", provider=self.provider)

    def _cache_create_request(self, prefix: Request, ttl_seconds: int | None, label: str | None) -> TransportRequest:
        raise self._caches_unsupported()

    def _cache_info_from_body(self, body: str) -> CacheInfo:
        raise self._caches_unsupported()

    def _cache_get_request(self, cache_id: str) -> TransportRequest:
        raise self._caches_unsupported()

    def _cache_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        raise self._caches_unsupported()

    def _cache_page_from_list_body(self, body: str) -> CachePage:
        raise self._caches_unsupported()

    def _cache_delete_request(self, cache_id: str) -> TransportRequest:
        raise self._caches_unsupported()

    def _cache_update_request(self, cache_id: str, ttl_seconds: int) -> TransportRequest:
        raise self._caches_unsupported()

    @staticmethod
    def _check_cache_prefix(prefix: Request, ttl_seconds: int | None) -> None:
        if prefix.config != Config():
            raise ValueError("cache_create: the prefix Request must carry a default Config (a stored cache has no generation settings)")
        if ttl_seconds is not None and (isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0):
            raise ValueError("ttl_seconds must be a positive int")

    def cache_create(self, prefix: Request, *, ttl_seconds: int | None = None, label: str | None = None) -> CacheInfo:
        """Store the prefix (model, system, tools, messages) as a cache object."""
        self._require("caches")
        self._check_cache_prefix(prefix, ttl_seconds)
        resp = self._send(self._cache_create_request(prefix, ttl_seconds, label))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._cache_info_from_body(resp.text())

    def cache_get(self, cache_id: str) -> CacheInfo:
        self._require("caches")
        resp = self._send(self._cache_get_request(cache_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._cache_info_from_body(resp.text())

    def cache_list(self, limit: int = 20, cursor: str | None = None) -> CachePage:
        self._require("caches")
        resp = self._send(self._cache_list_request(limit, cursor))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._cache_page_from_list_body(resp.text())

    def cache_delete(self, cache_id: str) -> None:
        """Returning without an exception IS the confirmation (the files precedent)."""
        self._require("caches")
        resp = self._send(self._cache_delete_request(cache_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())

    def cache_update(self, cache_id: str, *, ttl_seconds: int) -> CacheInfo:
        self._require("caches")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive int")
        resp = self._send(self._cache_update_request(cache_id, ttl_seconds))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._cache_info_from_body(resp.text())

    def cache(self, prefix: Request, *, ttl_seconds: int | None = None, label: str | None = None) -> CachedPrefix:
        """Make a prompt beginning reusable with the best tier this provider has.

        Resource tier: creates the stored object (one network call, billed
        per hour while it lives) and returns it inside the CachedPrefix.
        Marks and automatic tiers: pure — the CachedPrefix only records the
        boundary; `cached + messages` places the mark or sends nothing.
        """
        if self.supports.caches:
            return CachedPrefix(prefix, self.cache_create(prefix, ttl_seconds=ttl_seconds, label=label))
        self._check_cache_prefix(prefix, ttl_seconds)
        return CachedPrefix(prefix)

    # ─── Files (account-scoped storage: upload / get / list / delete / download) ─
    #
    # Adapters override the pure build/parse hooks; the drivers are shared
    # so the async twins and a future harness direction drive identical
    # code.  All five operations exist on all three first-party providers
    # (verified live 2026-08-31); which FILES support download is per-file
    # provider policy — lm15 forwards the provider's typed refusal instead
    # of second-guessing it.

    def _files_unsupported(self) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(f"{self.provider}: files not supported", provider=self.provider)

    def _file_upload_request(self, request: FileUploadRequest) -> TransportRequest:
        raise self._files_unsupported()

    def _file_info_from_body(self, body: str) -> FileInfo:
        raise self._files_unsupported()

    def _file_get_request(self, file_id: str) -> TransportRequest:
        raise self._files_unsupported()

    def _file_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        raise self._files_unsupported()

    def _file_page_from_list_body(self, body: str) -> FilePage:
        raise self._files_unsupported()

    def _file_delete_request(self, file_id: str) -> TransportRequest:
        raise self._files_unsupported()

    def _file_download_request(self, file_id: str) -> TransportRequest:
        raise self._files_unsupported()

    def file_upload(self, request: FileUploadRequest) -> FileInfo:
        """Store a file with the provider; returns its canonical snapshot.

        ``FileInfo.id`` is the reference to place in a media Part's
        ``file_id``.  On Gemini the file may come back ``pending``
        (processing); ``file_wait_ready`` covers that.
        """
        self._require("files")
        resp = self._send(self._file_upload_request(request))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._file_info_from_body(resp.text())

    def file_get(self, file_id: str) -> FileInfo:
        self._require("files")
        resp = self._send(self._file_get_request(file_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._file_info_from_body(resp.text())

    def file_list(self, limit: int = 20, cursor: str | None = None) -> FilePage:
        """One page of this credential's stored files.

        The provider is the system of record: a lost file id is recovered
        by listing, never by client-side bookkeeping.  ``cursor`` is the
        opaque ``next_cursor`` of the previous page.
        """
        self._require("files")
        resp = self._send(self._file_list_request(limit, cursor))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return self._file_page_from_list_body(resp.text())

    def file_delete(self, file_id: str) -> None:
        """Delete a stored file.  Returning without an exception IS the
        confirmation; provider acknowledgement bodies differ and carry no
        canonical information."""
        self._require("files")
        resp = self._send(self._file_delete_request(file_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())

    def file_download(self, file_id: str) -> bytes:
        """Download a file's content, when THIS file supports download.

        Every provider restricts which files are downloadable (Anthropic:
        tool-generated only; OpenAI: by purpose; Gemini: generated only)
        and refuses the rest with a typed error — forwarded, not masked.
        """
        self._require("files")
        resp = self._send(self._file_download_request(file_id))
        if resp.status >= 400:
            raise self.normalize_error(resp.status, resp.text())
        return resp.body

    def file_wait_ready(self, file_id: str, poll_every: float = 2.0, timeout: float | None = None) -> FileInfo:
        """Poll until the file leaves ``pending``; returns the terminal
        snapshot (``ready`` or ``failed``) — check ``readiness``, mirroring
        BatchJob.wait's return-don't-raise convention. Gemini uploads and
        Azure OpenAI v1 uploads can be pending; the first poll usually returns."""
        self._require("files")
        import time as _time

        deadline = None if timeout is None else _time.monotonic() + timeout
        info = self.file_get(file_id)
        while info.readiness == "pending":
            if deadline is not None and _time.monotonic() >= deadline:
                raise TimeoutError(f"file {file_id} still pending after {timeout}s")
            _time.sleep(poll_every)
            info = self.file_get(file_id)
        return info


def batch_entry_request(model: object) -> Request:
    """Synthetic Request for parsing a batch entry body.

    Batch results outlive the submitting process (re-attach by id), so the
    original Request is not available; parse_response only reads
    ``request.model`` as a fallback when the body lacks one, and every
    batch entry body carries its model.
    """
    from ..types import Message

    name = model if isinstance(model, str) and model else "batch"
    return Request(model=name, messages=(Message.user("-"),))


def batch_entry_http(body: "dict[str, Any]", status: int = 200) -> HttpResponse:
    """Wrap a decoded batch entry body for the frozen parse_response path."""
    return HttpResponse(
        status=status,
        reason="OK" if status < 400 else "Error",
        headers=[("content-type", "application/json")],
        body=json.dumps(body).encode("utf-8"),
    )


class UnsupportedLiveSession:
    def send(self, event=None, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def send_turn(self, *args, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def send_audio(self, *args, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def send_image(self, *args, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def send_text(self, *args, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def send_tool_result(self, *args, **kwargs):
        raise UnsupportedFeatureError("live session not supported")

    def interrupt(self):
        raise UnsupportedFeatureError("live session not supported")

    def end_audio(self):
        raise UnsupportedFeatureError("live session not supported")

    def recv(self):
        raise UnsupportedFeatureError("live session not supported")

    def close(self) -> None:
        return


def _iter_lines(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Split arbitrary byte chunks into newline-terminated lines for SSE."""

    buf = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            idx = buf.find(b"\n")
            if idx < 0:
                break
            line = bytes(buf[: idx + 1])
            del buf[: idx + 1]
            yield line
    if buf:
        yield bytes(buf)
