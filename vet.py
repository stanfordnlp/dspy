"""
lm15.vet — the lm15 vet shim (reference implementation).

Speaks the newline-delimited JSON protocol defined in
``lm15-contract/harness/PROTOCOL.md``: one JSON request per stdin line, one
JSON reply per stdout line, same order, one output per input. The shim only
transforms — the harness performs all comparison and runs the shim inside a
no-network sandbox, so this module must never open a connection.

Run as: ``python -m lm15.vet`` (cwd: lm15-python).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import sys
import urllib.parse
from typing import Any, Callable, Iterator, Literal, get_args, get_origin

from . import types as lm15_types
from . import serde
from .errors import AmbiguousModelError, LM15Error, StreamAssemblyError, UnknownModelError, canonical_error_code
from .providers import HttpResponse
from .providers.base import BaseProviderLM
from .result import coalesce_stream, materialize_response
from .sse import parse_sse
from .types import Request, Response, StreamEvent

JsonObject = dict[str, Any]

LANGUAGE = "python"

try:
    from importlib.metadata import version as _dist_version

    IMPL_VERSION = _dist_version("lm15")
except Exception:  # pragma: no cover - metadata is absent in odd installs
    IMPL_VERSION = "0.0.0"

# Parse-only ops (parse_response, replay_stream, normalize_error) construct an
# adapter but never build auth headers; the key value is irrelevant and must
# never come from the environment.
_PARSE_ONLY_KEY = "vet-parse-only"


# ─── Adapters ────────────────────────────────────────────────────────

def _credential_of(msg: JsonObject):
    """PROTOCOL.md: ``credential`` (an AUTH-2 value) or the ``api_key`` shorthand."""
    from .credentials import credential_from_dict

    cred = msg.get("credential")
    if isinstance(cred, dict):
        return credential_from_dict(cred)
    return str(msg["api_key"])


def _clock_of(msg: JsonObject):
    """The fixed clock the harness pins (``now``), or None for wall time."""
    from .credentials import parse_rfc3339

    now = msg.get("now")
    if now is None:
        return None
    fixed = parse_rfc3339(str(now))
    return lambda: fixed


def _settings_of(msg: JsonObject) -> dict[str, str] | None:
    settings = msg.get("settings")
    return {str(k): str(v) for k, v in settings.items()} if isinstance(settings, dict) else None


def _adapter(msg: JsonObject) -> BaseProviderLM:
    return adapter_for_provider(str(msg["provider"]), _credential_of(msg), _base_url(msg),
                                settings=_settings_of(msg), clock=_clock_of(msg))


def adapter_for_provider(provider: str, api_key: Any, base_url: str | None = None, *,
                         settings: dict[str, str] | None = None, clock: Any = None) -> BaseProviderLM:
    """The LM a contract case's ``provider`` string names, from the registry.

    A chat-bound provider (groq, deepseek, …) is the chat dialect with that
    provider's compat preset and access policy bound, exactly as the router
    builds it; an adapter-owned provider is its class.  ``base_url``
    overrides the provider's default (local vLLM/SGLang cases).
    """
    from .registry import lookup

    definition = lookup(provider)
    if definition is None:
        raise ValueError(f"unknown provider: {provider}")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if definition.bound:
        kwargs["access"] = definition.access
        if definition.compat is not None:
            # The Gemini dialect takes no compat (the router does the same).
            kwargs["compat"] = definition.compat
    if definition.hosted:
        kwargs["settings"] = settings
    if clock is not None:
        kwargs["clock"] = clock
    if definition.id == "openai-codex":
        # PROTOCOL.md pins the harness account id: the ctor cannot derive one
        # from a non-JWT injected key, and the wire header must be exact.
        kwargs["account_id"] = "test-account"
    return definition.adapter(**kwargs)


# ─── Serde kind table ────────────────────────────────────────────────

JsonToObj = Callable[[JsonObject], Any]
ObjToJson = Callable[[Any], JsonObject]

def _credential_from_dict(data: JsonObject):
    from .credentials import credential_from_dict

    return credential_from_dict(data)


def _credential_to_dict(value: Any) -> JsonObject:
    from .credentials import credential_to_dict

    return credential_to_dict(value)


KIND_SERDE: dict[str, tuple[JsonToObj, ObjToJson]] = {
    "part": (serde.part_from_dict, serde.part_to_dict),
    "message": (serde.message_from_dict, serde.message_to_dict),
    "tool": (serde.tool_from_dict, serde.tool_to_dict),
    "tool_choice": (serde.tool_choice_from_dict, serde.tool_choice_to_dict),
    "reasoning": (serde.reasoning_from_dict, serde.reasoning_to_dict),
    "config": (serde.config_from_dict, serde.config_to_dict),
    "cache_config": (serde.cache_config_from_dict, serde.cache_config_to_dict),
    "cache_info": (serde.cache_info_from_dict, serde.cache_info_to_dict),
    "cache_page": (serde.cache_page_from_dict, serde.cache_page_to_dict),
    "cached_prefix": (serde.cached_prefix_from_dict, serde.cached_prefix_to_dict),
    "token_logprob": (serde.token_logprob_from_dict, serde.token_logprob_to_dict),
    "continuation_state": (serde.continuation_from_dict, serde.continuation_to_dict),
    "error_detail": (serde.error_detail_from_dict, serde.error_detail_to_dict),
    "delta": (serde.delta_from_dict, serde.delta_to_dict),
    "usage": (serde.usage_from_dict, serde.usage_to_dict),
    "credential": (_credential_from_dict, _credential_to_dict),
    "stream_event": (serde.stream_event_from_dict, serde.stream_event_to_dict),
    "request": (serde.request_from_dict, serde.request_to_dict),
    "response": (serde.response_from_dict, serde.response_to_dict),
    "model_info": (serde.model_info_from_dict, serde.model_info_to_dict),
    "batch_request": (serde.batch_request_from_dict, serde.batch_request_to_dict),
    "batch_job": (serde.batch_job_from_dict, serde.batch_job_to_dict),
    "batch_entry": (serde.batch_entry_from_dict, serde.batch_entry_to_dict),
    "file_upload_request": (serde.file_upload_request_from_dict, serde.file_upload_request_to_dict),
    "file_info": (serde.file_info_from_dict, serde.file_info_to_dict),
    "file_page": (serde.file_page_from_dict, serde.file_page_to_dict),
    "image_generation_request": (serde.image_generation_request_from_dict, serde.image_generation_request_to_dict),
    "image_generation_response": (serde.image_generation_response_from_dict, serde.image_generation_response_to_dict),
    "speech_generation_request": (serde.speech_generation_request_from_dict, serde.speech_generation_request_to_dict),
    "speech_generation_response": (serde.speech_generation_response_from_dict, serde.speech_generation_response_to_dict),
    "video_generation_request": (serde.video_generation_request_from_dict, serde.video_generation_request_to_dict),
    "video_job": (serde.video_job_from_dict, serde.video_job_to_dict),
    "audio_format": (serde.audio_format_from_dict, serde.audio_format_to_dict),
    "live_config": (serde.live_config_from_dict, serde.live_config_to_dict),
    "live_client_event": (serde.live_client_event_from_dict, serde.live_client_event_to_dict),
    "live_server_event": (serde.live_server_event_from_dict, serde.live_server_event_to_dict),
}


def _serde_for_kind(kind: str) -> tuple[JsonToObj, ObjToJson]:
    if kind not in KIND_SERDE:
        raise ValueError(f"unknown kind: {kind}")
    return KIND_SERDE[kind]


# ─── Normalization helpers ───────────────────────────────────────────

def normalize_transport_request(transport_req: Any) -> JsonObject:
    """Normalize a TransportRequest into the protocol's build_request shape.

    Mirrors conformance/cross_sdk/dump_request.py, except nothing is ever
    redacted — the harness asserts exact auth formatting against the api_key
    it injected.
    """
    parsed = urllib.parse.urlparse(transport_req.url)
    params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    headers: JsonObject = {key.lower(): value for key, value in transport_req.headers}

    out: JsonObject = {
        "method": transport_req.method,
        "url": url,
        "params": params,
        "headers": headers,
        "body": None,
    }

    body = transport_req.body
    if body:
        content_type = headers.get("content-type", "")
        if "json" in content_type.lower():
            try:
                out["body"] = json.loads(body.decode("utf-8"))
                return out
            except Exception:
                pass
        out["body_b64"] = base64.b64encode(body).decode("ascii")
    return out


def _http_response(status: int, body: bytes) -> HttpResponse:
    return HttpResponse(
        status=status,
        reason="OK",
        headers=[("content-type", "application/json")],
        body=body,
    )


def _response_result(response: Response) -> JsonObject:
    """Serialize a Response per the protocol, surfacing the unmapped canary."""
    result: JsonObject = {"canonical_response": serde.response_to_dict(response)}
    provider_data = response.provider_data if isinstance(response.provider_data, dict) else {}
    unmapped = provider_data.get("_lm15_unmapped")
    if unmapped is not None:
        result["unmapped"] = list(unmapped)
    return result


def _parse_stream_body(lm: BaseProviderLM, request: Request, body: bytes) -> list[StreamEvent]:
    def raw_events() -> Iterator[StreamEvent]:
        for raw in parse_sse(iter(body.splitlines(keepends=True))):
            for event in lm.parse_stream_events(request, raw):
                if event is not None:
                    yield event

    # MAP-3/MAP-4: the canonical event trace is the POST-coalesce trace —
    # exactly one merged StreamEndEvent, final, and exactly one leading
    # StreamStartEvent (synthesized with the request's model for dialects
    # without a start frame).
    return list(coalesce_stream(raw_events(), model=request.model))


# ─── Ops ─────────────────────────────────────────────────────────────

def op_capabilities(msg: JsonObject) -> JsonObject:
    return {
        "language": LANGUAGE,
        "ops": sorted(HANDLERS),
        "impl_version": IMPL_VERSION,
    }


def _base_url(msg: JsonObject) -> str | None:
    base_url = msg.get("base_url")
    return str(base_url) if base_url is not None else None


def op_build_request(msg: JsonObject) -> JsonObject:
    lm = _adapter(msg)
    request = serde.request_from_dict(msg["canonical_request"])
    transport_req = lm.build_request(request, stream=bool(msg.get("stream", False)))
    return normalize_transport_request(transport_req)


def op_ingest_openai_chat(msg: JsonObject) -> JsonObject:
    """MAP-12: a Chat Completions request body → canonical Request, under the
    compat the case's provider binds (the same adapter build_request uses).
    Pure; no credential is read (PROTOCOL.md)."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    ingest = getattr(lm, "request_from_openai_chat", None)
    if ingest is None:
        raise ValueError(f"provider {msg['provider']!r} does not speak the Chat Completions wire; nothing to ingest")
    return {"canonical_request": serde.request_to_dict(ingest(msg["body"]))}


def op_parse_response(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    request = serde.request_from_dict(msg["canonical_request"])
    body = base64.b64decode(msg["body_b64"])
    response = lm.parse_response(request, _http_response(int(msg["status"]), body))
    return _response_result(response)


def op_replay_stream(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    request = serde.request_from_dict(msg["canonical_request"])
    body = base64.b64decode(msg["body_b64"])
    events = _parse_stream_body(lm, request, body)
    event_dicts = [serde.stream_event_to_dict(e) for e in events]
    try:
        response = materialize_response(iter(events), request)
    except StreamAssemblyError as exc:
        # The trace parsed; assembly refused (MAP-9). Report the trace with
        # the error so the golden pins both what arrived and the refusal.
        raise _OpFailure(exc, {"events": event_dicts}) from exc
    result: JsonObject = {"events": event_dicts}
    result.update(_response_result(response))
    return result


def op_normalize_error(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    err = lm.normalize_error(int(msg["status"]), str(msg["body_text"]))
    return {
        "class": type(err).__name__,
        "code": err.code or canonical_error_code(err),
        "provider_code": err.provider_code,
        "message": err.message,
    }


def op_serde_roundtrip(msg: JsonObject) -> JsonObject:
    from_dict, to_dict = _serde_for_kind(str(msg["kind"]))
    return {"value": to_dict(from_dict(msg["value"]))}


def op_validate(msg: JsonObject) -> JsonObject:
    from_dict, to_dict = _serde_for_kind(str(msg["kind"]))
    obj = from_dict(msg["value"])
    return {"ok": True, "normalized": to_dict(obj)}


def op_resolve_model(msg: JsonObject) -> JsonObject:
    """PROTOCOL.md resolve_model: the router's pure ``resolve`` over
    harness-supplied inputs only — the env map (never the process env) and,
    when given, an explicit catalog of canonical ModelInfo values."""
    from .models import ModelRegistry
    from .router import LMRouter, RouterConfig

    env = {str(k): str(v) for k, v in (msg.get("env") or {}).items()}
    registry = None
    if "catalog" in msg:
        registry = ModelRegistry()
        for entry in msg["catalog"] or []:
            registry.add(serde.model_info_from_dict(entry), replace=False)
    resolution = LMRouter(RouterConfig(registry=registry, env=env)).resolve(str(msg["model"]))
    return {"provider": resolution.provider, "model": resolution.model, "source": resolution.source}


def op_build_models_request(msg: JsonObject) -> JsonObject:
    lm = _adapter(msg)
    return normalize_transport_request(lm._models_request())


def op_parse_models_response(msg: JsonObject) -> JsonObject:
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    body = base64.b64decode(msg["body_b64"]).decode("utf-8")
    status = int(msg["status"])
    if status >= 400:
        raise lm.normalize_error(status, body)
    models = lm._models_from_body(body)
    return {"models": [serde.model_info_to_dict(m) for m in models]}


def _headers_list(msg: JsonObject) -> list:
    headers = msg.get("headers") or {}
    return [(str(k), str(v)) for k, v in headers.items()]


def op_generation_build(msg: JsonObject) -> JsonObject:
    """Wire request for image_generate / speech_generate (kind discriminates)."""
    lm = _adapter(msg)
    kind = str(msg["kind"])
    if kind == "image":
        request = serde.image_generation_request_from_dict(msg["generation_request"])
        return normalize_transport_request(lm._image_generate_request(request))
    if kind == "speech":
        request = serde.speech_generation_request_from_dict(msg["generation_request"])
        return normalize_transport_request(lm._speech_generate_request(request))
    raise ValueError(f"unknown generation kind: {kind}")


def op_generation_parse(msg: JsonObject) -> JsonObject:
    """Canonical generation response from a pinned wire body (+ headers:
    OpenAI speech is raw bytes typed only by content-type)."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    kind = str(msg["kind"])
    status = int(msg["status"])
    body = base64.b64decode(msg["body_b64"])
    if status >= 400:
        raise lm.normalize_error(status, body.decode("utf-8", "replace"))
    resp = HttpResponse(status=status, reason="OK", headers=_headers_list(msg), body=body)
    if kind == "image":
        request = serde.image_generation_request_from_dict(msg["generation_request"])
        return serde.image_generation_response_to_dict(lm._image_generation_from_response(request, resp))
    if kind == "speech":
        request = serde.speech_generation_request_from_dict(msg["generation_request"])
        return serde.speech_generation_response_to_dict(lm._speech_generation_from_response(request, resp))
    raise ValueError(f"unknown generation kind: {kind}")


def op_file_op_build(msg: JsonObject) -> JsonObject:
    """Wire request for one files-lifecycle operation (file_op discriminates)."""
    lm = _adapter(msg)
    op = str(msg["file_op"])
    if op == "upload":
        request = serde.file_upload_request_from_dict(msg["upload_request"])
        return normalize_transport_request(lm._file_upload_request(request))
    if op == "get":
        return normalize_transport_request(lm._file_get_request(str(msg["file_id"])))
    if op == "list":
        cursor = msg.get("cursor")
        return normalize_transport_request(lm._file_list_request(int(msg.get("limit", 20)), cursor))
    if op == "delete":
        return normalize_transport_request(lm._file_delete_request(str(msg["file_id"])))
    if op == "download":
        return normalize_transport_request(lm._file_download_request(str(msg["file_id"])))
    raise ValueError(f"unknown file_op: {op}")


def op_file_op_parse(msg: JsonObject) -> JsonObject:
    """Canonical FileInfo / FilePage from a pinned wire body (kind discriminates)."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    kind = str(msg["kind"])
    status = int(msg["status"])
    body = base64.b64decode(msg["body_b64"]).decode("utf-8")
    if status >= 400:
        raise lm.normalize_error(status, body)
    if kind == "info":
        return {"file": serde.file_info_to_dict(lm._file_info_from_body(body))}
    if kind == "page":
        return {"page": serde.file_page_to_dict(lm._file_page_from_list_body(body))}
    raise ValueError(f"unknown file parse kind: {kind}")


def op_cache_op_build(msg: JsonObject) -> JsonObject:
    """Wire request for one cache-resource operation (cache_op discriminates)."""
    lm = _adapter(msg)
    op = str(msg["cache_op"])
    if op == "create":
        prefix = serde.request_from_dict(msg["prefix_request"])
        lm._check_cache_prefix(prefix, msg.get("ttl_seconds"))
        return normalize_transport_request(lm._cache_create_request(prefix, msg.get("ttl_seconds"), msg.get("label")))
    if op == "get":
        return normalize_transport_request(lm._cache_get_request(str(msg["cache_id"])))
    if op == "list":
        return normalize_transport_request(lm._cache_list_request(int(msg.get("limit", 20)), msg.get("cursor")))
    if op == "delete":
        return normalize_transport_request(lm._cache_delete_request(str(msg["cache_id"])))
    if op == "update":
        return normalize_transport_request(lm._cache_update_request(str(msg["cache_id"]), int(msg["ttl_seconds"])))
    raise ValueError(f"unknown cache_op: {op}")


def op_cache_op_parse(msg: JsonObject) -> JsonObject:
    """Canonical CacheInfo / CachePage from a pinned wire body (kind discriminates)."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    kind = str(msg["kind"])
    status = int(msg["status"])
    body = base64.b64decode(msg["body_b64"]).decode("utf-8")
    if status >= 400:
        raise lm.normalize_error(status, body)
    if kind == "info":
        return {"cache": serde.cache_info_to_dict(lm._cache_info_from_body(body))}
    if kind == "page":
        return {"page": serde.cache_page_to_dict(lm._cache_page_from_list_body(body))}
    raise ValueError(f"unknown cache parse kind: {kind}")


def op_video_op_build(msg: JsonObject) -> JsonObject:
    """Wire request(s) for one video-job operation (action discriminates).

    Always returns {"requests": [...]}: result_fetch may be zero requests
    (xAI's terminal body carries a public URL) or one (Sora content, Veo's
    key-bound file URI).
    """
    lm = _adapter(msg)
    action = str(msg["action"])
    if action == "submit":
        request = serde.video_generation_request_from_dict(msg["video_request"])
        return {"requests": [normalize_transport_request(lm._video_submit_request(request))]}
    if action == "status":
        return {"requests": [normalize_transport_request(lm._video_status_request(str(msg["video_id"])))]}
    if action == "result_fetch":
        fetch = lm._video_result_fetch(msg["status_body"])
        return {"requests": [] if fetch is None else [normalize_transport_request(fetch)]}
    if action == "list":
        return {"requests": [normalize_transport_request(lm._video_list_request(int(msg.get("limit", 20)), msg.get("model")))]}
    raise ValueError(f"unknown video action: {action}")


def op_video_op_parse(msg: JsonObject) -> JsonObject:
    """Canonical video snapshots / parts from pinned wire bodies."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    kind = str(msg["kind"])
    if kind == "job":
        status = int(msg["status"])
        body = base64.b64decode(msg["body_b64"]).decode("utf-8")
        if status >= 400:
            raise lm.normalize_error(status, body)
        return {"job": serde.video_job_to_dict(lm._video_job_from_body(body, msg.get("video_id")))}
    if kind == "list":
        body = base64.b64decode(msg["body_b64"]).decode("utf-8")
        return {"jobs": [serde.video_job_to_dict(j) for j in lm._video_jobs_from_list_body(body)]}
    if kind == "part":
        fetched = None
        if msg.get("fetched_b64") is not None:
            fetched = HttpResponse(
                status=200, reason="OK", headers=_headers_list(msg),
                body=base64.b64decode(msg["fetched_b64"]),
            )
        return {"part": serde.part_to_dict(lm._video_part(msg["status_body"], fetched))}
    raise ValueError(f"unknown video parse kind: {kind}")


def op_batch_op_build(msg: JsonObject) -> JsonObject:
    """Wire request(s) for one batch operation (action discriminates).

    Always returns {"requests": [...]}: upload may be zero requests
    (single-step providers), result_fetches may be several (OpenAI's
    output and error files).
    """
    lm = _adapter(msg)
    action = str(msg["action"])
    if action == "upload":
        request = serde.batch_request_from_dict(msg["batch_request"])
        upload = lm._batch_upload_request(request)
        return {"requests": [] if upload is None else [normalize_transport_request(upload)]}
    if action == "submit":
        request = serde.batch_request_from_dict(msg["batch_request"])
        upload_body = msg.get("upload_body")
        return {"requests": [normalize_transport_request(lm._batch_submit_request(request, upload_body))]}
    if action == "status":
        return {"requests": [normalize_transport_request(lm._batch_status_request(str(msg["batch_id"])))]}
    if action == "cancel":
        return {"requests": [normalize_transport_request(lm._batch_cancel_request(str(msg["batch_id"])))]}
    if action == "list":
        return {"requests": [normalize_transport_request(lm._batch_list_request(int(msg.get("limit", 20))))]}
    if action == "result_fetches":
        fetches = lm._batch_result_fetches(msg["status_body"])
        return {"requests": [normalize_transport_request(f) for f in fetches]}
    raise ValueError(f"unknown batch action: {action}")


def op_batch_op_parse(msg: JsonObject) -> JsonObject:
    """Canonical batch snapshots from pinned wire bodies (kind discriminates)."""
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    kind = str(msg["kind"])
    if kind == "job":
        status = int(msg["status"])
        body = base64.b64decode(msg["body_b64"]).decode("utf-8")
        if status >= 400:
            raise lm.normalize_error(status, body)
        return {"job": serde.batch_job_to_dict(lm._batch_job_from_body(body))}
    if kind == "list":
        body = base64.b64decode(msg["body_b64"]).decode("utf-8")
        return {"jobs": [serde.batch_job_to_dict(j) for j in lm._batch_jobs_from_list_body(body)]}
    if kind == "entries":
        fetched = tuple(base64.b64decode(b).decode("utf-8") for b in msg.get("fetched_b64", []))
        entries = lm._batch_entries(msg["status_body"], fetched)
        return {"entries": [serde.batch_entry_to_dict(e) for e in entries]}
    raise ValueError(f"unknown batch parse kind: {kind}")


def op_replay_live(msg: JsonObject) -> JsonObject:
    """Live transcript replay: the pure websocket codec, no socket.

    Three transformations per case, all driven by the recorded transcript:
    setup frames from the LiveConfig, wire frames from each canonical
    client event, and canonical server events from each verbatim recorded
    server frame (empty list = housekeeping frame, deliberately ignored).
    The harness performs all comparison; session mechanics (locking,
    queues, iteration sugar) are per-language and stay out of scope.
    """
    lm = adapter_for_provider(str(msg["provider"]), _PARSE_ONLY_KEY, _base_url(msg), settings=_settings_of(msg), clock=_clock_of(msg))
    config = serde.live_config_from_dict(msg["live_config"])
    encoder = lm._live_encoder(config)
    client_frames = [
        encoder(serde.live_client_event_from_dict(event))
        for event in msg.get("client_events", [])
    ]
    events = []
    for frame_b64 in msg.get("server_frames_b64", []):
        raw = base64.b64decode(frame_b64)
        events.append([serde.live_server_event_to_dict(e) for e in lm._decode_live_server_event(raw)])
    return {
        "setup_frames": lm._live_setup_frames(config),
        "client_frames": client_frames,
        "events": events,
    }


def op_explain_auth(msg: JsonObject) -> JsonObject:
    """AUTH-7 resolution chain over harness-supplied inputs only.

    The harness owns every input: ``env`` (always passed, even empty, so the
    real process environment never leaks in), ``api_keys_providers`` planted
    with the fixture sentinel, and ``credentials_path`` pointing at a
    harness-materialized borrowed file. ``report_text`` carries every human
    rendering so the harness can enforce AUTH-5 (sentinel absence) itself.
    """
    from .doctor import explain_auth

    provider = str(msg["provider"])
    sentinel = str(msg["sentinel"])
    kwargs: dict[str, Any] = {"env": {str(k): str(v) for k, v in (msg.get("env") or {}).items()}}
    providers = msg.get("api_keys_providers") or []
    if providers:
        kwargs["api_keys"] = {str(p): sentinel for p in providers}
    files = msg.get("files")
    if isinstance(files, dict):
        kwargs["files"] = {str(k): str(v) for k, v in files.items()}
    if kwargs["env"].get("HOME"):
        kwargs["home"] = kwargs["env"]["HOME"]
    if isinstance(msg.get("settings"), dict):
        kwargs["settings"] = {str(k): str(v) for k, v in msg["settings"].items()}
    credentials_path = msg.get("credentials_path")
    if credentials_path is not None:
        if provider == "claude-code":
            kwargs["claude_credentials_path"] = str(credentials_path)
        elif provider == "xai":
            kwargs["xai_credentials_path"] = str(credentials_path)
        else:
            kwargs["codex_auth_path"] = str(credentials_path)

    report = explain_auth(provider, **kwargs)
    return {
        "configured": report.configured,
        "steps": [{"kind": step.kind, "state": step.state} for step in report.steps],
        "report_text": "\n".join((report.describe(), repr(report), str(report))),
    }


def op_token_exchange_build(msg: JsonObject) -> JsonObject:
    """PROTOCOL.md token_exchange_build: the exact token-exchange request a
    chain rung sends under the fixed clock (auth/token-vectors.json)."""
    from .cloud.chains import ChainContext, token_exchange_build
    from .credentials import parse_rfc3339
    from .registry import lookup

    definition = lookup(str(msg["provider"]))
    if definition is None:
        raise ValueError(f"unknown provider: {msg['provider']}")
    inputs = dict(msg.get("input") or msg.get("credential") or {})
    env = {str(k): str(v) for k, v in (inputs.get("env") or {}).items()}
    files: dict[str, str] = {}
    if inputs.get("certificate_pem") and env.get("AZURE_CLIENT_CERTIFICATE_PATH"):
        # The harness gives the certificate and the key separately; the
        # environment rung reads one PEM file carrying both.
        files[env["AZURE_CLIENT_CERTIFICATE_PATH"]] = str(inputs["certificate_pem"]) + "\n" + str(inputs.get("private_key_pem") or "")
    fixed = parse_rfc3339(str(msg["now"]))
    ctx = ChainContext(env=env, files=files or None, now=lambda: fixed,
                       settings={str(k): str(v) for k, v in (inputs.get("settings") or msg.get("settings") or {}).items()})
    return token_exchange_build(definition.access, str(msg["rung"]), inputs, ctx)


def op_token_exchange_parse(msg: JsonObject) -> JsonObject:
    from .cloud.chains import ChainContext, token_exchange_parse
    from .credentials import credential_to_dict, parse_rfc3339
    from .registry import lookup

    definition = lookup(str(msg["provider"]))
    if definition is None:
        raise ValueError(f"unknown provider: {msg['provider']}")
    fixed = parse_rfc3339(str(msg["now"]))
    body = msg.get("body")
    if body is None and msg.get("body_b64") is not None:
        body = json.loads(base64.b64decode(str(msg["body_b64"])))
    ctx = ChainContext(env={}, now=lambda: fixed)
    try:
        credential = token_exchange_parse(definition.access, str(msg["rung"]), int(msg.get("status", 200)), dict(body or {}), ctx)
    except LM15Error as exc:
        return {"ok": False, "error": {"class": type(exc).__name__, "code": getattr(exc, "code", None)}}
    return {"ok": True, "credential": credential_to_dict(credential)}


def op_sigv4_sign(msg: JsonObject) -> JsonObject:
    """PROTOCOL.md sigv4_sign: the AWS test-suite vectors (auth/sigv4-vectors.json)."""
    from .cloud import sigv4
    from .credentials import credential_from_dict, parse_rfc3339

    credential = credential_from_dict(msg["credential"])
    req = msg["request"]
    headers = {str(k): (v if isinstance(v, str) else ",".join(v)) for k, v in dict(req.get("headers") or {}).items()
               if str(k).lower() not in ("host", "x-amz-date", "x-amz-security-token")}
    signature = sigv4.sign(method=str(req["method"]), url=str(req["url"]), headers=headers,
                           payload=str(req.get("body") or "").encode("utf-8"), credentials=credential,
                           region=str(msg["region"]), service=str(msg["service"]), now=parse_rfc3339(str(msg["now"])))
    return {"canonical_request": signature.canonical_request, "string_to_sign": signature.string_to_sign,
            "authorization": signature.authorization, "headers": signature.headers}


def op_surface_dump(msg: JsonObject) -> JsonObject:
    return {
        "types": _reflect_types(),
        "enums": _reflect_enums(),
        "providers": _reflect_providers(),
    }


def _reflect_providers() -> JsonObject:
    """Every registered provider's access policy: who supports what, by reflection.

    This is the support matrix the contract pins (spec/support-matrix.json):
    a port that silently disagrees about endpoint support fails the audit,
    not a user at runtime.
    """
    from .registry import PROVIDERS

    out: JsonObject = {}
    for provider in sorted(PROVIDERS):
        manifest = PROVIDERS[provider].access
        supports = manifest.supports
        out[provider] = {
            "supports": {
                f.name: sorted(getattr(supports, f.name)) if f.name == "extra" else getattr(supports, f.name)
                for f in dataclasses.fields(supports)
            },
            "auth_modes": list(manifest.auth_modes),
            "env_keys": list(manifest.env_keys),
        }
    return out


def _reflect_types() -> JsonObject:
    """Every public dataclass in lm15.types, by reflection."""
    out: JsonObject = {}
    for name in sorted(vars(lm15_types)):
        obj = getattr(lm15_types, name)
        if name.startswith("_") or not isinstance(obj, type):
            continue
        if obj.__module__ != lm15_types.__name__ or not dataclasses.is_dataclass(obj):
            continue
        out[obj.__name__] = {"fields": [f.name for f in dataclasses.fields(obj)]}
    return out


def _reflect_enums() -> JsonObject:
    """Every string vocabulary in lm15.types, by reflection.

    Harvests Literal type aliases (Role, FinishReason, …) and module-level
    string collections (FINISH_REASONS, PART_TYPES, …) — never a
    hand-maintained list. Unordered collections are sorted; Literal aliases
    keep declaration order.
    """
    from . import credentials as lm15_credentials
    from . import features as lm15_features

    out: JsonObject = {}
    # lm15.types, plus the auth vocabularies (spec/vocabularies.md AuthScheme,
    # CredentialKind, CredentialPolicy, StreamFraming, ModelPlacement; 2026-09-03).
    for module in (lm15_types, lm15_features, lm15_credentials):
        for name in sorted(vars(module)):
            if name.startswith("_") or name in ("AuthHeader",):
                continue
            vocabulary = _string_vocabulary(getattr(module, name))
            if vocabulary:
                out[name] = vocabulary
    return out


def _string_vocabulary(obj: Any) -> list[str] | None:
    if get_origin(obj) is Literal:
        args = get_args(obj)
        if args and all(isinstance(a, str) for a in args):
            return list(args)
        return None
    if isinstance(obj, (frozenset, set)):
        if obj and all(isinstance(v, str) for v in obj):
            return sorted(obj)
        return None
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) for k in obj) and all(isinstance(v, type) for v in obj.values()):
            return sorted(obj)
        return None
    if isinstance(obj, (tuple, list)):
        if obj and all(isinstance(v, str) for v in obj):
            return sorted(obj)
        return None
    return None


# ─── Framing ─────────────────────────────────────────────────────────

HANDLERS: dict[str, Callable[[JsonObject], JsonObject]] = {
    "capabilities": op_capabilities,
    "build_request": op_build_request,
    "ingest_openai_chat": op_ingest_openai_chat,
    "parse_response": op_parse_response,
    "replay_stream": op_replay_stream,
    "normalize_error": op_normalize_error,
    "serde_roundtrip": op_serde_roundtrip,
    "validate": op_validate,
    "surface_dump": op_surface_dump,
    "explain_auth": op_explain_auth,
    "resolve_model": op_resolve_model,
    "token_exchange_build": op_token_exchange_build,
    "token_exchange_parse": op_token_exchange_parse,
    "sigv4_sign": op_sigv4_sign,
    "build_models_request": op_build_models_request,
    "parse_models_response": op_parse_models_response,
    "replay_live": op_replay_live,
    "generation_build": op_generation_build,
    "generation_parse": op_generation_parse,
    "file_op_build": op_file_op_build,
    "file_op_parse": op_file_op_parse,
    "video_op_build": op_video_op_build,
    "video_op_parse": op_video_op_parse,
    "batch_op_build": op_batch_op_build,
    "batch_op_parse": op_batch_op_parse,
    "cache_op_build": op_cache_op_build,
    "cache_op_parse": op_cache_op_parse,
}


class _OpFailure(Exception):
    """An op failed with a typed error plus extra fields for the error reply."""

    def __init__(self, cause: BaseException, extra: JsonObject) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.extra = extra


def _error_reply(req_id: Any, exc: BaseException) -> JsonObject:
    # lm15-typed errors report the canonical class name and ErrorCode;
    # unexpected exceptions report the native exception name. A
    # StreamAssemblyError also carries the partial Response it salvaged
    # (MAP-9), so a golden can pin a raise and what survived it.
    extra: JsonObject = {}
    if isinstance(exc, _OpFailure):
        extra = exc.extra
        exc = exc.cause
    error: JsonObject = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, LM15Error) and exc.code is not None:
        error["code"] = exc.code
    if isinstance(exc, StreamAssemblyError) and exc.partial is not None:
        error["partial_response"] = _response_result(exc.partial)["canonical_response"]
    if isinstance(exc, (UnknownModelError, AmbiguousModelError)):
        # The payload spec/vocabularies.md pins for the router codes.
        error["model"] = exc.model
        if isinstance(exc, AmbiguousModelError):
            error["providers"] = list(exc.providers)
    error.update(extra)
    return {"id": req_id, "ok": False, "error": error}


def handle_line(line: str) -> JsonObject:
    try:
        msg = json.loads(line)
    except Exception as exc:
        return _error_reply(None, exc)

    if not isinstance(msg, dict):
        return _error_reply(None, ValueError("request must be a JSON object"))

    req_id = msg.get("id")
    try:
        op = msg.get("op")
        handler = HANDLERS.get(str(op))
        if handler is None:
            raise ValueError(f"unknown op: {op}")
        result = handler(msg)
    except Exception as exc:
        return _error_reply(req_id, exc)
    return {"id": req_id, "ok": True, "result": result}


def main(argv: list[str] | None = None) -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle_line(line)
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
