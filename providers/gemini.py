from __future__ import annotations

from datetime import datetime

import base64
import json
import os
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterator, Mapping

from ..errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    canonical_error_code,
    map_http_error,
)
from ..access import GEMINI_API
from ..features import ProviderManifest
from ..live import WebSocketLiveSession, require_websocket_sync_connect
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    CacheInfo,
    CachePage,
    VideoGenerationRequest,
    VideoJobInfo,
    VideoPart,
    AudioDelta,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    AudioPart,
    BatchEntry,
    BatchJobInfo,
    BatchRequest,
    ContinuationDelta,
    ContinuationState,
    BinaryPart,
    BuiltinTool,
    CitationPart,
    Config,
    DocumentPart,
    ErrorDetail,
    FileInfo,
    FilePage,
    FileUploadRequest,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientImageEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveConfig,
    LiveServerAudioEvent,
    LiveServerErrorEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    LiveServerUsageEvent,
    Message,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    TokenLogprob,
    ToolCallDelta,
    ToolCallPart,
    ToolResultPart,
    TopLogprob,
    continuation_data,
    Usage,
)
from .base import (
    BaseProviderLM,
    Credential,
    HttpResponse,
    SyncTransport,
    batch_entry_http,
    batch_entry_request,
    default_transport,
    resolve_credential,
)
from .common import EFFORT_THINKING_BUDGETS, MEDIA_KINDS, build_url, iso_utc, model_infos_from_entries, multipart_related_body, parts_to_text, path_id, unnamed_tool_call_error

# Canonical builtin tool name → Gemini tool key
_GEMINI_BUILTIN_MAP: dict[str, str] = {
    "web_search": "googleSearch",
    "code_execution": "codeExecution",
}

GEMINI_PROVIDER_EXECUTED_PART_KEYS = {
    "executableCode",
    "codeExecutionResult",
}

def gemini_level_class(model: str) -> bool:
    """True for the Gemini 3.x class: `thinkingLevel`, no full off (MAP-7 rule 10).

    Model-name table, receipted 2026-09-02: 2.5 models reject thinkingLevel,
    3.x models take it; 3.7 Flash accepts thinkingBudget: 0 and still spends
    thinking tokens.  A table that rots; `extensions` overrides.
    """
    lowered = model.lower()
    return lowered.startswith("models/gemini-3") or lowered.startswith("gemini-3")




def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_gemini(tool: BuiltinTool) -> dict[str, Any]:
    return {_GEMINI_BUILTIN_MAP.get(tool.name, tool.name): tool.config or {}}


def _gemini_token_logprobs(logprobs_result: Any) -> tuple[TokenLogprob, ...]:
    """Map Gemini logprobsResult to canonical TokenLogprob tuples.

    Doc-based mapping (generate-content reference — LogprobsResult /
    TopCandidates / Candidate): chosenCandidates[i] pairs with
    topCandidates[i] per decoding step. No live capture exists because
    every currently served model rejects responseLogprobs (2026-09-01).
    Derived aggregates (logProbabilitySum, avgLogprobs) stay in
    provider_data — they are computable from the per-token values.
    """
    if not isinstance(logprobs_result, dict):
        return ()
    chosen = logprobs_result.get("chosenCandidates") or []
    top_steps = logprobs_result.get("topCandidates") or []
    out: list[TokenLogprob] = []
    for i, cand in enumerate(chosen):
        if not isinstance(cand, dict):
            continue
        top: list[TopLogprob] = []
        step = top_steps[i] if i < len(top_steps) and isinstance(top_steps[i], dict) else {}
        for alt in step.get("candidates") or []:
            if not isinstance(alt, dict):
                continue
            top.append(
                TopLogprob(
                    token=str(alt.get("token") or ""),
                    logprob=float(alt.get("logProbability", 0.0) or 0.0),
                    token_id=alt.get("tokenId"),
                )
            )
        out.append(
            TokenLogprob(
                token=str(cand.get("token") or ""),
                logprob=float(cand.get("logProbability", 0.0) or 0.0),
                token_id=cand.get("tokenId"),
                top=tuple(top),
            )
        )
    return tuple(out)


def _gemini_schema_field(schema: dict[str, Any]) -> str:
    """Pick Gemini's schema field for an lm15 JSON schema.

    ``responseSchema`` is Gemini's OpenAPI-ish schema type and rejects JSON
    Schema keywords such as ``additionalProperties``.  ``responseJsonSchema``
    accepts those keywords, so use it when the schema needs full JSON Schema.
    """
    return "responseJsonSchema" if _contains_key(schema, "additionalProperties") else "responseSchema"


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


def _gemini_number(value: float) -> float | int:
    """Gemini wire dialect: integral float knobs are sent in integer form.

    The canonical model declares temperature/top_p as float fields (Number
    rule, docs/serde-rules.md), but Gemini's proto3-JSON wire form for
    integral doubles is the integer digits (live capture:
    lm15-contract/cases/gemini/temperature.json sends ``"temperature": 1``).
    This is a provider wire-dialect mapping, not a canonical form.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _response_format_to_gemini_config(format_config: dict[str, Any]) -> dict[str, Any]:
    """Canonical response_format (INV-050) -> Gemini generationConfig.

    `responseJsonSchema` accepts JSON Schema keywords; `responseSchema` is
    the OpenAPI subset — `_gemini_schema_field` picks by the presence of
    `additionalProperties` (pinned by gemini.response_schema and the
    2026-09-02 receipts).  `strict` is satisfied (always constrained);
    `name` is a label with no slot.
    """
    if format_config["type"] == "json_object":
        return {"responseMimeType": "application/json"}
    schema = format_config["schema"]
    return {"responseMimeType": "application/json", _gemini_schema_field(schema): schema}


def _gemini_usage(usage_payload: Any, *, output_keys: tuple[str, ...]) -> Usage:
    """Map a Gemini ``usageMetadata`` object to canonical Usage (INV-029).

    Gemini's proto3-JSON wire omits zero-valued fields, so inside a present
    ``usageMetadata`` an absent primary counter means "0", not "not
    reported" (pinned: lm15-contract goldens/gemini/max_output_tokens.json,
    where ``candidatesTokenCount`` is absent and ``totalTokenCount`` equals
    prompt + thoughts).  When ``usageMetadata`` itself is missing nothing
    was reported and every counter stays None.  Secondary counters are
    conditional on a feature (cache, thinking) and stay verbatim: absent is
    "not reported".  ``output_keys`` lists the wire names for output tokens
    in priority order (``candidatesTokenCount`` on generateContent,
    ``responseTokenCount`` on Live).
    """
    if not isinstance(usage_payload, dict) or not usage_payload:
        return Usage()
    output_tokens: Any = 0
    for key in output_keys:
        if key in usage_payload:
            output_tokens = usage_payload[key]
            break
    return Usage(
        input_tokens=usage_payload.get("promptTokenCount", 0),
        output_tokens=output_tokens,
        total_tokens=usage_payload.get("totalTokenCount"),
        cache_read_tokens=usage_payload.get("cachedContentTokenCount"),
        reasoning_tokens=usage_payload.get("thoughtsTokenCount"),
        # Modality breakdowns: {modality, tokenCount} entries. AUDIO has a
        # canonical slot on both sides (OpenAI fills the same slots); IMAGE
        # and VIDEO do not and stay in provider_data. Independent review
        # 2026-09-02: three goldens carried audio counts nowhere.
        input_audio_tokens=_modality_tokens(usage_payload.get("promptTokensDetails"), "AUDIO"),
        output_audio_tokens=_modality_tokens(
            usage_payload.get("candidatesTokensDetails") or usage_payload.get("responseTokensDetails"), "AUDIO"
        ),
    )


def _modality_tokens(details: Any, modality: str) -> int | None:
    """Sum of ``tokenCount`` over the entries for ``modality``; None when the
    breakdown has no such entry (not reported, per INV-029)."""
    if not isinstance(details, list):
        return None
    counts = [e.get("tokenCount", 0) for e in details if isinstance(e, dict) and e.get("modality") == modality]
    return sum(counts) if counts else None


def _thought_signature_state(part: dict[str, Any]) -> tuple[ContinuationState, ...]:
    """The part's ``thoughtSignature`` as canonical replay state, or ()."""
    signature = part.get("thoughtSignature")
    if signature is None:
        return ()
    return (ContinuationState(provider="gemini", kind="thought_signature", data={"value": str(signature)}),)


def _finish_reason(reason: str | None, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    r = str(reason or "").upper()
    if r == "MAX_TOKENS":
        return "length"
    if r in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
        return "content_filter"
    return "stop"


def _gemini_batch_status(data: dict[str, Any]) -> str:
    """Map a Gemini batch operation to the canonical BatchStatus.

    States observed live 2026-08-31: BATCH_STATE_PENDING / RUNNING /
    SUCCEEDED (wire fact — the docs' JOB_STATE_* naming is wrong for
    this endpoint).
    """
    state = str((data.get("metadata") or {}).get("state") or "").upper()
    mapping = {
        "BATCH_STATE_PENDING": "queued",
        "BATCH_STATE_RUNNING": "running",
        "BATCH_STATE_CANCELLING": "cancelling",
        "BATCH_STATE_SUCCEEDED": "completed",
        "BATCH_STATE_FAILED": "failed",
        "BATCH_STATE_CANCELLED": "cancelled",
        "BATCH_STATE_EXPIRED": "expired",
    }
    if state in mapping:
        return mapping[state]
    return "completed" if data.get("done") else "queued"


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gemini_segment_text(segment: dict[str, Any], full_text: str) -> str | None:
    text = segment.get("text")
    if isinstance(text, str) and text:
        return text
    start = _int_or_none(segment.get("startIndex"))
    end = _int_or_none(segment.get("endIndex"))
    if start is not None and end is not None and 0 <= start < end <= len(full_text):
        return full_text[start:end]
    return None


def _gemini_grounding_chunk(chunks: list[Any], index: Any) -> dict[str, Any]:
    idx = _int_or_none(index)
    if idx is None or idx < 0 or idx >= len(chunks):
        return {}
    chunk = chunks[idx]
    return chunk if isinstance(chunk, dict) else {}


def _gemini_citations(candidate: dict[str, Any], full_text: str) -> list[CitationPart]:
    grounding = candidate.get("groundingMetadata")
    if not isinstance(grounding, dict):
        return []

    chunks = grounding.get("groundingChunks") or []
    if not isinstance(chunks, list):
        chunks = []
    supports = grounding.get("groundingSupports") or []
    if not isinstance(supports, list):
        return []

    citations: list[CitationPart] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for support in supports:
        if not isinstance(support, dict):
            continue
        segment = support.get("segment") if isinstance(support.get("segment"), dict) else {}
        cited_text = _gemini_segment_text(segment, full_text)
        indices = support.get("groundingChunkIndices") or []
        if not isinstance(indices, list):
            continue
        for index in indices:
            chunk = _gemini_grounding_chunk(chunks, index)
            source = chunk.get("web") or chunk.get("retrievedContext") or chunk.get("googleSearch") or {}
            if not isinstance(source, dict):
                source = {}
            url = source.get("uri") or source.get("url")
            title = source.get("title") or source.get("name")
            url_s = str(url) if url else None
            title_s = str(title) if title else None
            key = (url_s, title_s, cited_text)
            if key in seen or (url_s is None and title_s is None and cited_text is None):
                continue
            seen.add(key)
            citations.append(CitationPart(url=url_s, title=title_s, text=cited_text))
    return citations


_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(slots=True)
class GeminiLM(BaseProviderLM):
    api_key: Credential | None = field(default=None, repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = _DEFAULT_BASE_URL
    upload_base_url: str = "https://generativelanguage.googleapis.com/upload/v1beta"
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="gemini", init=False)
    account_id: str | None = field(default=None, init=False, repr=False)
    manifest: ClassVar[ProviderManifest] = GEMINI_API
    _api_key_header: ClassVar[str] = "x-goog-api-key"

    def __post_init__(self) -> None:
        self._bind_access(self.access, credentials_path=self.credentials_path, default_base_url=_DEFAULT_BASE_URL, settings=self.settings)

    _error_status_map: ClassVar[dict[str, type[ProviderError]]] = {
        "INVALID_ARGUMENT": InvalidRequestError,
        "FAILED_PRECONDITION": BillingError,
        "PERMISSION_DENIED": AuthError,
        "UNAUTHENTICATED": AuthError,
        "NOT_FOUND": InvalidRequestError,
        "RESOURCE_EXHAUSTED": RateLimitError,
        "INTERNAL": ServerError,
        "UNAVAILABLE": ServerError,
        "DEADLINE_EXCEEDED": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        lowered = msg.lower()
        return (
            ("token" in lowered and ("limit" in lowered or "exceed" in lowered))
            or "too long" in lowered
            or "context is too long" in lowered
            or "context length" in lowered
        )

    @staticmethod
    def _is_model_error(message: str) -> bool:
        lowered = message.lower()
        return "model" in lowered and any(
            marker in lowered
            for marker in (
                "not found",
                "does not exist",
                "not exist",
                "not supported",
                "unsupported",
                "not available",
                "unknown",
            )
        )

    @staticmethod
    def _is_candidate_finish_error(finish_reason: str) -> bool:
        return finish_reason in {
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "MALFORMED_FUNCTION_CALL",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_OTHER",
            "NO_IMAGE",
            "IMAGE_RECITATION",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
            "MALFORMED_RESPONSE",
        }

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_status_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        elif provider_code == "NOT_FOUND" and self._is_model_error(message):
            cls = UnsupportedModelError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def _inband_error(self, data: dict[str, Any]) -> ProviderError | None:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = str(prompt_feedback.get("blockReason") or "")
            if block_reason and block_reason != "BLOCK_REASON_UNSPECIFIED":
                return self._provider_error(
                    InvalidRequestError,
                    f"Prompt blocked: {block_reason}",
                    provider_code="promptFeedback",
                )

        candidate = (data.get("candidates") or [{}])[0]
        if isinstance(candidate, dict):
            finish_reason = str(candidate.get("finishReason") or "")
            if self._is_candidate_finish_error(finish_reason):
                finish_message = str(candidate.get("finishMessage") or "")
                return self._provider_error(
                    InvalidRequestError,
                    finish_message or f"Candidate blocked: {finish_reason}",
                    provider_code=finish_reason or "finishReason",
                )
        return None

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_status = str(err.get("status") or "") if isinstance(err, dict) else ""
            if self._is_context_length_message(msg):
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=err_status or None,
                )
            if err_status == "NOT_FOUND" and self._is_model_error(msg):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=err_status,
                )
            cls = self._error_status_map.get(err_status)
            if cls:
                return self._provider_error(
                    cls,
                    msg,
                    status=status,
                    provider_code=err_status or None,
                )
            if err_status and err_status not in msg:
                msg = f"{msg} ({err_status})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            err_status = ""
        return map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.access.env_keys,
            provider_code=err_status or None,
        )

    # ─── Request serialization ──────────────────────────────────────

    def _model_path(self, model: str) -> str:
        model = urllib.parse.quote(model, safe="/:@")
        return model if model.startswith("models/") else f"models/{model}"

    def _auth_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}  # auth is applied once, in _emit (AUTH-2)
        if extra:
            headers.update(extra)
        return headers

    def _auth_params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return dict(extra or {})

    def _function_response(self, part: ToolResultPart, names: dict[str, str] | None) -> dict[str, Any]:
        """`functionResponse` (MAP-10 on this wire): text in `response.result`
        (`response.error` when is_error), media nested under `parts[]` as
        inlineData/fileData in the caller's order; the name resolved from the
        transcript's matching functionCall when the caller gave none (rule 6).
        Every model takes the shape; Gemini 2.5 answers HTTP 400 itself."""
        text_parts = tuple(p for p in part.content if p.type not in MEDIA_KINDS)
        media_parts = [p for p in part.content if p.type in MEDIA_KINDS]
        for p in media_parts:
            if p.type not in ("image", "document"):
                raise UnsupportedFeatureError(
                    f"{self.provider}: a {p.type} part in tool_result {part.id!r} cannot reach a functionResponse — "
                    "multimodal function responses take images (png/jpeg/webp) and documents (pdf, text/plain) only (MAP-10)",
                    provider=self.provider,
                )
        name = part.name or (names or {}).get(part.id)
        if not name:
            raise UnsupportedFeatureError(
                f"{self.provider}: tool_result {part.id!r} needs a function name on the Gemini wire and no preceding "
                "assistant tool_call with that id is in the transcript; set ToolResultPart.name (MAP-10 rule 6)",
                provider=self.provider,
            )
        text = parts_to_text(text_parts, provider=self.provider, where="functionResponse.response")
        if part.is_error:
            response: dict[str, Any] = {"error": text}
        elif media_parts and not text_parts:
            response = {}  # the media IS the result; no fabricated text (live 2026-09-07: accepted)
        else:
            response = {"result": text}
        fr: dict[str, Any] = {"name": name, "response": response}
        if part.id:
            fr["id"] = part.id
        if media_parts:
            fr["parts"] = [self._part(p) for p in media_parts]
        return {"functionResponse": fr}

    def _part(self, part, names: dict[str, str] | None = None) -> dict[str, Any]:
        if isinstance(part, TextPart):
            out: dict[str, Any] = {"text": part.text}
            thought = continuation_data(part, "gemini", "thought_signature")
            if thought and thought.get("value"):
                out["thoughtSignature"] = thought["value"]
            return out
        if isinstance(part, (ImagePart, AudioPart, VideoPart, DocumentPart, BinaryPart)):
            mime = part.media_type or "application/octet-stream"
            if part.url is not None:
                return {"fileData": {"mimeType": mime, "fileUri": part.url}}
            if part.file_id is not None:
                return {"fileData": {"mimeType": mime, "fileUri": part.file_id}}
            if part.data is not None:
                return {"inlineData": {"mimeType": mime, "data": part.data}}
            if part.path is not None:
                return {"inlineData": {"mimeType": mime, "data": base64.b64encode(part.path.read_bytes()).decode("ascii")}}
        if isinstance(part, ToolCallPart):
            out: dict[str, Any] = {"functionCall": {"name": part.name, "args": part.input}}
            if part.id:
                out["functionCall"]["id"] = part.id
            thought = continuation_data(part, "gemini", "thought_signature")
            if thought and thought.get("value"):
                out["thoughtSignature"] = thought["value"]
            return out
        if isinstance(part, ToolResultPart):
            return self._function_response(part, names)
        if isinstance(part, ThinkingPart):
            out: dict[str, Any] = {"text": part.text}
            thought = continuation_data(part, "gemini", "thought_signature")
            if thought and thought.get("value"):
                out["thought"] = True
                out["thoughtSignature"] = thought["value"]
            return out
        return {"text": getattr(part, "text", "") or ""}

    def _message(self, msg: Message, names: dict[str, str] | None = None) -> dict[str, Any]:
        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "developer":
            return {"role": "user", "parts": [{"text": f"[developer]\n{parts_to_text(msg.parts, provider=self.provider, where='a developer turn')}"}]}
        return {"role": role, "parts": [self._part(part, names) for part in msg.parts]}

    @staticmethod
    def _call_names(messages) -> dict[str, str]:
        """tool_call id → function name, over the transcript (MAP-10 rule 6)."""
        names: dict[str, str] = {}
        for m in messages:
            for p in m.parts:
                if isinstance(p, ToolCallPart) and p.id:
                    names[p.id] = p.name
        return names

    def _tool_config_payload(self, request: Request) -> dict[str, Any] | None:
        tc = request.config.tool_choice
        if tc is None:
            return None
        if tc.parallel is False:
            # MAP-8 rule 2 (live 2026-09-02): no wire knob; two calls came back
            # on 2.5 and 3.7 with the preference set.  The outcome is not
            # observable from usage, so the MAP-6 fallback exception does not
            # apply — raise.
            raise UnsupportedFeatureError(
                "gemini: tool_choice.parallel=False is not supported — GenerateContent has no "
                "parallel-tool-calls knob and returns several calls regardless (OpenAI and "
                "Anthropic carry it)",
                provider=self.provider,
            )
        mode = {"none": "NONE", "required": "ANY", "auto": "AUTO"}[tc.mode]
        cfg: dict[str, Any] = {"mode": mode}
        if tc.allowed:
            by_name = {t.name: t for t in request.tools}
            builtins = [name for name in tc.allowed if isinstance(by_name.get(name), BuiltinTool)]
            if builtins:
                raise UnsupportedFeatureError(
                    f"gemini: cannot force builtin tools {builtins} — "
                    "functionCallingConfig addresses function declarations only; "
                    "googleSearch/codeExecution have no tool_choice form "
                    "(OpenAI Responses and Anthropic carry builtin forcing)",
                    provider=self.provider,
                )
            cfg["allowedFunctionNames"] = list(tc.allowed)
            if tc.mode == "auto":
                # allowedFunctionNames is only legal with ANY or VALIDATED.
                # VALIDATED = "function call or text, restricted to the
                # allowlist" — exactly canonical mode=auto + allowed.
                cfg["mode"] = "VALIDATED"
        return {"functionCallingConfig": cfg}

    def _payload(self, request: Request) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        cache_cfg = request.config.cache
        # MAP-6 on Gemini.  Automatic tier: nothing to send; prefix intents
        # fall back to it (no cost, visible in usageMetadata).  Resource
        # tier: `cachedContent` names a stored object that already holds
        # system, tools, and the prefix messages, so the wire carries only
        # the suffix — the server rejects system/tools next to a cache.
        # key / retention name mechanisms this wire does not have.
        resource: str | None = None
        suffix_from = 0
        if cache_cfg is not None and cache_cfg.mode != "off":
            if cache_cfg.key is not None:
                raise UnsupportedFeatureError(
                    "gemini: cache.key is not supported — GenerateContent has no cache "
                    "affinity key; use cache.resource with a stored cache (lm.cache(prefix))",
                    provider=self.provider,
                )
            if cache_cfg.retention is not None and cache_cfg.retention != "short":
                raise UnsupportedFeatureError(
                    "gemini: cache.retention is not supported in-request — lifetime belongs to "
                    "the stored cache (cache_create(..., ttl_seconds=...) / cache_update)",
                    provider=self.provider,
                )
            if cache_cfg.resource is not None:
                resource = cache_cfg.resource
                if cache_cfg.prefix_until_index is not None:
                    suffix_from = min(cache_cfg.prefix_until_index, len(request.messages) - 1) + 1
        wire_messages = request.messages[suffix_from:]
        if resource is not None and not wire_messages:
            raise ValueError("gemini: a request against a stored cache needs at least one message after the prefix")

        names = self._call_names(request.messages)
        payload: dict[str, Any] = {"contents": [self._message(m, names) for m in wire_messages]}
        if resource is not None:
            payload["cachedContent"] = self._cache_resource(resource)
        if request.system and resource is None:
            text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            payload["systemInstruction"] = {"parts": [{"text": text}]}

        generation_config: dict[str, Any] = {}
        if request.config.temperature is not None:
            generation_config["temperature"] = _gemini_number(request.config.temperature)
        if request.config.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.config.max_tokens
        if request.config.top_p is not None:
            generation_config["topP"] = _gemini_number(request.config.top_p)
        if request.config.top_k is not None:
            generation_config["topK"] = request.config.top_k
        if request.config.stop:
            generation_config["stopSequences"] = list(request.config.stop)
        if request.config.logprobs is not None:
            # Documented wire knobs (generate-content reference). Live note
            # 2026-09-01: every currently served Gemini model rejects this
            # with "Logprobs is not enabled" — the mapping is doc-based and
            # the rejection surfaces as the provider's InvalidRequestError.
            generation_config["responseLogprobs"] = True
            if request.config.logprobs > 0:
                generation_config["logprobs"] = request.config.logprobs
        if request.config.response_format:
            generation_config.update(_response_format_to_gemini_config(request.config.response_format))
        if request.config.reasoning is not None:
            reasoning = request.config.reasoning
            level_class = gemini_level_class(request.model)
            if reasoning.is_off:
                if level_class:
                    # MAP-7 rule 4 / MAP-5: Gemini 3.x cannot fully disable
                    # thinking; 3.7 Flash accepts thinkingBudget 0 and still
                    # spent 58 tokens (live 2026-09-02) — a silent paid no-op.
                    raise UnsupportedFeatureError(
                        f"gemini: reasoning cannot be disabled on {request.model} — the Gemini 3 "
                        "class has no full off switch (thinkingBudget 0 is accepted but not honoured); "
                        "use effort='low' or a 2.5 model",
                        provider=self.provider,
                    )
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
            else:
                if reasoning.summary in ("concise", "detailed"):
                    raise UnsupportedFeatureError(
                        f"gemini: reasoning.summary={reasoning.summary!r} is an OpenAI detail level; "
                        "GenerateContent has includeThoughts only (use 'auto')",
                        provider=self.provider,
                    )
                thinking: dict[str, Any] = {}
                if reasoning.summary is not None:
                    thinking["includeThoughts"] = True  # MAP-7 rule 7: only when asked
                if reasoning.thinking_budget is not None:
                    thinking["thinkingBudget"] = reasoning.thinking_budget  # 3.x: accepted, docs warn
                elif level_class:
                    if reasoning.effort in ("xhigh", "max"):
                        raise UnsupportedFeatureError(
                            f"gemini: reasoning.effort={reasoning.effort!r} has no thinkingLevel on the "
                            "Gemini 3 class (minimal|low|medium|high); 'high' is the ceiling",
                            provider=self.provider,
                        )
                    thinking["thinkingLevel"] = reasoning.effort
                else:
                    thinking["thinkingBudget"] = EFFORT_THINKING_BUDGETS[reasoning.effort]
                generation_config["thinkingConfig"] = thinking
        if generation_config:
            payload["generationConfig"] = generation_config

        if request.tools and resource is None:
            function_declarations = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools
                if isinstance(t, FunctionTool)
            ]
            tools_wire: list[dict[str, Any]] = []
            if function_declarations:
                tools_wire.append({"functionDeclarations": function_declarations})
            for tool in request.tools:
                if isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_gemini(tool))
            payload["tools"] = tools_wire

        tool_config = self._tool_config_payload(request) if resource is None else None
        if tool_config is not None:
            payload["toolConfig"] = tool_config

        output = extensions.get("output")
        if output == "image":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]
        elif output == "audio":
            payload.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]

        # Promoted cross-provider knobs (changes/2026-09-01-extensions-burn-down):
        # store maps verbatim (same wire key as OpenAI, verified live).
        # service_tier maps to the top-level `serviceTier` enum
        # (unspecified|standard|flex|priority) — added to GenerateContent
        # between the April and September 2026 doc snapshots and verified
        # live 2026-09-01 (accepted + echoed in usageMetadata.serviceTier;
        # an unknown value is rejected with HTTP 400 INVALID_ARGUMENT, so
        # the server validates the field).  user_id has no Gemini wire
        # field — raise, never silently drop.
        if request.config.store is not None:
            payload["store"] = request.config.store
        if request.config.service_tier is not None:
            payload["serviceTier"] = request.config.service_tier
        if request.config.user_id is not None:
            raise UnsupportedFeatureError(
                "gemini: config.user_id is not supported — GenerateContent has no "
                "end-user attribution field (OpenAI and Anthropic carry it)",
                provider=self.provider,
            )

        if extensions:
            passthrough = {k: v for k, v in extensions.items() if k not in {"prompt_caching", "output"}}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        endpoint = "streamGenerateContent" if stream else "generateContent"
        params = self._auth_params({"alt": "sse"} if stream else None)
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/{self._model_path(request.model)}:{endpoint}",
            endpoint="generateContent",
            stream=stream,
            model=request.model,
            headers=self._auth_headers({"Content-Type": "application/json"}),
            params=params,
            payload=self._payload(request),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def _parse_candidate_parts(
        self,
        parts_payload: list[dict[str, Any]],
        *,
        unmapped: list[dict[str, str]] | None = None,
        path_prefix: str = "parts",
    ) -> list[Any]:
        parts: list[Any] = []
        for part_index, part in enumerate(parts_payload):
            if not isinstance(part, dict):
                if unmapped is not None:
                    _record_unmapped(unmapped, f"{path_prefix}[{part_index}]", type(part).__name__)
                continue
            if part.get("thought") and "text" in part:
                # A thought part is classified by its flag, not by non-empty
                # text: 3.x can send {thought: true, text: "", thoughtSignature}
                # and the signature is the replay state (MAP-7 rule 8).
                parts.append(ThinkingPart(text=str(part.get("text") or ""), continuation=_thought_signature_state(part)))
            elif "text" in part:
                # On Gemini 3.x the final answer text carries the turn's
                # thoughtSignature (independent review 2026-09-02: dropped
                # silently before). It is replay state exactly as on a
                # thought or functionCall part, and goes back on the wire.
                parts.append(TextPart(text=str(part.get("text") or ""), continuation=_thought_signature_state(part)))
            elif "functionCall" in part and isinstance(part["functionCall"], dict):
                fc = part["functionCall"]
                continuation: tuple[ContinuationState, ...] = ()
                thought_signature = part.get("thoughtSignature") or fc.get("thoughtSignature")
                if thought_signature is not None:
                    continuation = (
                        ContinuationState(
                            provider="gemini",
                            kind="thought_signature",
                            data={"value": str(thought_signature)},
                        ),
                    )
                if not fc.get("name"):
                    raise unnamed_tool_call_error(self.provider, f"{path_prefix}[{part_index}]")
                parts.append(ToolCallPart(
                    # MAP-9: a missing call id is the lm15 correlator tool_call_<index>,
                    # the same one the stream assembler mints (INV-051 parity).
                    id=str(fc.get("id") or f"tool_call_{len(parts)}"),
                    name=str(fc["name"]),
                    input=fc.get("args") if isinstance(fc.get("args"), dict) else {},
                    continuation=continuation,
                ))
            elif "inlineData" in part and isinstance(part["inlineData"], dict):
                inline = part["inlineData"]
                mime = str(inline.get("mimeType") or "application/octet-stream")
                data = str(inline.get("data") or "")
                if not data:
                    continue
                if mime.startswith("image/"):
                    parts.append(ImagePart(media_type=mime, data=data))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(media_type=mime, data=data))
                else:
                    parts.append(DocumentPart(media_type=mime, data=data))
            elif "fileData" in part and isinstance(part["fileData"], dict):
                fd = part["fileData"]
                uri = str(fd.get("fileUri") or "")
                mime = str(fd.get("mimeType") or "application/octet-stream")
                if not uri:
                    continue
                if mime.startswith("image/"):
                    parts.append(ImagePart(media_type=mime, url=uri))
                elif mime.startswith("audio/"):
                    parts.append(AudioPart(media_type=mime, url=uri))
                else:
                    parts.append(DocumentPart(media_type=mime, url=uri))
            elif any(key in part for key in GEMINI_PROVIDER_EXECUTED_PART_KEYS):
                continue
            elif unmapped is not None:
                _record_unmapped(unmapped, f"{path_prefix}[{part_index}]", "+".join(sorted(part)) or "<empty>")
        return parts

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()
        inband = self._inband_error(data)
        if inband is not None:
            raise inband
        candidate = (data.get("candidates") or [{}])[0]
        candidate = candidate if isinstance(candidate, dict) else {}
        content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
        unmapped: list[dict[str, str]] = []
        parts = self._parse_candidate_parts(
            content.get("parts", []) or [],
            unmapped=unmapped,
            path_prefix="candidates[0].content.parts",
        )
        full_text = "".join(part.text for part in parts if isinstance(part, TextPart))
        parts.extend(_gemini_citations(candidate, full_text))
        if not parts:
            parts = [TextPart(text="")]
        usage = _gemini_usage(data.get("usageMetadata"), output_keys=("candidatesTokenCount", "responseTokenCount"))
        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        # D8 (2026-09-06): Response.id carries responseId; no message-level
        # continuation state is minted for it.
        logprob_seq = _gemini_token_logprobs(candidate.get("logprobsResult"))
        return Response(
            id=str(data.get("responseId")) if data.get("responseId") else None,
            model=request.model,
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=_finish_reason(candidate.get("finishReason"), has_tool_call=has_tool),
            usage=usage,
            logprobs=logprob_seq or None,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        if not raw_event.data:
            return
        payload = json.loads(raw_event.data)
        if not isinstance(payload, dict):
            return
        if "error" in payload:
            err = payload["error"]
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            yield StreamErrorEvent(error=self._error_detail(provider_code, message))
            return

        inband = self._inband_error(payload)
        if inband is not None:
            yield StreamErrorEvent(error=ErrorDetail(code=canonical_error_code(inband), provider_code="inband_finish_reason", message=str(inband)))
            return

        candidates = payload.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        yielded_delta = False
        saw_tool = False
        finish = None
        if candidate is not None:
            content = candidate.get("content", {}) if isinstance(candidate.get("content"), dict) else {}
            # Chunk-level decoding telemetry rides the chunk's first text
            # delta (doc-based; no model currently serves logprobs live).
            chunk_logprobs = _gemini_token_logprobs(candidate.get("logprobsResult"))
            for idx, part in enumerate(content.get("parts", []) or []):
                if not isinstance(part, dict):
                    continue
                if part.get("thought") and "text" in part:
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=ThinkingDelta(text=str(part.get("text") or ""), part_index=idx))
                    if part.get("thoughtSignature") is not None:
                        yield StreamDeltaEvent(
                            delta=ContinuationDelta(
                                provider="gemini",
                                kind="thought_signature",
                                data={"value": str(part.get("thoughtSignature"))},
                                part_index=idx,
                            )
                        )
                elif "text" in part:
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=TextDelta(text=str(part.get("text") or ""), part_index=idx, logprobs=chunk_logprobs))
                    chunk_logprobs = ()
                    if part.get("thoughtSignature") is not None:
                        # Same replay state on an answer-text part (3.x).
                        yield StreamDeltaEvent(
                            delta=ContinuationDelta(
                                provider="gemini",
                                kind="thought_signature",
                                data={"value": str(part.get("thoughtSignature"))},
                                part_index=idx,
                            )
                        )
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    saw_tool = True
                    yielded_delta = True
                    yield StreamDeltaEvent(delta=ToolCallDelta(
                        input=json.dumps(fc.get("args", {}), separators=(",", ":")),
                        part_index=idx,
                        id=str(fc.get("id") or "") or None,
                        name=str(fc.get("name") or "") or None,
                    ))
                    thought_signature = part.get("thoughtSignature") or fc.get("thoughtSignature")
                    if thought_signature is not None:
                        yield StreamDeltaEvent(
                            delta=ContinuationDelta(
                                provider="gemini",
                                kind="thought_signature",
                                data={"value": str(thought_signature)},
                                part_index=idx,
                            )
                        )
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "application/octet-stream")
                    data = str(inline.get("data") or "")
                    if mime.startswith("audio/"):
                        yielded_delta = True
                        yield StreamDeltaEvent(delta=AudioDelta(data=data, part_index=idx, media_type=mime))
                    elif mime.startswith("image/"):
                        yielded_delta = True
                        yield StreamDeltaEvent(delta=ImageDelta(data=data, part_index=idx, media_type=mime))
            finish = candidate.get("finishReason")

        if finish:
            yield StreamEndEvent(finish_reason=_finish_reason(finish, has_tool_call=saw_tool), usage=self._usage_from_payload(payload), provider_data=payload)
        elif not yielded_delta and "usageMetadata" in payload:
            yield StreamEndEvent(finish_reason="stop", usage=self._usage_from_payload(payload), provider_data=payload)

    def _usage_from_payload(self, payload: dict[str, Any]) -> Usage:
        return _gemini_usage(payload.get("usageMetadata"), output_keys=("candidatesTokenCount", "responseTokenCount"))

    # ─── Streaming via Gemini Live for live models ──────────────────

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        if self._should_use_live_completion(request):
            yield from self._stream_via_live_completion(request)
            return
        yield from BaseProviderLM.stream(self, request)

    def _should_use_live_completion(self, request: Request) -> bool:
        extensions = request.config.extensions or {}
        transport_mode = str(extensions.get("transport") or "").lower()
        if transport_mode in {"live", "websocket", "ws"}:
            return True
        model_name = request.model.lower()
        return "-live" in model_name or model_name.endswith("live")

    @staticmethod
    def _is_audio_native_live_model(model: str) -> bool:
        lowered = model.lower()
        return "live-preview" in lowered or "native-audio" in lowered

    @staticmethod
    def _wav_to_pcm(data: bytes) -> tuple[bytes, int]:
        if len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            sample_rate = struct.unpack_from("<I", data, 24)[0]
            pos = 12
            while pos + 8 <= len(data):
                chunk_id = data[pos : pos + 4]
                chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
                if chunk_id == b"data":
                    return data[pos + 8 : pos + 8 + chunk_size], sample_rate
                pos += 8 + chunk_size
            return data[44:], sample_rate
        return data, 16000

    def _stream_via_live_completion(self, request: Request) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url())
        saw_tool_call = False
        audio_native = self._is_audio_native_live_model(request.model)
        acc_usage = Usage()
        try:
            setup_payload = self._live_setup_payload_from_request(request)
            setup_inner = setup_payload.setdefault("setup", {})
            if not audio_native:
                setup_inner.setdefault("generationConfig", {}).setdefault("responseModalities", ["TEXT"])
            ws.send(json.dumps(setup_payload))
            self._wait_for_setup_complete(ws)
            for msg in self._live_client_content_payload_from_request(request):
                ws.send(json.dumps(msg))

            yield StreamStartEvent(model=request.model)
            while True:
                raw = ws.recv()
                events, turn_complete, usage = self._decode_live_completion_stream_events(raw)
                acc_usage = Usage(
                    input_tokens=max(acc_usage.input_tokens, usage.input_tokens),
                    output_tokens=max(acc_usage.output_tokens, usage.output_tokens),
                    total_tokens=max(acc_usage.total_tokens or 0, usage.total_tokens or 0),
                )
                for event in events:
                    if event.type == "delta" and isinstance(event.delta, ToolCallDelta):
                        saw_tool_call = True
                    if event.type == "error":
                        yield event
                        return
                    yield event
                if turn_complete:
                    yield StreamEndEvent(finish_reason="tool_call" if saw_tool_call else "stop", usage=acc_usage)
                    return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_setup_payload_from_request(self, request: Request) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        extensions.pop("transport", None)
        extensions.pop("prompt_caching", None)
        extensions.pop("output", None)
        config = LiveConfig(model=request.model, system=request.system, tools=request.tools, extensions=extensions or None)
        payload = self._live_setup_payload(config)
        output = (request.config.extensions or {}).get("output")
        audio_native = self._is_audio_native_live_model(request.model)
        if output == "audio" or audio_native:
            setup = payload.setdefault("setup", {})
            setup.setdefault("generationConfig", {})["responseModalities"] = ["AUDIO"]
            if output != "audio":
                setup["outputAudioTranscription"] = {}
            has_media = any(isinstance(p, (AudioPart, VideoPart)) for m in request.messages for p in m.parts)
            if has_media:
                setup.setdefault("realtimeInputConfig", {}).setdefault("automaticActivityDetection", {})["disabled"] = True
        elif output == "image":
            payload.setdefault("setup", {}).setdefault("generationConfig", {})["responseModalities"] = ["IMAGE"]
        return payload

    def _live_client_content_payload_from_request(self, request: Request) -> list[dict[str, Any]]:
        if self._is_audio_native_live_model(request.model):
            return self._build_realtime_input_payloads(request)
        if len(request.messages) == 1 and request.messages[0].role == "user" and all(isinstance(p, TextPart) for p in request.messages[0].parts):
            return [{"realtimeInput": {"text": parts_to_text(request.messages[0].parts)}}]
        names = self._call_names(request.messages)
        return [{"clientContent": {"turns": [self._message(m, names) for m in request.messages], "turnComplete": True}}]

    def _build_realtime_input_payloads(self, request: Request) -> list[dict[str, Any]]:
        text_payloads: list[dict[str, Any]] = []
        media_payloads: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []
        sent_audio_or_video = False
        for msg in request.messages:
            for part in msg.parts:
                if isinstance(part, TextPart) and part.text:
                    text_payloads.append({"realtimeInput": {"text": part.text}})
                elif isinstance(part, AudioPart):
                    if part.data is not None or part.path is not None:
                        mime = part.media_type or "audio/pcm"
                        raw = part.bytes
                        if "wav" in mime or "wave" in mime:
                            pcm, rate = self._wav_to_pcm(raw)
                            data = base64.b64encode(pcm).decode("ascii")
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": f"audio/pcm;rate={rate}", "data": data}}})
                        else:
                            data = part.data or base64.b64encode(raw).decode("ascii")
                            media_payloads.append({"realtimeInput": {"audio": {"mimeType": mime, "data": data}}})
                        sent_audio_or_video = True
                elif isinstance(part, VideoPart):
                    if part.data is not None or part.path is not None:
                        data = part.data or base64.b64encode(part.path.read_bytes()).decode("ascii")
                        media_payloads.append({"realtimeInput": {"video": {"mimeType": part.media_type or "video/mp4", "data": data}}})
                        sent_audio_or_video = True
                elif isinstance(part, (ImagePart, DocumentPart, BinaryPart)):
                    content_parts.append(self._part(part))
        payloads: list[dict[str, Any]] = []
        if content_parts:
            payloads.append({"clientContent": {"turns": [{"role": "user", "parts": content_parts}], "turnComplete": False}})
        payloads.extend(text_payloads + media_payloads)
        if sent_audio_or_video:
            payloads.insert(0, {"realtimeInput": {"activityStart": {}}})
            payloads.append({"realtimeInput": {"activityEnd": {}}})
        if not payloads:
            payloads.append({"realtimeInput": {"text": ""}})
        return payloads

    def _decode_live_completion_stream_events(self, raw: str | bytes) -> tuple[list[StreamEvent], bool, Usage]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return [], False, Usage()
        if not isinstance(payload, dict):
            return [], False, Usage()
        if "error" in payload:
            err = payload["error"]
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            return [StreamErrorEvent(error=self._error_detail(provider_code, message))], False, Usage()
        events: list[StreamEvent] = []
        tool_call = payload.get("toolCall")
        if isinstance(tool_call, dict):
            for idx, fc in enumerate(tool_call.get("functionCalls") or []):
                if isinstance(fc, dict):
                    events.append(StreamDeltaEvent(delta=ToolCallDelta(input=json.dumps(fc.get("args", {})), part_index=idx, id=str(fc.get("id") or f"fc_{idx}"), name=str(fc.get("name") or "tool"))))
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events, False, self._live_usage(payload, None)
        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for idx, part in enumerate(model_turn.get("parts", []) or []):
                if "text" in part:
                    events.append(StreamDeltaEvent(delta=TextDelta(text=str(part.get("text") or ""), part_index=idx)))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(StreamDeltaEvent(delta=ToolCallDelta(input=json.dumps(fc.get("args", {})), part_index=idx, id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"))))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    data = str(inline.get("data") or "")
                    if mime.startswith("audio/"):
                        events.append(StreamDeltaEvent(delta=AudioDelta(data=data, part_index=idx, media_type=mime)))
                    elif mime.startswith("image/"):
                        events.append(StreamDeltaEvent(delta=ImageDelta(data=data, part_index=idx, media_type=mime)))
        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(StreamDeltaEvent(delta=TextDelta(text=str(out_tx["text"]))))
        return events, bool(server.get("turnComplete")), self._live_usage(payload, server)

    # ─── Live sessions ──────────────────────────────────────────────

    def live(self, config: LiveConfig):
        self._require("live")
        ws = self._live_connect(self._live_url())
        for frame in self._live_setup_frames(config):
            ws.send(json.dumps(frame))
        self._wait_for_setup_complete(ws)
        return WebSocketLiveSession(
            ws=ws,
            encode_event=self._live_encoder(config),
            decode_event=self._decode_live_server_event,
        )

    def _live_connect(self, url: str):
        connect = require_websocket_sync_connect()
        return connect(url)

    # Pure pieces shared by the sync session, the native async twin, and
    # the vet shim's replay_live op (uniform hook: _live_setup_frames).

    def _live_setup_frames(self, config: LiveConfig) -> list[dict[str, Any]]:
        return [self._live_setup_frame(config)]

    def _live_setup_frame(self, config: LiveConfig) -> dict[str, Any]:
        payload = self._live_setup_payload(config)
        if self._is_audio_native_live_model(config.model):
            payload.setdefault("setup", {})["outputAudioTranscription"] = {}
        return payload

    def _live_encoder(self, config: LiveConfig):
        audio_native = self._is_audio_native_live_model(config.model)

        def encode_event(event: LiveClientEvent) -> list[dict[str, Any]]:
            if audio_native and isinstance(event, LiveClientTextEvent):
                return [{"realtimeInput": {"text": event.text}}]
            return self._encode_live_client_event(event)

        return encode_event

    def _live_setup_status(self, raw: str | bytes) -> bool:
        """True = setupComplete, False = keep waiting; raises typed on error."""
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return False
        if isinstance(payload, dict) and "setupComplete" in payload:
            return True
        if isinstance(payload, dict) and "error" in payload:
            err = payload["error"]
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            provider_code = str(err.get("status") or "live_setup") if isinstance(err, dict) else "live_setup"
            raise self._provider_error(
                InvalidRequestError,
                f"Live setup failed: {msg}",
                provider_code=provider_code,
            )
        return False

    def _wait_for_setup_complete(self, ws: Any) -> None:
        while not self._live_setup_status(ws.recv()):
            pass

    def _live_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        query = urllib.parse.urlencode({"key": resolve_credential(self.api_key)})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_setup_payload(self, config: LiveConfig) -> dict[str, Any]:
        setup: dict[str, Any] = {"model": self._model_path(config.model)}
        if config.system:
            setup["systemInstruction"] = {"parts": [{"text": config.system if isinstance(config.system, str) else parts_to_text(config.system)}]}
        function_tools = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in config.tools
            if isinstance(t, FunctionTool)
        ]
        if function_tools:
            setup["tools"] = [{"functionDeclarations": function_tools}]
        generation_config: dict[str, Any] = {}
        if config.output_format is not None:
            generation_config["responseModalities"] = ["AUDIO"]
        elif self._is_audio_native_live_model(config.model):
            generation_config["responseModalities"] = ["AUDIO"]
        if config.voice:
            generation_config.setdefault("speechConfig", {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}})
        if generation_config:
            setup["generationConfig"] = generation_config
        if config.extensions:
            setup.update(config.extensions)
        return {"setup": setup}

    def _live_usage(self, payload: dict[str, Any], server: dict[str, Any] | None) -> Usage:
        usage_payload = payload.get("usageMetadata")
        if not isinstance(usage_payload, dict) and isinstance(server, dict):
            usage_payload = server.get("usageMetadata")
        return _gemini_usage(usage_payload, output_keys=("responseTokenCount", "candidatesTokenCount"))

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if isinstance(event, LiveClientTurnEvent):
            return [{"clientContent": {"turns": [{"role": "user", "parts": [self._part(part) for part in event.parts]}], "turnComplete": event.turn_complete}}]
        if isinstance(event, LiveClientAudioEvent):
            return [{"realtimeInput": {"audio": {"mimeType": event.media_type, "data": event.data}}}]
        if isinstance(event, LiveClientImageEvent):
            return [{"realtimeInput": {"video": {"mimeType": event.media_type, "data": event.data}}}]
        if isinstance(event, LiveClientInterruptEvent):
            return [{"clientContent": {"turnComplete": True}}]
        if isinstance(event, LiveClientEndAudioEvent):
            return [{"realtimeInput": {"audioStreamEnd": True}}]
        if isinstance(event, LiveClientTextEvent):
            return [{"clientContent": {"turns": [{"role": "user", "parts": [{"text": event.text}]}], "turnComplete": True}}]
        if isinstance(event, LiveClientToolResultEvent):
            response_parts = [{"text": parts_to_text(event.content)}]
            return [{"toolResponse": {"functionResponses": [{"id": event.id, "response": {"output": response_parts}}]}}]
        return []

    def _decode_live_server_event(self, raw: str | bytes):
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        if "error" in payload:
            err = payload.get("error")
            provider_code = str(err.get("status") or err.get("code") or "provider") if isinstance(err, dict) else "provider"
            message = str(err.get("message") or "") if isinstance(err, dict) else ""
            return [LiveServerErrorEvent(error=self._error_detail(provider_code, message))]
        events: list[Any] = []
        tool_call = payload.get("toolCall")
        if isinstance(tool_call, dict):
            for fc in tool_call.get("functionCalls") or []:
                if isinstance(fc, dict):
                    events.append(LiveServerToolCallEvent(id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"), input=fc.get("args") if isinstance(fc.get("args"), dict) else {}))
        server = payload.get("serverContent")
        if not isinstance(server, dict):
            return events
        model_turn = server.get("modelTurn", {})
        if isinstance(model_turn, dict):
            for part in model_turn.get("parts", []) or []:
                if "text" in part:
                    events.append(LiveServerTextEvent(text=str(part.get("text") or "")))
                elif "inlineData" in part and isinstance(part["inlineData"], dict):
                    inline = part["inlineData"]
                    mime = str(inline.get("mimeType") or "")
                    if mime.startswith("audio/"):
                        events.append(LiveServerAudioEvent(data=str(inline.get("data") or ""), media_type=mime or None))
                elif "functionCall" in part and isinstance(part["functionCall"], dict):
                    fc = part["functionCall"]
                    events.append(LiveServerToolCallEvent(id=str(fc.get("id") or "fc_0"), name=str(fc.get("name") or "tool"), input=fc.get("args") if isinstance(fc.get("args"), dict) else {}))
        out_tx = server.get("outputTranscription")
        if isinstance(out_tx, dict) and out_tx.get("text"):
            events.append(LiveServerTextEvent(text=str(out_tx["text"])))
        has_usage = isinstance(payload.get("usageMetadata"), dict) or isinstance(server.get("usageMetadata"), dict)
        if has_usage and not server.get("turnComplete"):
            # Usage reported on a frame that does not end the turn (an
            # interrupted or tool-call frame) rides a usage event; the
            # pinned Gemini transcripts report usage only on turnComplete,
            # so this is the rule, not a receipt.
            events.append(LiveServerUsageEvent(usage=self._live_usage(payload, server)))
        if server.get("interrupted"):
            events.append(LiveServerInterruptedEvent())
        if server.get("turnComplete"):
            events.append(LiveServerTurnEndEvent(usage=self._live_usage(payload, server)))
        return events

    # ─── Other endpoints ────────────────────────────────────────────

    # ─── Live model listing (provisional endpoint) ──────────────────────

    def _models_request(self):
        # pageSize=1000 covers the catalog in one page (53 models observed
        # live 2026-08-31, no nextPageToken).
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/models",
            params={"pageSize": 1000},
            headers=self._auth_headers(),
            read_timeout=30.0,
        )

    def _models_from_body(self, body: str):
        data = json.loads(body)
        entries = data.get("models") if isinstance(data, dict) else None

        def id_of(entry: dict) -> str | None:
            # The wire name is "models/<id>"; the usable Request.model string
            # is the bare id (build_request re-prefixes via _model_path).
            name = entry.get("name")
            if isinstance(name, str) and name.startswith("models/"):
                return name[len("models/"):]
            return name if isinstance(name, str) else None

        return model_infos_from_entries(
            entries,
            provider=self.provider,
            api_family="gemini_generate_content",
            id_of=id_of,
        )

    # ─── File hooks (Files API) ─────────────────────────────────
    #
    # Wire shapes verified live 2026-08-31 (curl-fixtures/files-2026-08-31/):
    # multipart/related upload (metadata display_name + media) so the
    # filename survives the round trip; upload wraps the object in
    # {"file": {...}} while get/list return it bare; list paginates with
    # pageToken; download (`:download?alt=media`) EXISTS but the server
    # refuses non-generated files (400, forwarded typed).
    #
    # FileInfo.id is the file's `uri` VERBATIM — Gemini model requests
    # address files by URI (the frozen chat mapping places Part.file_id
    # into fileData.fileUri), while the REST resource lives at the
    # `files/<id>` name; _file_resource() derives one from the other.

    @staticmethod
    def _file_resource(file_id: str) -> str:
        """`files/<id>` resource path from a canonical id (URI or name)."""
        if "://" in file_id:
            tail = file_id.rstrip("/").rsplit("/files/", 1)
            if len(tail) == 2 and tail[1]:
                return f"files/{tail[1]}"
        if file_id.startswith("files/"):
            return file_id
        return f"files/{file_id}"

    def _file_upload_request(self, request: FileUploadRequest) -> TransportRequest:
        url = build_url(f"{self.upload_base_url.rstrip('/')}/files", request.extensions)
        content_type, body = multipart_related_body(
            metadata={"file": {"display_name": request.filename}},
            media_type=request.media_type,
            data=request.bytes,
        )
        return self._emit(
            method="POST",
            url=url,
            headers=list(self._auth_headers({
                "X-Goog-Upload-Protocol": "multipart",
                "Content-Type": content_type,
            }).items()),
            body=body,
            read_timeout=300.0,
        )

    def _file_info_from_body(self, body: str) -> FileInfo:
        data = json.loads(body)
        if isinstance(data.get("file"), dict):  # upload wraps; get/list do not
            return self._file_info(data["file"])
        return self._file_info(data)

    def _file_info(self, data: dict[str, Any]) -> FileInfo:
        uri = data.get("uri")
        name = data.get("name")
        file_id = uri if isinstance(uri, str) and uri else name
        if not isinstance(file_id, str) or not file_id:
            raise ProviderError("gemini: file object carries no uri or name", provider=self.provider)
        state = str(data.get("state") or "")
        if state.endswith("PROCESSING"):
            readiness = "pending"
        elif state.endswith("FAILED"):
            readiness = "failed"
        else:  # ACTIVE, absent, or unknown
            readiness = "ready"
        if isinstance(data.get("downloadUri"), str) and data.get("downloadUri"):
            downloadable: bool | None = True
        elif data.get("source") == "UPLOADED":
            downloadable = False  # the server's stated rule: only GENERATED files download
        else:
            downloadable = None
        display_name = data.get("displayName")
        mime = data.get("mimeType")
        size_raw = data.get("sizeBytes")  # the wire carries int64 as a string
        try:
            size_bytes = int(size_raw) if isinstance(size_raw, (str, int)) and not isinstance(size_raw, bool) else None
        except ValueError:
            size_bytes = None
        return FileInfo(
            id=file_id,
            filename=display_name if isinstance(display_name, str) and display_name else None,
            media_type=mime if isinstance(mime, str) and mime else None,
            size_bytes=size_bytes,
            created_at=iso_utc(data.get("createTime")),
            expires_at=iso_utc(data.get("expirationTime")),
            readiness=readiness,
            downloadable=downloadable,
            provider_data=data,
        )

    def _file_get_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/{path_id(self._file_resource(file_id), resource_name=True)}",
            headers=self._auth_headers(), read_timeout=60.0,
        )

    def _file_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        params: dict[str, Any] = {"pageSize": limit}
        if cursor is not None:
            params["pageToken"] = cursor
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/files",
            params=params, headers=self._auth_headers(), read_timeout=60.0,
        )

    def _file_page_from_list_body(self, body: str) -> FilePage:
        data = json.loads(body)
        entries = data.get("files") if isinstance(data.get("files"), list) else []
        items = tuple(self._file_info(entry) for entry in entries if isinstance(entry, dict))
        cursor = data.get("nextPageToken")
        return FilePage(items=items, next_cursor=cursor if isinstance(cursor, str) and cursor else None)

    def _file_delete_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="DELETE", url=f"{self.base_url.rstrip('/')}/{path_id(self._file_resource(file_id), resource_name=True)}",
            headers=self._auth_headers(), read_timeout=60.0,
        )

    def _file_download_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/{path_id(self._file_resource(file_id), resource_name=True)}:download?alt=media",
            headers=self._auth_headers(), read_timeout=300.0,
        )

    # ─── Cache resource hooks (cachedContents; the stored tier of MAP-6) ──
    #
    # Wire shapes verified live 2026-09-01 (research/caching/receipts/
    # gemini__explicit-resource__*): POST /cachedContents {model, contents,
    # systemInstruction?, tools?, toolConfig?, ttl "Ns", displayName?} ->
    # the object with name, model, createTime, expireTime,
    # usageMetadata.totalTokenCount; GET/PATCH(ttl)/DELETE by name; list
    # -> {cachedContents: [...], nextPageToken}.  The object pins its
    # model and owns system/tools: a request may not repeat them.
    # CacheInfo.id is the resource name verbatim ("cachedContents/<id>").

    @staticmethod
    def _cache_resource(cache_id: str) -> str:
        return cache_id if cache_id.startswith("cachedContents/") else f"cachedContents/{cache_id}"

    def _cache_create_request(self, prefix: Request, ttl_seconds: int | None, label: str | None) -> TransportRequest:
        body: dict[str, Any] = {
            "model": self._model_path(prefix.model),
            "contents": [self._message(m, self._call_names(prefix.messages)) for m in prefix.messages],
        }
        if prefix.system:
            text = prefix.system if isinstance(prefix.system, str) else parts_to_text(prefix.system)
            body["systemInstruction"] = {"parts": [{"text": text}]}
        if prefix.tools:
            declarations = [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in prefix.tools if isinstance(t, FunctionTool)
            ]
            tools_wire: list[dict[str, Any]] = []
            if declarations:
                tools_wire.append({"functionDeclarations": declarations})
            tools_wire += [_builtin_to_gemini(t) for t in prefix.tools if isinstance(t, BuiltinTool)]
            body["tools"] = tools_wire
        if ttl_seconds is not None:
            body["ttl"] = f"{ttl_seconds}s"
        if label is not None:
            body["displayName"] = label
        return self._emit(
            method="POST", url=f"{self.base_url.rstrip('/')}/cachedContents",
            headers=self._auth_headers({"Content-Type": "application/json"}), payload=body, read_timeout=120.0,
        )

    def _cache_info_from_body(self, body: str) -> CacheInfo:
        return self._cache_info(json.loads(body))

    def _cache_info(self, data: dict[str, Any]) -> CacheInfo:
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderError("gemini: cache object carries no name", provider=self.provider)
        model = str(data.get("model") or "")
        model = model[len("models/"):] if model.startswith("models/") else model
        if not model:
            raise ProviderError("gemini: cache object carries no model", provider=self.provider)
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        tokens = usage.get("totalTokenCount")
        label = data.get("displayName")
        return CacheInfo(
            id=name,
            model=model,
            tokens=int(tokens) if isinstance(tokens, (int, str)) and str(tokens).isdigit() else None,
            created_at=iso_utc(data.get("createTime")),
            expires_at=iso_utc(data.get("expireTime")),
            label=label if isinstance(label, str) and label else None,
            provider_data=data,
        )

    def _cache_get_request(self, cache_id: str) -> TransportRequest:
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/{path_id(self._cache_resource(cache_id), resource_name=True)}",
            headers=self._auth_headers(), read_timeout=60.0,
        )

    def _cache_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        params: dict[str, Any] = {"pageSize": int(limit)}
        if cursor is not None:
            params["pageToken"] = cursor
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/cachedContents",
            params=params, headers=self._auth_headers(), read_timeout=60.0,
        )

    def _cache_page_from_list_body(self, body: str) -> CachePage:
        data = json.loads(body)
        entries = data.get("cachedContents") if isinstance(data.get("cachedContents"), list) else []
        items = tuple(self._cache_info(e) for e in entries if isinstance(e, dict))
        cursor = data.get("nextPageToken")
        return CachePage(items=items, next_cursor=cursor if isinstance(cursor, str) and cursor else None)

    def _cache_delete_request(self, cache_id: str) -> TransportRequest:
        return self._emit(
            method="DELETE", url=f"{self.base_url.rstrip('/')}/{path_id(self._cache_resource(cache_id), resource_name=True)}",
            headers=self._auth_headers(), read_timeout=60.0,
        )

    def _cache_update_request(self, cache_id: str, ttl_seconds: int) -> TransportRequest:
        return self._emit(
            method="PATCH", url=f"{self.base_url.rstrip('/')}/{path_id(self._cache_resource(cache_id), resource_name=True)}",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            payload={"ttl": f"{ttl_seconds}s"}, read_timeout=60.0,
        )

    # ─── Batch hooks (Batch Mode, inline requests) ───────────────────
    #
    # Wire shapes verified live 2026-08-31: submit → POST
    # {model}:batchGenerateContent, job id = operation "name"
    # ("batches/<id>"), inline results under
    # response.inlinedResponses.inlinedResponses[] with metadata.key
    # correlation, list → GET /batches?pageSize=N → {"operations": [...]}.

    def _batch_submit_request(self, request: BatchRequest, upload_body: dict[str, Any] | None) -> TransportRequest:
        model = request.model or request.requests[0].model
        batch: dict[str, Any] = {
            "inputConfig": {
                "requests": {
                    "requests": [
                        {"request": self._payload(nested), "metadata": {"key": str(i)}}
                        for i, nested in enumerate(request.requests)
                    ]
                }
            },
        }
        if request.label is not None:
            batch["displayName"] = request.label
        payload = {"batch": batch, **(request.extensions or {})}
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/{self._model_path(model)}:batchGenerateContent",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            payload=payload,
            read_timeout=120.0,
        )

    def _batch_job_from_body(self, body: str) -> BatchJobInfo:
        return self._batch_job_info(json.loads(body))

    def _batch_job_info(self, data: dict[str, Any]) -> BatchJobInfo:
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderError("gemini: batch operation carries no name", provider=self.provider)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        label = metadata.get("displayName")
        return BatchJobInfo(
            id=name,
            status=_gemini_batch_status(data),
            label=label if isinstance(label, str) and label else None,
            created_at=iso_utc(metadata.get("createTime")),
            provider_data=data,
        )

    def _batch_status_request(self, batch_id: str) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/{path_id(batch_id, resource_name=True)}",
            headers=self._auth_headers(),
            read_timeout=60.0,
        )

    def _batch_cancel_request(self, batch_id: str) -> TransportRequest:
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/{path_id(batch_id, resource_name=True)}:cancel",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            payload={},
            read_timeout=60.0,
        )

    def _batch_result_fetches(self, status_body: dict[str, Any]) -> tuple[TransportRequest, ...]:
        # Inline submissions carry their results in the operation body.
        return ()

    def _batch_entries(self, status_body: dict[str, Any], fetched: tuple[str, ...]) -> tuple[BatchEntry, ...]:
        response_obj = status_body.get("response") if isinstance(status_body.get("response"), dict) else {}
        inlined = response_obj.get("inlinedResponses")
        if isinstance(inlined, dict):
            inlined = inlined.get("inlinedResponses")
        entries: list[BatchEntry] = []
        for position, item in enumerate(inlined or []):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            try:
                index = int(str(metadata.get("key")))
            except (TypeError, ValueError):
                index = position
            if isinstance(item.get("response"), dict):
                body_obj = item["response"]
                response = self.parse_response(
                    batch_entry_request(body_obj.get("modelVersion")), batch_entry_http(body_obj)
                )
                entries.append(BatchEntry(index=index, outcome="succeeded", response=response))
            else:
                # Per-entry failures arrive as a google.rpc.Status, not the
                # HTTP error envelope; map directly.
                err = item.get("error") if isinstance(item.get("error"), dict) else {}
                provider_code = err.get("status") or err.get("code")
                entries.append(BatchEntry(
                    index=index,
                    outcome="errored",
                    error=ErrorDetail(
                        code="provider",
                        message=str(err.get("message") or "batch entry errored"),
                        provider_code=str(provider_code) if provider_code is not None else None,
                    ),
                ))
        return tuple(sorted(entries, key=lambda e: e.index))

    def _batch_list_request(self, limit: int) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/batches",
            params={"pageSize": int(limit)},
            headers=self._auth_headers(),
            read_timeout=60.0,
        )

    def _batch_jobs_from_list_body(self, body: str) -> tuple[BatchJobInfo, ...]:
        data = json.loads(body)
        items = data.get("operations") if isinstance(data, dict) else None
        return tuple(self._batch_job_info(item) for item in (items or []) if isinstance(item, dict))

    # ─── Video generation (Veo; captured live 2026-09-01) ───────────────
    #
    # predictLongRunning returns an operation name (the ticket); polling
    # the operation yields done + response.generateVideoResponse with a
    # file URI that is KEY-BOUND (403 without the header, verified), so
    # the result step fetches the bytes — a URL the user cannot open is
    # not an honest VideoPart.  Listing is per model.

    def _video_submit_request(self, request: VideoGenerationRequest) -> TransportRequest:
        if request.images:
            # Veo's image input (instances[].image) is documented but not yet
            # live-receipted; the generations-ignores-input trap on xAI made
            # unverified media-input mappings a named hazard.  Raise for now.
            raise UnsupportedFeatureError(
                "gemini: video input images are not mapped yet; "
                "use extensions until the mapping is live-receipted",
                provider=self.provider,
            )
        instance: dict[str, Any] = {"prompt": request.prompt}
        parameters: dict[str, Any] = {}
        if request.seconds is not None:
            parameters["durationSeconds"] = request.seconds
        payload: dict[str, Any] = {"instances": [instance], **(request.extensions or {})}
        if parameters:
            payload.setdefault("parameters", parameters)
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/{self._model_path(request.model)}:predictLongRunning",
            headers=self._auth_headers({"Content-Type": "application/json"}),
            payload=payload,
            read_timeout=120.0,
        )

    def _video_job_from_body(self, body: str, video_id: "str | None" = None) -> VideoJobInfo:
        return self._video_job_info(json.loads(body))

    def _video_job_info(self, data: dict[str, Any]) -> VideoJobInfo:
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderError("gemini: video operation carries no name", provider=self.provider)
        if data.get("done") is True:
            status = "failed" if isinstance(data.get("error"), dict) else "completed"
        else:
            # Operations expose no queued/running distinction before done.
            status = "running"
        return VideoJobInfo(id=name, status=status, provider_data=data)

    def _video_status_request(self, video_id: str) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/{path_id(video_id, resource_name=True)}",
            headers=self._auth_headers({}),
            read_timeout=60.0,
        )

    def _video_result_uri(self, status_body: dict[str, Any]) -> str:
        response = status_body.get("response") if isinstance(status_body.get("response"), dict) else {}
        gvr = response.get("generateVideoResponse") if isinstance(response.get("generateVideoResponse"), dict) else {}
        samples = gvr.get("generatedSamples")
        if isinstance(samples, list) and samples:
            video = samples[0].get("video") if isinstance(samples[0], dict) else None
            uri = video.get("uri") if isinstance(video, dict) else None
            if isinstance(uri, str) and uri:
                return uri
        raise ProviderError("gemini: terminal video operation carries no video uri", provider=self.provider)

    def _video_result_fetch(self, status_body: dict[str, Any]) -> TransportRequest:
        return self._emit(
            method="GET",
            url=self._video_result_uri(status_body),
            headers=self._auth_headers({}),
            read_timeout=600.0,
        )

    def _video_part(self, status_body: dict[str, Any], fetched: "HttpResponse | None") -> VideoPart:
        if fetched is None:
            raise ProviderError("gemini: video content fetch is required", provider=self.provider)
        content_type = (fetched.header("content-type") or "").split(";", 1)[0].strip()
        if not content_type:
            raise ProviderError("gemini: video download carries no content-type", provider=self.provider)
        return VideoPart(media_type=content_type, data=base64.b64encode(fetched.body).decode("ascii"))

    def _video_list_request(self, limit: int, model: str | None) -> TransportRequest:
        if not model:
            raise UnsupportedFeatureError(
                "gemini: video jobs list per model — pass model= (operations live under models/<model>/operations)",
                provider=self.provider,
            )
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/{self._model_path(model)}/operations",
            params={"pageSize": int(limit)},
            headers=self._auth_headers({}),
            read_timeout=60.0,
        )

    def _video_jobs_from_list_body(self, body: str) -> tuple[VideoJobInfo, ...]:
        data = json.loads(body)
        ops = data.get("operations") if isinstance(data, dict) else None
        return tuple(self._video_job_info(op) for op in (ops or []) if isinstance(op, dict))

    # ─── Media generation (captured live 2026-09-01) ────────────────────
    #
    # Gemini has no dedicated generation endpoints: image models and TTS
    # models answer the ordinary generateContent call.  The hooks compose
    # the frozen chat mapping (build_request / parse_response), so the
    # shared sync drivers and the async mirrors both work unchanged.
    # Input images (edits) are ordinary parts in the same call — verified
    # honored by pixel check.  Image responses routinely carry narration
    # text next to the image; it lands in ImageGenerationResponse.text.

    def _image_generation_lm_request(self, request: ImageGenerationRequest) -> Request:
        extensions = dict(request.extensions or {})
        if request.size is not None:
            generation_config = dict(extensions.get("generationConfig") or {})
            image_config = dict(generation_config.get("imageConfig") or {})
            image_config.setdefault("aspectRatio", request.size)
            generation_config["imageConfig"] = image_config
            extensions["generationConfig"] = generation_config
        parts = (TextPart(text=request.prompt), *request.images)
        return Request(
            model=request.model,
            messages=(Message(role="user", parts=parts),),
            config=Config(extensions=extensions) if extensions else Config(),
        )

    def _image_generate_request(self, request: ImageGenerationRequest) -> TransportRequest:
        return self.build_request(self._image_generation_lm_request(request), stream=False)

    def _image_generation_from_response(self, request: ImageGenerationRequest, resp: HttpResponse) -> ImageGenerationResponse:
        chat = self.parse_response(self._image_generation_lm_request(request), resp)
        images = tuple(part for part in chat.message.parts if isinstance(part, ImagePart))
        if not images:
            raise ProviderError("gemini: model returned no image parts", provider=self.provider)
        texts = [part.text for part in chat.message.parts if isinstance(part, TextPart) and part.text]
        return ImageGenerationResponse(
            images=images,
            text="".join(texts) or None,
            id=chat.id,
            model=chat.model,
            usage=chat.usage,
            provider_data=chat.provider_data,
        )

    def _speech_generation_lm_request(self, request: SpeechGenerationRequest) -> Request:
        if request.format is not None:
            # No wire slot: Gemini TTS always answers PCM (captured
            # audio/L16;codec=pcm;rate=24000).  Raising beats dropping.
            raise UnsupportedFeatureError(
                "gemini: speech format cannot be chosen; the wire always returns PCM",
                provider=self.provider,
            )
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if request.voice is not None:
            generation_config["speechConfig"] = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}}
        extensions = {"generationConfig": generation_config, **(request.extensions or {})}
        return Request(model=request.model, messages=(Message.user(request.prompt),), config=Config(extensions=extensions))

    def _speech_generate_request(self, request: SpeechGenerationRequest) -> TransportRequest:
        return self.build_request(self._speech_generation_lm_request(request), stream=False)

    def _speech_generation_from_response(self, request: SpeechGenerationRequest, resp: HttpResponse) -> SpeechGenerationResponse:
        chat = self.parse_response(self._speech_generation_lm_request(request), resp)
        audio = chat.message.first(AudioPart)
        if audio is None:
            raise ProviderError("gemini: model returned no audio part", provider=self.provider)
        return SpeechGenerationResponse(audio=audio, id=chat.id, model=chat.model, usage=chat.usage, provider_data=chat.provider_data)
