from __future__ import annotations

from datetime import datetime

import json
import os
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
from ..access import ANTHROPIC_API
from ..compat import ANTHROPIC_PRESET_BASE_URLS, AnthropicCompat, ResolvedAnthropicCompat, resolve_anthropic_compat
from ..features import ProviderManifest
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    BatchEntry,
    BatchJobInfo,
    BatchRequest,
    BuiltinTool,
    CitationDelta,
    ContinuationDelta,
    ContinuationState,
    CitationPart,
    DocumentPart,
    ErrorDetail,
    FileInfo,
    FilePage,
    FileUploadRequest,
    FunctionTool,
    ImagePart,
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
    ToolCallDelta,
    continuation_data,
    ToolCallPart,
    ToolResultPart,
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
)
from .common import EFFORT_THINKING_BUDGETS, MEDIA_KINDS, anthropic_source, check_tool_result_media, iso_utc, model_infos_from_entries, multipart_form_body, parts_to_text, unnamed_tool_call_error

# Canonical builtin tool name → Anthropic tool format
_ANTHROPIC_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_20250305",
    "code_execution": "code_execution_20250522",
}

ANTHROPIC_PROVIDER_EXECUTED_BLOCKS = {
    "server_tool_use",
    "web_search_tool_result",
    "code_execution_tool_result",
}

_DEFAULT_ANTHROPIC_VISIBLE_TOKENS = 1024
_DEFAULT_ANTHROPIC_THINKING_BUDGET = 1024


def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_anthropic(tool: BuiltinTool) -> dict[str, Any]:
    out: dict[str, Any] = {"type": _ANTHROPIC_BUILTIN_MAP.get(tool.name, tool.name), "name": tool.name}
    if tool.config:
        out.update(tool.config)
    return out


def _response_format_to_anthropic_output_config(format_config: dict[str, Any]) -> dict[str, Any]:
    """Canonical response_format (INV-050) -> Anthropic output_config.

    The Messages API has no any-JSON mode: `format.schema` must be an object
    schema with `additionalProperties: false` (HTTP 400 otherwise, live
    2026-09-02), so `json_object` RAISES.  `strict` is satisfied (always
    constrained); `name` is a label with no slot (dropped, stated in MAP-8).
    """
    if format_config["type"] == "json_object":
        raise UnsupportedFeatureError(
            "anthropic: response_format json_object is not supported — the Messages API has no "
            "any-JSON mode; give a json_schema (objects need additionalProperties: false)",
            provider="anthropic",
        )
    return {"format": {"type": "json_schema", "schema": format_config["schema"]}}


def _reasoning_tokens(usage_payload: dict) -> int | None:
    # The wire nests thinking spend under output_tokens_details; absent
    # when thinking never ran, so None stays the honest "not reported".
    details = usage_payload.get("output_tokens_details")
    if isinstance(details, dict) and details.get("thinking_tokens") is not None:
        return int(details["thinking_tokens"])
    return None


_ADAPTIVE_CLASS_MARKERS = ("sonnet-5", "opus-5", "sonnet-4-6", "opus-4-6", "opus-4-7", "opus-4-8", "fable", "mythos", "haiku-5")


def anthropic_adaptive_class(model: str) -> bool:
    """True for models that take `thinking: {type: adaptive}` + `output_config.effort`.

    MAP-7 rule 10: a model-name table, receipted 2026-09-02 (Sonnet 5 accepts
    adaptive and rejects `enabled`; Sonnet 4.5 and Haiku 4.5 do the reverse).
    A table that rots: a new model outside it is treated as the manual class
    and the server answers with a clear 400 ("thinking.type.enabled is not
    supported for this model"); `extensions={"thinking": ...}` overrides.
    """
    lowered = model.lower()
    return any(marker in lowered for marker in _ADAPTIVE_CLASS_MARKERS)


def _reasoning_thinking_budget(request: Request) -> int | None:
    """Manual class only: the budget on the wire (MAP-7 rules 3 and 5)."""
    reasoning = request.config.reasoning
    if reasoning is None or reasoning.is_off:
        return None
    if reasoning.thinking_budget is not None:
        return reasoning.thinking_budget
    return EFFORT_THINKING_BUDGETS[reasoning.effort]


def _max_tokens_for_anthropic(request: Request, thinking_budget: int | None) -> int:
    """Manual class: max_tokens includes thinking, so the wire ceiling is the
    budget plus the visible cap.  Adaptive class (thinking_budget None):
    Config.max_tokens is the total ceiling — provider semantics, stated in
    spec/types.md."""
    if thinking_budget is None:
        return request.config.max_tokens or _DEFAULT_ANTHROPIC_VISIBLE_TOKENS
    visible_budget = request.config.max_tokens or _DEFAULT_ANTHROPIC_VISIBLE_TOKENS
    return thinking_budget + visible_budget


def _finish_reason(stop_reason: str | None, *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    reason = str(stop_reason or "").lower()
    if reason in {"max_tokens", "model_context_window_exceeded"}:
        return "length"
    if reason in {"tool_use", "pause_turn"}:
        return "tool_call"
    if reason in {"refusal", "safety", "content_filter"}:
        return "content_filter"
    return "stop"


def _anthropic_batch_status(data: dict[str, Any]) -> str:
    """Map an Anthropic Message Batch object to the canonical BatchStatus.

    ``ended`` is Anthropic's single terminal processing_status; the
    canonical terminal splits on request_counts — all-cancelled →
    ``cancelled``, all-expired → ``expired``, anything else →
    ``completed`` (per-entry outcomes live in the results, not the job).
    """
    status = str(data.get("processing_status") or "").lower()
    if status == "in_progress":
        return "running"
    if status == "canceling":
        return "cancelling"
    if status == "ended":
        counts = data.get("request_counts") or {}

        def n(key: str) -> int:
            try:
                return int(counts.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        if n("canceled") and not (n("succeeded") or n("errored") or n("expired")):
            return "cancelled"
        if n("expired") and not (n("succeeded") or n("errored") or n("canceled")):
            return "expired"
        return "completed"
    return "queued"


def _citation_from_anthropic(citation: dict[str, Any]) -> CitationPart | None:
    url = citation.get("url") or citation.get("uri")
    title = citation.get("title") or citation.get("document_title") or citation.get("source_title")
    text = citation.get("cited_text") or citation.get("text") or citation.get("quote")
    url_s = str(url) if url else None
    title_s = str(title) if title else None
    text_s = str(text) if text else None
    if url_s is None and title_s is None and text_s is None:
        return None
    return CitationPart(url=url_s, title=title_s, text=text_s)


_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


@dataclass(slots=True)
class AnthropicLM(BaseProviderLM):
    """Anthropic Messages dialect, bound to an access policy.

    ``access`` defaults to the API-key policy (``lm15.access.ANTHROPIC_API``);
    ``lm15.access.CLAUDE_CODE`` binds the same dialect to a local Claude
    Code login (``ClaudeCodeLM`` is that binding under a name). The policy
    is consulted at exactly these points: the auth header, static headers
    (``anthropic-beta`` is joined with the dialect's own betas), the system
    prefix, the login hint on errors, and the endpoint surfaces.
    """

    api_key: Credential | None = field(default=None, repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = _DEFAULT_BASE_URL
    api_version: str = "2023-06-01"
    compat: AnthropicCompat | str | None = None
    access: ProviderManifest | None = None
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="anthropic", init=False)
    account_id: str | None = field(default=None, init=False, repr=False)
    manifest: ClassVar[ProviderManifest] = ANTHROPIC_API
    _resolved_compat: ResolvedAnthropicCompat = field(init=False, repr=False, default=ResolvedAnthropicCompat())

    def __post_init__(self) -> None:
        self._bind_access(self.access, credentials_path=self.credentials_path, default_base_url=_DEFAULT_BASE_URL, settings=self.settings)
        compat = self.compat if self.compat is not None else self._registry_compat()
        if isinstance(compat, str):
            # A preset name also supplies that server's default base_url; an
            # explicit non-default base_url argument always wins (same rule
            # as OpenAIChatLM).
            resolved = resolve_anthropic_compat(AnthropicCompat.preset(compat))
            if self.base_url == _DEFAULT_BASE_URL:
                self.base_url = ANTHROPIC_PRESET_BASE_URLS.get(compat.lower(), _DEFAULT_BASE_URL)
        elif isinstance(compat, AnthropicCompat):
            resolved = resolve_anthropic_compat(compat)
        else:
            resolved = resolve_anthropic_compat(AnthropicCompat())
        self._resolved_compat = resolved

    _error_type_map: ClassVar[dict[str, type[ProviderError]]] = {
        "authentication_error": AuthError,
        "permission_error": AuthError,
        "billing_error": BillingError,
        "rate_limit_error": RateLimitError,
        "request_too_large": InvalidRequestError,
        "not_found_error": InvalidRequestError,
        "resource_not_found_error": InvalidRequestError,  # Moonshot's Anthropic wire (errors.md)
        "DeploymentNotFound": UnsupportedModelError,  # Azure Foundry: top-level code, model string = deployment
        "invalid_authentication_error": AuthError,  # Moonshot's Anthropic wire (errors.md)
        "invalid_request_error": InvalidRequestError,
        "api_error": ServerError,
        "overloaded_error": ServerError,
        "timeout_error": TimeoutError,
    }

    @staticmethod
    def _is_context_length_message(msg: str) -> bool:
        lowered = msg.lower()
        return (
            "prompt is too long" in lowered
            or "too many tokens" in lowered
            or "context window" in lowered
            or "context length" in lowered
            or ("token" in lowered and ("limit" in lowered or "exceed" in lowered))
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

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._error_type_map.get(provider_code, ProviderError)
        if self._is_context_length_message(message):
            cls = ContextLengthError
        elif provider_code == "not_found_error" and self._is_model_error(message):
            cls = UnsupportedModelError
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        try:
            data = json.loads(body)
            inner = data.get("error") if isinstance(data, dict) else None
            # Anthropic's envelope nests under `error`; Azure Foundry's
            # gateway uses top-level {code, message} before a deployment is
            # reached (live 2026-09-04).  Keep both shapes on one wire.
            err = inner if isinstance(inner, (dict, str)) else data if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            err_type = str(err.get("type") or err.get("code") or "") if isinstance(err, dict) else ""
            request_id = str(data.get("request_id") or "") if isinstance(data, dict) else ""

            if self._is_context_length_message(msg):
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=err_type or None,
                    request_id=request_id or None,
                )
            # `resource_not_found_error` is Moonshot's spelling of the same
            # 404 on its Anthropic wire ("Not found the model …", live
            # 2026-09-03); the message rule decides, as on the chat wire.
            if err_type == "DeploymentNotFound" or (
                err_type in ("not_found_error", "resource_not_found_error") and self._is_model_error(msg)
            ):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=err_type,
                    request_id=request_id or None,
                )

            cls = self._error_type_map.get(err_type)
            if cls:
                return self._provider_error(
                    cls,
                    msg,
                    status=status,
                    provider_code=err_type or None,
                    request_id=request_id or None,
                )
            if err_type and err_type not in msg:
                msg = f"{msg} ({err_type})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            err_type = ""
            request_id = ""
        return self._with_login_hint(map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.access.env_keys,
            provider_code=err_type or None,
            request_id=request_id or None,
        ))

    def _headers(self, request: Request | None = None) -> dict[str, str]:
        # The auth header is the policy's business: _emit applies it once per
        # request for every scheme (AUTH-2, AUTH-10).
        headers: dict[str, str] = {}
        headers["anthropic-version"] = self.api_version
        headers["content-type"] = "application/json"
        betas: list[str] = []
        for key, static in self.access.headers:
            if key.lower() == "anthropic-beta":
                betas.extend(b for b in static.split(",") if b)
            else:
                headers[key] = static
        if request is not None and any(
            isinstance(tool, BuiltinTool) and tool.name == "code_execution"
            for tool in request.tools
        ):
            betas.append("code-execution-2025-05-22")
        if betas:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    # ─── Request serialization ──────────────────────────────────────

    def _part(self, part) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            return {"type": "image", "source": anthropic_source(part)}
        if isinstance(part, DocumentPart):
            return {"type": "document", "source": anthropic_source(part)}
        if isinstance(part, ToolCallPart):
            return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.input}
        if isinstance(part, ToolResultPart):
            check_tool_result_media(self.provider, part, self._resolved_compat.tool_result_media, wire="a tool_result block")
            content_blocks = [self._tool_result_content(p) for p in part.content]
            out: dict[str, Any] = {"type": "tool_result", "tool_use_id": part.id}
            if content_blocks:
                # Anthropic accepts either a string or content blocks.  Blocks
                # preserve image/document tool outputs when present.
                if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
                    out["content"] = content_blocks[0]["text"]
                else:
                    out["content"] = content_blocks
            if part.is_error:
                out["is_error"] = True
            return out
        if isinstance(part, ThinkingPart):
            redacted = continuation_data(part, "anthropic", "redacted_thinking")
            if redacted is not None:
                return {"type": "redacted_thinking", **redacted}
            signature = continuation_data(part, "anthropic", "thinking_signature")
            if signature and signature.get("signature"):
                return {
                    "type": "thinking",
                    "thinking": part.text,
                    "signature": signature["signature"],
                }
            if self._resolved_compat.thinking_replay == "unsigned" and part.text:
                # The server signs nothing (`signature: ""`) and takes the
                # block back unsigned (Moonshot, live 2026-09-03); a text
                # replay would turn the reasoning into a spoken turn.
                return {"type": "thinking", "thinking": part.text}
            return {"type": "text", "text": part.text}
        return {"type": "text", "text": getattr(part, "text", "") or ""}

    def _tool_result_content(self, part) -> dict[str, Any]:
        if isinstance(part, TextPart):
            return {"type": "text", "text": part.text}
        if isinstance(part, ImagePart):
            return {"type": "image", "source": anthropic_source(part)}
        if isinstance(part, DocumentPart):
            return {"type": "document", "source": anthropic_source(part)}
        if part.type in MEDIA_KINDS:
            # ToolResultBlockParam takes text, image and document blocks only.
            raise UnsupportedFeatureError(
                f"{self.provider}: a {part.type} part cannot reach a tool_result block (text, image and document only; MAP-10)",
                provider=self.provider,
            )
        return {"type": "text", "text": parts_to_text((part,), provider=self.provider)}

    def _message(self, msg: Message) -> dict[str, Any]:
        role = "assistant" if msg.role == "assistant" else "user"
        parts = [self._part(part) for part in msg.parts]
        if msg.role == "developer":
            text = parts_to_text(msg.parts)
            parts = [{"type": "text", "text": f"[developer]\n{text}"}]
        return {"role": role, "content": parts}

    def _tool_choice_payload(self, request: Request) -> dict[str, Any] | None:
        tc = request.config.tool_choice
        if tc is None:
            return None

        payload: dict[str, Any] = {}
        if tc.mode == "none":
            payload["type"] = "none"
        elif tc.allowed:
            # {"type": "tool", "name": ...} forces client tools AND server
            # tools — verified live 2026-09-01 with web_search (the API
            # reference is silent on server tools; the capture is the
            # evidence).  Builtin names ride the same form because
            # _builtin_to_anthropic puts the canonical name on the wire.
            if len(tc.allowed) == 1 and tc.mode == "required":
                payload["type"] = "tool"
                payload["name"] = tc.allowed[0]
            elif set(tc.allowed) == {t.name for t in request.tools}:
                # Allowing every declared tool is no restriction at all.
                payload["type"] = "any" if tc.mode == "required" else "auto"
            else:
                # A proper-subset allowlist has no Anthropic wire form.
                # Degrading to any/auto would let the model call excluded
                # tools — raise, never silently widen the caller's policy.
                raise UnsupportedFeatureError(
                    "anthropic: tool_choice.allowed subsets are not supported — "
                    "the Messages API can force one named tool or allow all "
                    "declared tools, but cannot restrict to a subset. Send only "
                    "the allowed tools in Request.tools instead",
                    provider=self.provider,
                )
        elif tc.mode == "required":
            payload["type"] = "any"
        else:
            payload["type"] = "auto"

        if tc.parallel is False and payload["type"] != "none":
            payload["disable_parallel_tool_use"] = True

        return payload

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        compat = self._resolved_compat
        if compat.model_prefixes is not None and not request.model.startswith(compat.model_prefixes):
            # The server maps foreign model names onto its own models without
            # saying so (DeepSeek: claude-opus* → deepseek-v4-pro, live
            # 2026-09-03, the response `model` field is the only tell).
            raise UnsupportedModelError(
                f"{self.provider}: model {request.model!r} is not one this endpoint serves as typed "
                f"(expected a prefix in {list(compat.model_prefixes)}); it would be silently "
                "substituted by another model. Name the model you actually want.",
                provider=self.provider,
            )
        cache_cfg = request.config.cache
        # cache_control="none": the server ignores marks and caches
        # implicitly; nothing is placed and an explicit CacheConfig is not an
        # error — the same rule as the chat dialect's "none" presets.
        use_cache = cache_cfg is not None and cache_cfg.mode != "off" and compat.cache_control == "anthropic"
        long_cache = cache_cfg is not None and cache_cfg.retention == "long" and compat.cache_control == "anthropic"

        messages = [self._message(m) for m in request.messages]

        # MAP-6 marks.  prefix_until_index=N and prefix="history" (N = last
        # message) put cache_control on the last block of message N; the
        # explicit block form rather than the top-level automatic marker,
        # which walks backwards silently when the last block is ineligible.
        # prefix="stable" (and plain auto) mark the system block below.
        # key / resource name mechanisms the Messages API does not have.
        if use_cache and cache_cfg is not None:
            if cache_cfg.key is not None:
                raise UnsupportedFeatureError(
                    "anthropic: cache.key is not supported — the Messages API has no "
                    "cache affinity key (OpenAI's prompt_cache_key); marks on blocks "
                    "are the mechanism (prefix / prefix_until_index)",
                    provider=self.provider,
                )
            if cache_cfg.resource is not None:
                raise UnsupportedFeatureError(
                    "anthropic: cache.resource is not supported — the Messages API has no "
                    "stored-cache tier; it caches by marks on blocks",
                    provider=self.provider,
                )
            idx = None
            if cache_cfg.prefix_until_index is not None:
                idx = min(cache_cfg.prefix_until_index, len(messages) - 1)
            elif cache_cfg.prefix == "history":
                idx = len(messages) - 1
            if idx is not None and idx >= 0 and messages[idx].get("content"):
                last_block = messages[idx]["content"][-1]
                if isinstance(last_block, dict):
                    marker: dict[str, Any] = {"type": "ephemeral"}
                    if long_cache:
                        marker["ttl"] = "1h"
                    last_block.setdefault("cache_control", marker)

        reasoning = request.config.reasoning
        deepseek_thinking = compat.thinking_format == "deepseek"
        # "adaptive": every model on this server is the adaptive class (Meta
        # Model API, protocols--messages.md § Reasoning) — no model-name table.
        always_adaptive = compat.thinking_format == "adaptive"
        # "effort": no `thinking` object exists on this server; the dial is
        # output_config.effort alone (Moonshot messages--create.md).
        effort_only = compat.thinking_format == "effort"
        adaptive = (
            reasoning is not None and not reasoning.is_off
            and (deepseek_thinking or always_adaptive or effort_only or anthropic_adaptive_class(request.model))
        )
        if reasoning is not None and not reasoning.is_off:
            if compat.reasoning_efforts is not None and reasoning.effort not in compat.reasoning_efforts:
                # MAP-7 rule 2: a word with no native level raises here when
                # the server would not refuse it (Moonshot's Anthropic wire
                # answered 200 to `medium` and to `bogus`, live 2026-09-03).
                raise UnsupportedFeatureError(
                    f"{self.provider}: reasoning.effort={reasoning.effort!r} has no level on this server "
                    f"(it accepts {', '.join(compat.reasoning_efforts)}) and would be accepted silently",
                    provider=self.provider,
                )
            if reasoning.summary in ("concise", "detailed"):
                raise UnsupportedFeatureError(
                    f"anthropic: reasoning.summary={reasoning.summary!r} is an OpenAI detail level; "
                    "the Messages API returns thinking blocks whenever thinking runs (use 'auto' or None)",
                    provider=self.provider,
                )
            if adaptive:
                if reasoning.thinking_budget is not None:
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.thinking_budget is not supported on {request.model} — "
                        + ("this server ignores budget_tokens (a silent no-op); effort is the dial"
                           if deepseek_thinking else
                           "this server accepts budget_tokens without translating it (a silent no-op); "
                           "effort is the dial (protocols--messages.md)"
                           if always_adaptive else
                           "this model class takes thinking.type 'adaptive' with output_config.effort; "
                           "budget_tokens is rejected by the API (live 2026-09-02)"),
                        provider=self.provider,
                    )
                # MAP-7: the word goes verbatim on the always-adaptive server;
                # it answers an unsupported level with a 400 of its own.
                if reasoning.effort == "minimal" and not (deepseek_thinking or always_adaptive or effort_only):
                    raise UnsupportedFeatureError(
                        "anthropic: reasoning.effort='minimal' has no level on this model class "
                        "(output_config.effort is low|medium|high|xhigh|max); 'low' is the floor",
                        provider=self.provider,
                    )
        thinking_budget = None if adaptive else _reasoning_thinking_budget(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": _max_tokens_for_anthropic(request, thinking_budget),
        }

        if request.system:
            system_text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            if use_cache:
                cache_marker: dict[str, Any] = {"type": "ephemeral"}
                if long_cache:
                    cache_marker["ttl"] = "1h"
                payload["system"] = [{"type": "text", "text": system_text, "cache_control": cache_marker}]
            else:
                payload["system"] = system_text
        if compat.sampling_params == "reject":
            for name in ("temperature", "top_p", "top_k"):
                if getattr(request.config, name) is not None:
                    # The server documents none of these and swallows them
                    # silently (Moonshot, live 2026-09-03: temperature 0.5 is
                    # HTTP 200 here, "only 1 is allowed" on its chat wire).
                    raise UnsupportedFeatureError(
                        f"{self.provider}: config.{name} is silently ignored by this server "
                        "(the model's sampling is fixed); omit it",
                        provider=self.provider,
                    )
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.config.top_k is not None:
            payload["top_k"] = request.config.top_k
        if request.config.stop:
            payload["stop_sequences"] = list(request.config.stop)
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    tools_wire.append({"name": tool.name, "description": tool.description, "input_schema": tool.parameters})
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_anthropic(tool))
            payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            tc = request.config.tool_choice
            if compat.parallel_tool_calls == "reject" and tc is not None and tc.parallel is not None:
                # disable_parallel_tool_use is documented as ignored
                # (guide--anthropic-api.md); a silent no-op is refused (MAP-8 §2).
                raise UnsupportedFeatureError(
                    f"{self.provider}: tool_choice.parallel is silently ignored by this server "
                    "(disable_parallel_tool_use is not applied); omit it",
                    provider=self.provider,
                )
            payload["tool_choice"] = tool_choice
        if deepseek_thinking:
            # DeepSeek over the Anthropic wire (guide--anthropic-api.md; live
            # 2026-09-03): thinking is ON by default, so absence is not off —
            # an explicit off must reach the wire.  On: type=enabled and
            # output_config.effort (budget_tokens is ignored by the server).
            if reasoning is not None and reasoning.is_off:
                payload["thinking"] = {"type": "disabled"}
            elif reasoning is not None:
                payload["thinking"] = {"type": "enabled"}
                payload["output_config"] = {"effort": reasoning.effort}
        elif effort_only:
            # Moonshot over the Anthropic wire (messages--create.md): no
            # thinking object is documented; output_config.effort alone.
            # Off goes out as thinking.type=disabled, which the server
            # honours (live 2026-09-03: no thinking block, no thinking_tokens).
            if reasoning is not None and reasoning.is_off:
                payload["thinking"] = {"type": "disabled"}
            elif reasoning is not None:
                payload["output_config"] = {"effort": reasoning.effort}
        elif always_adaptive and reasoning is not None and reasoning.is_off:
            # The server reasons by default and cannot stop (Meta: Muse Spark
            # "always reasons").  Explicit off must reach the wire so the
            # server refuses it loudly (HTTP 400) instead of spending hidden
            # reasoning tokens the caller asked to disable.
            payload["thinking"] = {"type": "disabled"}
        elif adaptive:
            # MAP-7 rule 2 on the adaptive class (live 2026-09-02, Sonnet 5):
            # the model decides when to think; effort steers depth.
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": reasoning.effort}
        elif thinking_budget is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        if request.config.response_format:
            if compat.structured_output == "reject":
                # The server accepts output_config.format and ignores the
                # schema (DeepSeek, live 2026-09-03: 200 with keys the schema
                # never named).  Silent, so refuse before the wire.
                raise UnsupportedFeatureError(
                    f"{self.provider}: response_format is silently ignored by this server "
                    "(output_config.format is accepted and not applied); describe the shape in the prompt",
                    provider=self.provider,
                )
            output_config = _response_format_to_anthropic_output_config(request.config.response_format)
            payload["output_config"] = {**payload.get("output_config", {}), **output_config}
        # Promoted cross-provider knobs (changes/2026-09-01-extensions-burn-down):
        # user_id rides Anthropic's metadata.user_id; store has no Anthropic
        # wire field — raise, never silently drop.
        if request.config.service_tier is not None:
            payload["service_tier"] = request.config.service_tier
        if request.config.user_id is not None:
            payload["metadata"] = {"user_id": request.config.user_id}
        if request.config.store is not None:
            raise UnsupportedFeatureError(
                "anthropic: config.store is not supported — the Messages API has no "
                "response-storage opt-out field (OpenAI and Gemini carry it)",
                provider=self.provider,
            )
        if request.config.logprobs is not None:
            raise UnsupportedFeatureError(
                "anthropic: config.logprobs is not supported — the Messages API "
                "does not expose token log probabilities (OpenAI and Gemini "
                "carry them)",
                provider=self.provider,
            )
        if request.config.extensions:
            passthrough = {k: v for k, v in request.config.extensions.items() if k != "prompt_caching"}
            payload.update(passthrough)
        if self.access.system_prefix:
            # The access path requires this text first in the system prompt
            # (Claude Code's backend checks for it); the caller's system
            # follows, cache markers and all.
            prefix = {"type": "text", "text": self.access.system_prefix}
            existing = payload.get("system")
            if existing is None:
                payload["system"] = [prefix]
            elif isinstance(existing, list):
                payload["system"] = [prefix, *existing]
            else:
                payload["system"] = [prefix, {"type": "text", "text": str(existing)}]
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/messages",
            headers=self._headers(request),
            payload=self._payload(request, stream=stream),
            endpoint="messages",
            stream=stream,
            model=request.model,
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()
        parts: list[Any] = []
        unmapped: list[dict[str, str]] = []
        for block_index, block in enumerate(data.get("content", []) or []):
            if not isinstance(block, dict):
                _record_unmapped(unmapped, f"content[{block_index}]", type(block).__name__)
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(TextPart(text=str(block.get("text") or "")))
                for citation_payload in block.get("citations", []) or []:
                    if not isinstance(citation_payload, dict):
                        continue
                    citation = _citation_from_anthropic(citation_payload)
                    if citation is not None:
                        parts.append(citation)
            elif block_type == "tool_use":
                if not block.get("name"):
                    raise unnamed_tool_call_error(self.provider, f"content[{block_index}]")
                parts.append(ToolCallPart(
                    id=str(block.get("id") or f"tool_{len(parts)}"),
                    name=str(block["name"]),
                    input=block.get("input") if isinstance(block.get("input"), dict) else {},
                ))
            elif block_type == "thinking":
                continuation: tuple[ContinuationState, ...] = ()
                if block.get("signature"):
                    continuation = (
                        ContinuationState(
                            provider="anthropic",
                            kind="thinking_signature",
                            data={"signature": str(block.get("signature"))},
                        ),
                    )
                parts.append(
                    ThinkingPart(
                        text=str(block.get("thinking") or block.get("text") or ""),
                        continuation=continuation,
                    )
                )
            elif block_type == "redacted_thinking":
                # MAP-7 rule 11: hidden thinking is empty text plus replay
                # state; the blob goes back verbatim as redacted_thinking.
                continuation = ()
                redacted_payload = block.get("data")
                if redacted_payload is not None:
                    continuation = (
                        ContinuationState(
                            provider="anthropic",
                            kind="redacted_thinking",
                            data={"data": redacted_payload},
                        ),
                    )
                parts.append(ThinkingPart(text="", continuation=continuation))
            elif block_type in ANTHROPIC_PROVIDER_EXECUTED_BLOCKS:
                continue
            else:
                _record_unmapped(unmapped, f"content[{block_index}]", block_type)

        if not parts:
            parts = [TextPart(text="")]

        usage_payload = data.get("usage", {}) or {}
        # INV-029: absent counters stay None; Usage sums the total itself
        # only when both primaries are present.
        usage = Usage(
            input_tokens=usage_payload.get("input_tokens"),
            output_tokens=usage_payload.get("output_tokens"),
            cache_read_tokens=usage_payload.get("cache_read_input_tokens"),
            cache_write_tokens=usage_payload.get("cache_creation_input_tokens"),
            reasoning_tokens=_reasoning_tokens(usage_payload),
        )
        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        # D8 (2026-09-06): Response.id carries the message id; no
        # message-level continuation state is minted for it.
        return Response(
            id=str(data.get("id")) if data.get("id") else None,
            model=str(data.get("model") or request.model),
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=_finish_reason(data.get("stop_reason"), has_tool_call=has_tool),
            usage=usage,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        if not raw_event.data:
            return
        payload = json.loads(raw_event.data)
        et = payload.get("type")
        if et == "message_start":
            msg = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
            yield StreamStartEvent(
                id=str(msg.get("id")) if msg.get("id") else None,
                model=str(msg.get("model") or request.model),
            )
            return
        if et == "content_block_start":
            block = payload.get("content_block", {}) if isinstance(payload.get("content_block"), dict) else {}
            if block.get("type") == "tool_use":
                # A streamed tool_use block opens with input: {} and the
                # arguments arrive as input_json_delta fragments; the empty
                # object is a placeholder, not a fragment. Serialising it
                # produced "{}" + '{"city": ...}' — unparseable — the day
                # the first streaming tool call was pinned (2026-09-02). A
                # non-empty input on the start frame is kept verbatim.
                start_input = block.get("input")
                yield StreamDeltaEvent(
                    delta=ToolCallDelta(
                        input=json.dumps(start_input, separators=(",", ":")) if isinstance(start_input, dict) and start_input else ("" if isinstance(start_input, dict) else str(start_input or "")),
                        part_index=int(payload.get("index", 0) or 0),
                        id=str(block.get("id") or "") or None,
                        name=str(block.get("name") or "") or None,
                    )
                )
                return
            if block.get("type") == "redacted_thinking" and block.get("data") is not None:
                # MAP-7 rule 11: an empty thinking delta opens the hidden
                # block; its replay state is the block's only content.  The
                # block has no interior frames, so the state is emitted here
                # (the adapter is stateless per frame) and the trace reads the
                # same as an emission at content_block_stop.
                idx = int(payload.get("index", 0) or 0)
                yield StreamDeltaEvent(delta=ThinkingDelta(text="", part_index=idx))
                yield StreamDeltaEvent(
                    delta=ContinuationDelta(
                        provider="anthropic",
                        kind="redacted_thinking",
                        data={"data": block.get("data")},
                        part_index=idx,
                    )
                )
            return
        if et == "content_block_delta":
            delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
            idx = int(payload.get("index", 0) or 0)
            dtype = delta.get("type")
            if dtype == "text_delta":
                yield StreamDeltaEvent(delta=TextDelta(text=str(delta.get("text") or ""), part_index=idx))
            elif dtype == "input_json_delta":
                yield StreamDeltaEvent(delta=ToolCallDelta(input=str(delta.get("partial_json") or ""), part_index=idx))
            elif dtype == "thinking_delta":
                yield StreamDeltaEvent(delta=ThinkingDelta(text=str(delta.get("thinking") or ""), part_index=idx))
            elif dtype == "signature_delta" and delta.get("signature"):
                yield StreamDeltaEvent(
                    delta=ContinuationDelta(
                        provider="anthropic",
                        kind="thinking_signature",
                        data={"signature": str(delta.get("signature"))},
                        part_index=idx,
                    )
                )
            elif dtype in {"citation_delta", "citations_delta"}:
                citation = delta.get("citation", {}) if isinstance(delta.get("citation"), dict) else delta
                yield StreamDeltaEvent(delta=CitationDelta(
                    part_index=idx,
                    text=str(citation.get("cited_text") or citation.get("text") or "") or None,
                    url=str(citation.get("url") or "") or None,
                    title=str(citation.get("title") or "") or None,
                ))
            return
        if et == "message_delta":
            # Anthropic sends the authoritative stop_reason and final usage here;
            # message_stop is just the terminator and carries neither.
            delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
            usage_payload = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
            usage = None
            if usage_payload:
                usage = Usage(
                    input_tokens=usage_payload.get("input_tokens"),
                    output_tokens=usage_payload.get("output_tokens"),
                    cache_read_tokens=usage_payload.get("cache_read_input_tokens"),
                    cache_write_tokens=usage_payload.get("cache_creation_input_tokens"),
                    reasoning_tokens=_reasoning_tokens(usage_payload),
                )
            stop_reason = delta.get("stop_reason")
            if stop_reason is not None or usage is not None:
                # MAP-3 (D9, 2026-09-06): this frame supplied usage (and the
                # stop reason), so it is the end event's provider_data,
                # verbatim; the bare message_stop contributes nothing.
                yield StreamEndEvent(
                    finish_reason=_finish_reason(stop_reason) if stop_reason is not None else None,
                    usage=usage,
                    provider_data=payload,
                )
            return
        if et == "message_stop":
            yield StreamEndEvent()
            return
        if et == "error":
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("type") or err.get("code") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            yield StreamErrorEvent(error=self._error_detail(provider_code, message))

    # ─── Other endpoints ────────────────────────────────────────────

    # ─── Live model listing (provisional endpoint) ──────────────────────

    def _models_request(self):
        # limit=1000 is the endpoint maximum; the catalog fits in one page
        # today (has_more=false observed live 2026-08-31).
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/models",
            params={"limit": 1000},
            headers=self._headers(),
            read_timeout=30.0,
        )

    def _models_from_body(self, body: str):
        data = json.loads(body)
        entries = data.get("data") if isinstance(data, dict) else None
        return model_infos_from_entries(
            entries,
            provider=self.provider,
            api_family="anthropic_messages",
            id_of=lambda entry: entry.get("id"),
        )

    # ─── File hooks (Files API, GA) ──────────────────────────────
    #
    # Wire shapes verified live 2026-08-31 (curl-fixtures/files-2026-08-31/):
    # multipart/form-data upload with NO beta header (GA), file objects
    # carry mime_type / size_bytes / downloadable verbatim, list paginates
    # with an opaque `next_page` token passed back as `?page=`, download
    # is refused for non-tool-generated files (400, forwarded typed).

    def _file_upload_request(self, request: FileUploadRequest) -> TransportRequest:
        content_type, body = multipart_form_body(
            fields=[(k, str(v)) for k, v in (request.extensions or {}).items()],
            files=[("file", request.filename, request.media_type, request.bytes)],
        )
        headers = self._headers()
        headers["content-type"] = content_type
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/files",
            headers=list(headers.items()),
            body=body,
            read_timeout=300.0,
        )

    def _file_info_from_body(self, body: str) -> FileInfo:
        return self._file_info(json.loads(body))

    def _file_info(self, data: dict[str, Any]) -> FileInfo:
        file_id = data.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ProviderError("anthropic: file object carries no id", provider=self.provider)
        filename = data.get("filename")
        mime = data.get("mime_type")
        return FileInfo(
            id=file_id,
            filename=filename if isinstance(filename, str) and filename else None,
            media_type=mime if isinstance(mime, str) and mime else None,
            size_bytes=data.get("size_bytes") if isinstance(data.get("size_bytes"), int) else None,
            created_at=iso_utc(data.get("created_at")),
            expires_at=iso_utc(data.get("expires_at")),
            readiness="ready",  # Anthropic files have no processing state
            downloadable=data.get("downloadable") if isinstance(data.get("downloadable"), bool) else None,
            provider_data=data,
        )

    def _file_get_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/files/{file_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _file_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["page"] = cursor
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/files",
            params=params, headers=self._headers(), read_timeout=60.0,
        )

    def _file_page_from_list_body(self, body: str) -> FilePage:
        data = json.loads(body)
        entries = data.get("data") if isinstance(data.get("data"), list) else []
        items = tuple(self._file_info(entry) for entry in entries if isinstance(entry, dict))
        cursor = data.get("next_page")
        return FilePage(items=items, next_cursor=cursor if isinstance(cursor, str) and cursor else None)

    def _file_delete_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="DELETE", url=f"{self.base_url.rstrip('/')}/files/{file_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _file_download_request(self, file_id: str) -> TransportRequest:
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/files/{file_id}/content",
            headers=self._headers(), read_timeout=300.0,
        )

    # ─── Batch hooks (Message Batches API) ────────────────────────

    def _batch_submit_request(self, request: BatchRequest, upload_body: dict[str, Any] | None) -> TransportRequest:
        if request.label is not None:
            raise UnsupportedFeatureError(
                "anthropic: batch labels are not supported — the Message Batches "
                "create body has no metadata field (verified live 2026-08-31); "
                "submit without a label and correlate by id",
                provider=self.provider,
            )
        payload = {
            "requests": [
                {"custom_id": str(i), "params": self._payload(nested, stream=False)}
                for i, nested in enumerate(request.requests)
            ],
            **(request.extensions or {}),
        }
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/messages/batches",
            headers=self._headers(),
            payload=payload,
            read_timeout=120.0,
        )

    def _batch_job_from_body(self, body: str) -> BatchJobInfo:
        return self._batch_job_info(json.loads(body))

    def _batch_job_info(self, data: dict[str, Any]) -> BatchJobInfo:
        batch_id = data.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ProviderError("anthropic: batch object carries no id", provider=self.provider)
        return BatchJobInfo(
            id=batch_id,
            status=_anthropic_batch_status(data),
            created_at=iso_utc(data.get("created_at")),
            provider_data=data,
        )

    def _batch_status_request(self, batch_id: str) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/messages/batches/{batch_id}",
            headers=self._headers(),
            read_timeout=60.0,
        )

    def _batch_cancel_request(self, batch_id: str) -> TransportRequest:
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/messages/batches/{batch_id}/cancel",
            headers=self._headers(),
            read_timeout=60.0,
        )

    def _batch_result_fetches(self, status_body: dict[str, Any]) -> tuple[TransportRequest, ...]:
        url = status_body.get("results_url")
        if not isinstance(url, str) or not url:
            raise ProviderError("anthropic: ended batch carries no results_url", provider=self.provider)
        return (self._emit(method="GET", url=url, headers=self._headers(), read_timeout=300.0),)

    def _batch_entries(self, status_body: dict[str, Any], fetched: tuple[str, ...]) -> tuple[BatchEntry, ...]:
        entries: list[BatchEntry] = []
        for line in fetched[0].splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            index = int(str(item.get("custom_id")))
            result = item.get("result") or {}
            rtype = str(result.get("type") or "")
            if rtype == "succeeded":
                message = result.get("message") or {}
                response = self.parse_response(
                    batch_entry_request(message.get("model")), batch_entry_http(message)
                )
                entries.append(BatchEntry(index=index, outcome="succeeded", response=response))
            elif rtype == "errored":
                raw = result.get("error") or {}
                envelope = raw if isinstance(raw, dict) and "error" in raw else {"error": raw}
                err = self.normalize_error(400, json.dumps(envelope))
                entries.append(BatchEntry(
                    index=index,
                    outcome="errored",
                    error=ErrorDetail(
                        code=canonical_error_code(err),
                        message=err.message or "batch entry errored",
                        provider_code=err.provider_code,
                    ),
                ))
            elif rtype == "canceled":
                entries.append(BatchEntry(index=index, outcome="cancelled"))
            elif rtype == "expired":
                entries.append(BatchEntry(index=index, outcome="expired"))
            else:
                entries.append(BatchEntry(
                    index=index,
                    outcome="errored",
                    error=ErrorDetail(code="provider", message=f"unrecognized batch result type {rtype!r}"),
                ))
        return tuple(sorted(entries, key=lambda e: e.index))

    def _batch_list_request(self, limit: int) -> TransportRequest:
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/messages/batches",
            params={"limit": int(limit)},
            headers=self._headers(),
            read_timeout=60.0,
        )

    def _batch_jobs_from_list_body(self, body: str) -> tuple[BatchJobInfo, ...]:
        data = json.loads(body)
        items = data.get("data") if isinstance(data, dict) else None
        return tuple(self._batch_job_info(item) for item in (items or []) if isinstance(item, dict))
