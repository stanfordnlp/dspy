"""
lm15.providers.openai_chat — OpenAI Chat Completions adapter.

The Chat Completions dialect is the wire format spoken by OpenAI's legacy
endpoint and by most OpenAI-compatible servers: ollama, Groq, OpenRouter,
vLLM, SGLang, DeepSeek, and friends.  Provider quirks are described by
``OpenAIChatCompat`` policies (see lm15.compat); named presets bundle a
policy with that server's default base URL.
"""

from __future__ import annotations

from datetime import datetime

import json
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterator, Mapping

from ..compat import (
    OPENAI_CHAT_PRESET_BASE_URLS,
    OpenAIChatCompat,
    ResolvedOpenAIChatCompat,
    resolve_openai_chat_compat,
)
from ..errors import ProviderError, ServerError, UnsupportedFeatureError
from ..access import OPENAI_CHAT_API
from ..features import ProviderManifest
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    BuiltinTool,
    CacheConfig,
    CitationPart,
    Config,
    FunctionTool,
    ImagePart,
    Message,
    Part,
    Reasoning,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    audio,
    document,
    image,
)
from .base import BaseProviderLM, Credential, HttpResponse, SyncTransport, default_transport
from .common import (
    MEDIA_KINDS,
    check_tool_result_media,
    media_data_uri,
    tool_result_error_text,
    model_infos_from_entries,
    openai_token_logprobs,
    parse_json_object,
    unnamed_tool_call_error,
    parts_to_text,
)
from .openai import (
    OpenAILM,
    _attach_unmapped,
    _breakpoint_unsupported,
    _cache_breakpoint_index,
    _cache_common_payload,
    _cache_stable_prefix,
    _record_unmapped,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Canonical builtin tool name → Groq server-executed tool type (compat
# builtin_tools="groq"; both verified live 2026-09-01: Groq runs them
# server-side and reports the trace in message.executed_tools, which
# stays in provider_data per MAP-1).
_GROQ_BUILTIN_MAP: dict[str, str] = {
    "web_search": "browser_search",
    "code_execution": "code_interpreter",
}

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
}


def _chat_image_block(part: Any, provider: str) -> dict[str, Any]:
    """`{"type": "image_url", "image_url": {url, detail?}}`: a URL verbatim,
    inline data or a path as a data URI. `file_id` has no slot on this wire."""
    if part.file_id is not None:
        raise UnsupportedFeatureError(
            f"{provider}: an image addressed by file_id cannot be sent on the Chat Completions wire "
            "(no file reference form); pass a URL or inline data", provider=provider,
        )
    payload: dict[str, Any] = {"url": part.url if part.url is not None else media_data_uri(part)}
    if getattr(part, "detail", None):
        payload["detail"] = part.detail
    return {"type": "image_url", "image_url": payload}


def _chat_content_parts(msg: Message, *, force_array: bool = False, provider: str = "openai-chat") -> str | list[dict[str, Any]]:
    """Map non-assistant message parts to chat-completions content.

    Single text part → plain string; anything multimodal → content array.
    ``force_array`` keeps the array form for a lone text part (a cache
    breakpoint rides on a text content block, never on a bare string).
    A part the wire has no slot for RAISES (MAP-10); it is never rendered
    as text or dropped.
    """
    parts = [p for p in msg.parts if not isinstance(p, (ToolCallPart, ToolResultPart))]
    if len(parts) == 1 and isinstance(parts[0], TextPart) and not force_array:
        return parts[0].text
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append({"type": "text", "text": part.text})
        elif part.type == "image":
            out.append(_chat_image_block(part, provider))
        elif isinstance(part, ThinkingPart):
            continue  # thinking is never replayed as user content
        elif part.type in MEDIA_KINDS:
            raise UnsupportedFeatureError(
                f"{provider}: a {part.type} part in a {msg.role} message has no slot on the Chat Completions wire "
                "(text and image_url only); the OpenAI Responses, Anthropic and Gemini dialects carry it (MAP-10)",
                provider=provider,
            )
        else:
            out.append({"type": "text", "text": parts_to_text((part,), provider=provider)})
    return out


def _tool_row_content(provider: str, part: ToolResultPart, policy: str) -> str | list[dict[str, Any]]:
    """A `role: tool` row's content (MAP-10): a string when text-only; on a
    preset that proved the array form live, text and image_url blocks; a
    media part the preset does not admit raises first. `is_error` rides
    as an `[error] ` prefix (rule 5: the wire has no flag)."""
    check_tool_result_media(provider, part, policy, wire="a Chat Completions tool row")
    if all(p.type not in MEDIA_KINDS for p in part.content):
        return tool_result_error_text(part, parts_to_text(part.content, provider=provider, where="a Chat Completions tool row"))
    blocks: list[dict[str, Any]] = []
    for p in part.content:
        if p.type == "image":
            blocks.append(_chat_image_block(p, provider))
        else:
            blocks.append({"type": "text", "text": parts_to_text((p,), provider=provider)})
    if part.is_error:
        first = next((b for b in blocks if b["type"] == "text"), None)
        if first is None:
            blocks.insert(0, {"type": "text", "text": "[error]"})
        else:
            first["text"] = "[error] " + first["text"]
    return blocks


def _response_format_to_chat(format_config: dict[str, Any]) -> dict[str, Any]:
    """Canonical response_format (INV-050) -> chat-completions response_format."""
    if format_config["type"] == "json_object":
        return {"type": "json_object"}
    inner: dict[str, Any] = {"name": format_config.get("name") or "response", "schema": format_config["schema"]}
    if "strict" in format_config:
        inner["strict"] = format_config["strict"]
    return {"type": "json_schema", "json_schema": inner}


# ─── Ingest: a Chat Completions request body → canonical Request (MAP-12) ──
#
# The decoder for the encoders above and for OpenAIChatLM._build_messages /
# _payload below.  It reads ONE preset's spellings (the same
# ResolvedOpenAIChatCompat the builder writes with), so for every canonical
# Request r that the builder can carry losslessly,
# request_from_openai_chat(build(r)) == r; the lossy cells are enumerated in
# docs/mapping-rules.md MAP-12 and pinned per case by the contract's `ingest`
# direction.  Every wire key has exactly one verdict
# (lm15-contract/tools/openai-chat-ingest-verdicts.json): it maps to a
# canonical field, it passes verbatim through config.extensions, it is
# refused with UnsupportedFeatureError, or — for the four call-mode /
# default-valued keys only — it is ignored by rule.  A key with no verdict is
# refused: lm15 never drops silently (port.md rule 4).
#
# Malformed input (a wrong JSON type, a missing required key, an unparsable
# tool-call argument string) is a ValueError / TypeError, like serde.

# Top-level keys forwarded verbatim into config.extensions: generation knobs
# OpenAI documents that no canonical field expresses and that the builder
# re-emits verbatim (payload.update(extensions)), so they round-trip.
_INGEST_EXTENSIONS_KEYS: frozenset[str] = frozenset({
    "seed", "logit_bias", "presence_penalty", "frequency_penalty", "metadata",
    "verbosity", "moderation", "provider",
})

# Top-level keys refused with the reason a canonical Request cannot carry them.
_INGEST_REFUSED_KEYS: dict[str, str] = {
    "n": "lm15 reads one choice per response; n>1 would silently lose choices — fan out in the caller",
    "functions": "the deprecated function-calling shape; declare tools with {type: function, function: {...}}",
    "function_call": "the deprecated function-calling shape; use tool_choice",
    "audio": "audio output parameters have no canonical slot on the chat surface",
    "modalities": "output modality selection has no canonical slot on the chat surface",
    "prediction": "predicted-output content has no canonical slot",
    "web_search_options": "a server-executed search the chat dialect cannot map to parts (MAP-1); the Responses dialect carries web_search as a BuiltinTool",
    "top_k": "the Chat Completions wire has no top_k (the builder raises on Config.top_k for the same reason); servers that take it do so through extensions",
}

# Call-mode keys: they say HOW the request is sent, not WHAT is asked.  A
# canonical Request has no stream flag (stream=... is an argument of
# complete()/stream()), so these are read and dropped — the one place ingest
# drops anything, stated in MAP-12.
_INGEST_CALL_MODE_KEYS: frozenset[str] = frozenset({"stream", "stream_options"})

# Keys the config decoder consumes itself (every other key is looked up in
# the tables above, and an unlisted key is refused).
_INGEST_CONFIG_KEYS: frozenset[str] = frozenset({
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "max_completion_tokens", "max_tokens", "temperature", "top_p", "stop",
    "logprobs", "top_logprobs", "response_format", "service_tier", "store",
    "user", "safety_identifier", "user_id",
    "reasoning_effort", "reasoning", "thinking", "enable_thinking", "chat_template_kwargs", "reasoning_format",
    "prompt_cache_key", "prompt_cache_retention", "prompt_cache_options",
})

_INGEST_GROQ_BUILTIN_INVERSE: dict[str, str] = {wire: name for name, wire in _GROQ_BUILTIN_MAP.items()}

_INGEST_AUDIO_MEDIA_TYPES: dict[str, str] = {"wav": "audio/wav", "mp3": "audio/mpeg"}


def _ingest_unsupported(provider: str, what: str, why: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(f"{provider}: {what} cannot be carried by a canonical Request — {why}", provider=provider)


def _ingest_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{where} must be a string, got {type(value).__name__}")
    return value


def _ingest_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a JSON object, got {type(value).__name__}")
    return value


def _ingest_only_keys(provider: str, obj: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    """An unlisted key inside a block or object is a refusal, never a drop."""
    extra = sorted(set(obj) - allowed)
    if extra:
        raise _ingest_unsupported(provider, f"{where} key {extra[0]!r}", "no canonical slot for it")


def _ingest_data_uri(value: str, where: str) -> tuple[str, str]:
    """``data:<media_type>;base64,<payload>`` → (media_type, payload).  The
    inverse of media_data_uri; anything else is malformed here."""
    if not value.startswith("data:"):
        raise ValueError(f"{where} must be a base64 data URI")
    head, sep, payload = value[5:].partition(",")
    if not sep or not head.endswith(";base64") or not payload:
        raise ValueError(f"{where} must be a base64 data URI (data:<media-type>;base64,<payload>)")
    media_type = head[: -len(";base64")]
    if not media_type:
        raise ValueError(f"{where} data URI has no media type")
    return media_type, payload


def _ingest_image_block(provider: str, block: Mapping[str, Any], where: str) -> ImagePart:
    """Inverse of _chat_image_block.  A data URI becomes inline data with the
    URI's media type; any other URL stays a URL (media_type guessed from the
    path, else the ImagePart default — the wire carries no type for a URL)."""
    _ingest_only_keys(provider, block, frozenset({"type", "image_url", "prompt_cache_breakpoint"}), where)
    spec = _ingest_object(block.get("image_url"), f"{where}.image_url")
    _ingest_only_keys(provider, spec, frozenset({"url", "detail"}), f"{where}.image_url")
    url = _ingest_str(spec.get("url"), f"{where}.image_url.url")
    detail = spec.get("detail")
    if url.startswith("data:"):
        media_type, payload = _ingest_data_uri(url, f"{where}.image_url.url")
        return image(data=payload, media_type=media_type, detail=detail)
    guessed = mimetypes.guess_type(url)[0]
    return image(url=url, media_type=guessed if guessed and guessed.startswith("image/") else None, detail=detail)


def _ingest_text_block(provider: str, block: Mapping[str, Any], where: str) -> TextPart:
    _ingest_only_keys(provider, block, frozenset({"type", "text", "prompt_cache_breakpoint"}), where)
    return TextPart(text=_ingest_str(block.get("text"), f"{where}.text"))


def _ingest_has_breakpoint(block: Mapping[str, Any], where: str) -> bool:
    mark = block.get("prompt_cache_breakpoint")
    if mark is None:
        return False
    mark = _ingest_object(mark, f"{where}.prompt_cache_breakpoint")
    if mark != {"mode": "explicit"}:
        raise ValueError(f"{where}.prompt_cache_breakpoint must be {{\"mode\": \"explicit\"}}")
    if block.get("type") != "text":
        # The builder places a mark on a text block only (CacheConfig table).
        raise ValueError(f"{where}: a prompt_cache_breakpoint rides on a text block, not {block.get('type')!r}")
    return True


def _ingest_content_blocks(
    provider: str, content: Any, *, role: str, where: str,
) -> tuple[list[Part], bool]:
    """A row's ``content`` → parts, plus whether its LAST block carries the
    prompt-cache breakpoint.  A string is one TextPart.  Which block types a
    role admits follows the wire's own schema (chat--create.md)."""
    if isinstance(content, str):
        return [TextPart(text=content)], False
    if not isinstance(content, list):
        raise TypeError(f"{where}.content must be a string or an array of content parts")
    parts: list[Part] = []
    breakpoint_at_end = False
    for index, block in enumerate(content):
        block = _ingest_object(block, f"{where}.content[{index}]")
        block_where = f"{where}.content[{index}]"
        kind = block.get("type")
        marked = _ingest_has_breakpoint(block, block_where)
        if marked and index != len(content) - 1:
            raise ValueError(f"{block_where}: a prompt_cache_breakpoint marks the end of a message; it must be on the last block")
        breakpoint_at_end = breakpoint_at_end or marked
        if kind == "text":
            parts.append(_ingest_text_block(provider, block, block_where))
        elif kind == "image_url" and role in ("user", "tool"):
            parts.append(_ingest_image_block(provider, block, block_where))
        elif kind == "input_audio" and role == "user":
            _ingest_only_keys(provider, block, frozenset({"type", "input_audio", "prompt_cache_breakpoint"}), block_where)
            spec = _ingest_object(block.get("input_audio"), f"{block_where}.input_audio")
            _ingest_only_keys(provider, spec, frozenset({"data", "format"}), f"{block_where}.input_audio")
            fmt = _ingest_str(spec.get("format"), f"{block_where}.input_audio.format")
            media_type = _INGEST_AUDIO_MEDIA_TYPES.get(fmt)
            if media_type is None:
                raise ValueError(f"{block_where}.input_audio.format must be one of {sorted(_INGEST_AUDIO_MEDIA_TYPES)}")
            parts.append(audio(data=_ingest_str(spec.get("data"), f"{block_where}.input_audio.data"), media_type=media_type))
        elif kind == "file" and role == "user":
            _ingest_only_keys(provider, block, frozenset({"type", "file", "prompt_cache_breakpoint"}), block_where)
            spec = _ingest_object(block.get("file"), f"{block_where}.file")
            _ingest_only_keys(provider, spec, frozenset({"file_data", "file_id", "filename"}), f"{block_where}.file")
            if spec.get("filename") is not None:
                raise _ingest_unsupported(provider, f"{block_where}.file.filename", "DocumentPart has no filename slot")
            if spec.get("file_id") is not None and spec.get("file_data") is None:
                parts.append(document(file_id=_ingest_str(spec["file_id"], f"{block_where}.file.file_id")))
            elif spec.get("file_data") is not None and spec.get("file_id") is None:
                media_type, payload = _ingest_data_uri(_ingest_str(spec["file_data"], f"{block_where}.file.file_data"), f"{block_where}.file.file_data")
                parts.append(document(data=payload, media_type=media_type))
            else:
                raise ValueError(f"{block_where}.file needs exactly one of file_data / file_id")
        elif kind == "refusal" and role == "assistant":
            _ingest_only_keys(provider, block, frozenset({"type", "refusal"}), block_where)
            parts.append(RefusalPart(text=_ingest_str(block.get("refusal"), f"{block_where}.refusal")))
        else:
            raise _ingest_unsupported(
                provider, f"{block_where} of type {kind!r} in a {role} message",
                "no canonical part for that block on this wire (a part is not a knob: there is no extensions door for content)",
            )
    return parts, breakpoint_at_end


# Assistant-row keys that are a client library's object model, not the wire
# (litellm's ChatCompletionMessage dumped back into history). Their null or
# empty form carries nothing and reads as absent; a non-empty one is refused.
_INGEST_CLIENT_OBJECT_KEYS: frozenset[str] = frozenset({"provider_specific_fields", "thinking_blocks", "images"})


def _ingest_all_empty(value: Any) -> bool:
    """A dict whose every value is null/empty (litellm: {"refusal": null})."""
    return isinstance(value, dict) and all(v is None or v == [] or v == {} for v in value.values())


def _ingest_annotations(provider: str, raw: Any, content_text: str | None, where: str) -> list[CitationPart]:
    """OpenAI's assistant ``annotations`` (url_citation entries with a span
    into the content) → CitationParts; an empty list is nothing."""
    if not isinstance(raw, list):
        raise TypeError(f"{where}.annotations must be an array")
    out: list[CitationPart] = []
    for index, entry in enumerate(raw):
        entry_where = f"{where}.annotations[{index}]"
        entry = _ingest_object(entry, entry_where)
        if entry.get("type") != "url_citation":
            raise _ingest_unsupported(provider, f"{entry_where} of type {entry.get('type')!r}", "only url_citation annotations have a canonical part (CitationPart)")
        _ingest_only_keys(provider, entry, frozenset({"type", "url_citation"}), entry_where)
        spec = _ingest_object(entry.get("url_citation"), f"{entry_where}.url_citation")
        _ingest_only_keys(provider, spec, frozenset({"url", "title", "start_index", "end_index"}), f"{entry_where}.url_citation")
        text: str | None = None
        start, end = spec.get("start_index"), spec.get("end_index")
        if content_text is not None and isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(content_text):
            text = content_text[start:end] or None
        out.append(CitationPart(
            url=_ingest_str(spec.get("url"), f"{entry_where}.url_citation.url"),
            title=_ingest_str(spec["title"], f"{entry_where}.url_citation.title") if spec.get("title") is not None else None,
            text=text,
        ))
    return out


def _ingest_tool_calls(provider: str, calls: Any, where: str) -> list[ToolCallPart]:
    if not isinstance(calls, list):
        raise TypeError(f"{where}.tool_calls must be an array")
    out: list[ToolCallPart] = []
    for index, call in enumerate(calls):
        call = _ingest_object(call, f"{where}.tool_calls[{index}]")
        call_where = f"{where}.tool_calls[{index}]"
        kind = call.get("type", "function")
        if kind != "function":
            raise _ingest_unsupported(provider, f"{call_where} of type {kind!r}", "only function tool calls have a canonical part")
        _ingest_only_keys(provider, call, frozenset({"id", "type", "function"}), call_where)
        function = _ingest_object(call.get("function"), f"{call_where}.function")
        _ingest_only_keys(provider, function, frozenset({"name", "arguments"}), f"{call_where}.function")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            # The builder writes json.dumps(input); the inverse is exact, and
            # a caller-authored string that is not a JSON object is malformed
            # (the lenient parse_json_object is for PROVIDER output only).
            try:
                parsed = json.loads(arguments) if arguments else {}
            except ValueError as exc:
                raise ValueError(f"{call_where}.function.arguments is not JSON: {exc}") from None
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            raise ValueError(f"{call_where}.function.arguments must encode a JSON object")
        out.append(ToolCallPart(
            id=_ingest_str(call.get("id"), f"{call_where}.id"),
            name=_ingest_str(function.get("name"), f"{call_where}.function.name"),
            input=parsed,
        ))
    return out


def _ingest_messages(
    provider: str, rows: Any, compat: ResolvedOpenAIChatCompat,
) -> tuple[str | tuple[Part, ...] | None, list[Message], bool, int | None]:
    """Wire rows → (system, messages, system_breakpoint, breakpoint_message_index).

    Rules (MAP-12): the first row, when system/developer, is Request.system
    (the builder renders system there under compat.instruction_role, so the
    wire cannot tell a system prompt from a leading developer message);
    a later system/developer row is a developer Message; consecutive tool
    rows form one tool Message (the builder emits one row per result part);
    a `name` on any row is refused (no canonical slot)."""
    if not isinstance(rows, list):
        raise TypeError("messages must be an array")
    system: str | tuple[Part, ...] | None = None
    system_breakpoint = False
    breakpoint_index: int | None = None
    messages: list[Message] = []
    pending_results: list[ToolResultPart] = []

    def flush_results() -> None:
        if pending_results:
            messages.append(Message(role="tool", parts=tuple(pending_results)))
            pending_results.clear()

    for index, row in enumerate(rows):
        row = _ingest_object(row, f"messages[{index}]")
        where = f"messages[{index}]"
        role = row.get("role")
        if row.get("name") is not None and role != "tool":
            raise _ingest_unsupported(provider, f"{where}.name", "a per-message participant name has no canonical slot")

        if role in ("system", "developer"):
            flush_results()
            _ingest_only_keys(provider, row, frozenset({"role", "content"}), where)
            parts, marked = _ingest_content_blocks(provider, row.get("content"), role="system", where=where)
            if index == 0:
                if marked:
                    system_breakpoint = True
                if len(parts) == 1 and isinstance(parts[0], TextPart):
                    system = parts[0].text
                else:
                    system = tuple(parts)
            else:
                if marked:
                    breakpoint_index = len(messages)
                messages.append(Message(role="developer", parts=tuple(parts)))
            continue

        if role == "user":
            flush_results()
            _ingest_only_keys(provider, row, frozenset({"role", "content", "name"}), where)
            parts, marked = _ingest_content_blocks(provider, row.get("content"), role="user", where=where)
            if marked:
                if breakpoint_index is not None or system_breakpoint:
                    raise ValueError(f"{where}: a request carries at most one prompt_cache_breakpoint")
                breakpoint_index = len(messages)
            messages.append(Message(role="user", parts=tuple(parts)))
            continue

        if role == "assistant":
            flush_results()
            _ingest_only_keys(
                provider, row,
                frozenset({"role", "content", "tool_calls", "refusal", "reasoning_content", "name", "audio", "function_call",
                           "annotations", *_INGEST_CLIENT_OBJECT_KEYS}),
                where,
            )
            if row.get("audio") is not None:
                raise _ingest_unsupported(provider, f"{where}.audio", "an assistant audio reference has no canonical part")
            if row.get("function_call") is not None:
                raise _ingest_unsupported(provider, f"{where}.function_call", "the deprecated function-calling shape; use tool_calls")
            for key in _INGEST_CLIENT_OBJECT_KEYS:
                # A client library's object model (litellm) dumped into history:
                # null or empty carries nothing; anything else has no mapping.
                value = row.get(key)
                if value is not None and value != [] and value != {} and not _ingest_all_empty(value):
                    raise _ingest_unsupported(provider, f"{where}.{key}", "a client library's own field with no canonical part; only its empty form reads as absent")
            parts: list[Part] = []
            reasoning_text = row.get("reasoning_content")
            if reasoning_text is not None:
                parts.append(ThinkingPart(text=_ingest_str(reasoning_text, f"{where}.reasoning_content")))
            content = row.get("content")
            if content is not None:
                text_parts, marked = _ingest_content_blocks(provider, content, role="assistant", where=where)
                if marked:
                    raise ValueError(f"{where}: a prompt_cache_breakpoint cannot mark an assistant message (the builder refuses the same cell)")
                parts.extend(text_parts)
            refusal_text = row.get("refusal")
            if refusal_text is not None:
                parts.append(RefusalPart(text=_ingest_str(refusal_text, f"{where}.refusal")))
            if row.get("tool_calls") is not None:
                parts.extend(_ingest_tool_calls(provider, row["tool_calls"], where))
            if row.get("annotations") is not None:
                parts.extend(_ingest_annotations(provider, row["annotations"], content if isinstance(content, str) else None, where))
            if not parts:
                parts.append(TextPart(text=""))  # the never-empty rule (MAP-2), applied to history
            messages.append(Message(role="assistant", parts=tuple(parts)))
            continue

        if role == "tool":
            _ingest_only_keys(provider, row, frozenset({"role", "content", "tool_call_id", "name"}), where)
            parts, marked = _ingest_content_blocks(provider, row.get("content"), role="tool", where=where)
            if marked:
                raise ValueError(f"{where}: a prompt_cache_breakpoint cannot mark a tool message (the builder refuses the same cell)")
            name = row.get("name")
            pending_results.append(ToolResultPart(
                id=_ingest_str(row.get("tool_call_id"), f"{where}.tool_call_id"),
                content=tuple(parts),
                name=_ingest_str(name, f"{where}.name") if name is not None else None,
            ))
            continue

        if role == "function":
            raise _ingest_unsupported(provider, f"{where} with role 'function'", "the deprecated function-calling shape; use a tool row with tool_call_id")
        raise ValueError(f"{where}.role must be one of system, developer, user, assistant, tool; got {role!r}")

    flush_results()
    return system, messages, system_breakpoint, breakpoint_index


def _ingest_tools(provider: str, raw: Any, compat: ResolvedOpenAIChatCompat) -> list[FunctionTool | BuiltinTool]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError("tools must be an array")
    tools: list[FunctionTool | BuiltinTool] = []
    for index, entry in enumerate(raw):
        entry = _ingest_object(entry, f"tools[{index}]")
        where = f"tools[{index}]"
        kind = entry.get("type")
        if kind == "function":
            _ingest_only_keys(provider, entry, frozenset({"type", "function"}), where)
            function = _ingest_object(entry.get("function"), f"{where}.function")
            _ingest_only_keys(provider, function, frozenset({"name", "description", "parameters", "strict"}), f"{where}.function")
            if function.get("strict") is True:
                # A per-tool strict flag has no canonical slot; dropping it
                # would lose an enforcement the caller asked for.  `false` is
                # the wire default and carries nothing.
                raise _ingest_unsupported(provider, f"{where}.function.strict = true", "no per-tool strict slot (compat.strict_tools is a preset policy)")
            kwargs: dict[str, Any] = {"name": _ingest_str(function.get("name"), f"{where}.function.name")}
            if function.get("description") is not None:
                kwargs["description"] = _ingest_str(function["description"], f"{where}.function.description")
            if function.get("parameters") is not None:
                kwargs["parameters"] = _ingest_object(function["parameters"], f"{where}.function.parameters")
            tools.append(FunctionTool(**kwargs))
        elif kind in _INGEST_GROQ_BUILTIN_INVERSE and compat.builtin_tools == "groq":
            config = {k: v for k, v in entry.items() if k != "type"}
            tools.append(BuiltinTool(name=_INGEST_GROQ_BUILTIN_INVERSE[kind], config=config or None))
        else:
            raise _ingest_unsupported(provider, f"{where} of type {kind!r}", "only function tools (and, on the groq preset, its server-executed tools) have a canonical form")
    return tools


def _ingest_tool_choice(provider: str, raw: Any, parallel: Any) -> ToolChoice | None:
    """Inverse of OpenAIChatLM._tool_choice_payload plus parallel_tool_calls."""
    mode: str | None = None
    allowed: tuple[str, ...] = ()
    if raw is not None:
        if raw in ("none", "auto", "required"):
            mode = str(raw)
        elif isinstance(raw, dict):
            kind = raw.get("type")
            if kind == "function":
                _ingest_only_keys(provider, raw, frozenset({"type", "function"}), "tool_choice")
                function = _ingest_object(raw.get("function"), "tool_choice.function")
                _ingest_only_keys(provider, function, frozenset({"name"}), "tool_choice.function")
                mode, allowed = "required", (_ingest_str(function.get("name"), "tool_choice.function.name"),)
            elif kind == "allowed_tools":
                _ingest_only_keys(provider, raw, frozenset({"type", "allowed_tools"}), "tool_choice")
                spec = _ingest_object(raw.get("allowed_tools"), "tool_choice.allowed_tools")
                _ingest_only_keys(provider, spec, frozenset({"mode", "tools"}), "tool_choice.allowed_tools")
                mode = _ingest_str(spec.get("mode"), "tool_choice.allowed_tools.mode")
                entries = spec.get("tools")
                if not isinstance(entries, list) or not entries:
                    raise ValueError("tool_choice.allowed_tools.tools must be a non-empty array")
                names: list[str] = []
                for index, entry in enumerate(entries):
                    entry = _ingest_object(entry, f"tool_choice.allowed_tools.tools[{index}]")
                    if entry.get("type") != "function":
                        raise _ingest_unsupported(provider, f"tool_choice.allowed_tools.tools[{index}] of type {entry.get('type')!r}", "only function tools can be allowed on this wire")
                    function = _ingest_object(entry.get("function"), f"tool_choice.allowed_tools.tools[{index}].function")
                    names.append(_ingest_str(function.get("name"), f"tool_choice.allowed_tools.tools[{index}].function.name"))
                allowed = tuple(names)
            elif kind == "custom":
                raise _ingest_unsupported(provider, "tool_choice of type 'custom'", "custom tools have no canonical form")
            else:
                raise ValueError(f"tool_choice.type must be function or allowed_tools; got {kind!r}")
        else:
            raise ValueError("tool_choice must be none, auto, required, or an object")
    if parallel is not None and not isinstance(parallel, bool):
        raise TypeError("parallel_tool_calls must be a boolean")
    if mode is None and parallel is None:
        return None
    return ToolChoice(mode=mode or "auto", allowed=allowed, parallel=parallel)


def _ingest_response_format(provider: str, raw: Any) -> dict[str, Any] | None:
    """Inverse of _response_format_to_chat (INV-050 shapes).  ``{type: text}``
    is the wire default and reads as absent.  A ``name`` of exactly
    "response" is the builder's default label for an unnamed schema, so it
    reads as absent too: the wire bytes are identical either way."""
    raw = _ingest_object(raw, "response_format")
    kind = raw.get("type")
    if kind == "text":
        _ingest_only_keys(provider, raw, frozenset({"type"}), "response_format")
        return None
    if kind == "json_object":
        _ingest_only_keys(provider, raw, frozenset({"type"}), "response_format")
        return {"type": "json_object"}
    if kind == "json_schema":
        _ingest_only_keys(provider, raw, frozenset({"type", "json_schema"}), "response_format")
        inner = _ingest_object(raw.get("json_schema"), "response_format.json_schema")
        _ingest_only_keys(provider, inner, frozenset({"name", "schema", "strict", "description"}), "response_format.json_schema")
        if inner.get("description") is not None:
            raise _ingest_unsupported(provider, "response_format.json_schema.description", "the canonical response_format has no description slot (INV-050)")
        out: dict[str, Any] = {"type": "json_schema", "schema": _ingest_object(inner.get("schema"), "response_format.json_schema.schema")}
        name = inner.get("name")
        if name is not None and name != "response":
            out["name"] = _ingest_str(name, "response_format.json_schema.name")
        if inner.get("strict") is not None:
            if not isinstance(inner["strict"], bool):
                raise TypeError("response_format.json_schema.strict must be a boolean")
            out["strict"] = inner["strict"]
        return out
    raise ValueError(f"response_format.type must be text, json_object or json_schema; got {kind!r}")


def _ingest_reasoning(provider: str, body: Mapping[str, Any], compat: ResolvedOpenAIChatCompat) -> tuple[Reasoning | None, dict[str, Any]]:
    """Inverse of the reasoning block of _payload for THIS preset's
    thinking_format (MAP-7 spellings).  Returns the Reasoning and any
    extensions entries the spelling implies (groq's `reasoning_format`
    alone is the documented extensions door)."""
    fmt = compat.thinking_format
    present = {k for k in ("reasoning_effort", "reasoning", "thinking", "enable_thinking", "chat_template_kwargs", "reasoning_format") if k in body}
    if not present:
        return None, {}
    spelled_by = {
        "reasoning_effort": {"reasoning_effort"},
        "openrouter": {"reasoning"},
        "deepseek": {"thinking", "reasoning_effort"},
        "kimi": {"thinking", "reasoning_effort"},
        "qwen": {"enable_thinking"},
        "qwen_chat_template": {"chat_template_kwargs"},
        "none": set(),
    }[fmt]
    if compat.builtin_tools == "groq":
        spelled_by = spelled_by | {"reasoning_format"}
    foreign = sorted(present - spelled_by)
    if foreign:
        raise _ingest_unsupported(
            provider, f"{foreign[0]!r}",
            f"this server's reasoning dial is spelled {sorted(spelled_by) or 'nowhere (no dial)'}; another server's spelling would be sent and ignored",
        )
    extensions: dict[str, Any] = {}
    effort: str | None = None
    off = False

    if "reasoning_effort" in present:
        word = _ingest_str(body["reasoning_effort"], "reasoning_effort")
        if word == "none":
            off = True
        else:
            effort = word
    if "thinking" in present:
        spec = _ingest_object(body["thinking"], "thinking")
        _ingest_only_keys(provider, spec, frozenset({"type"}), "thinking")
        if spec.get("type") == "disabled":
            if effort is not None:
                raise ValueError("thinking.type=disabled next to a reasoning_effort level is contradictory")
            off = True
        elif spec.get("type") == "enabled":
            if effort is None and not off:
                raise _ingest_unsupported(provider, "thinking.type=enabled without reasoning_effort", "lm15's dial is a level (MAP-7); set config.reasoning with an effort word")
        else:
            raise ValueError(f"thinking.type must be enabled or disabled; got {spec.get('type')!r}")
    if "reasoning" in present:
        spec = _ingest_object(body["reasoning"], "reasoning")
        _ingest_only_keys(provider, spec, frozenset({"effort", "enabled"}), "reasoning")
        if spec.get("enabled") is False:
            off = True
        elif spec.get("effort") is not None:
            effort = _ingest_str(spec["effort"], "reasoning.effort")
        else:
            raise ValueError("reasoning must carry effort or enabled: false")
    if "enable_thinking" in present:
        flag = body["enable_thinking"]
        if flag is False:
            off = True
        elif flag is True:
            raise _ingest_unsupported(provider, "enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")
        else:
            raise TypeError("enable_thinking must be a boolean")
    if "chat_template_kwargs" in present:
        spec = _ingest_object(body["chat_template_kwargs"], "chat_template_kwargs")
        _ingest_only_keys(provider, spec, frozenset({"enable_thinking", "preserve_thinking"}), "chat_template_kwargs")
        if spec.get("enable_thinking") is False:
            off = True
        elif spec.get("enable_thinking") is True:
            raise _ingest_unsupported(provider, "chat_template_kwargs.enable_thinking = true", "this wire has no effort level; lm15's dial is a level (MAP-7) — set config.reasoning yourself")
        else:
            raise TypeError("chat_template_kwargs.enable_thinking must be a boolean")

    summary: str | None = None
    if "reasoning_format" in present:
        value = body["reasoning_format"]
        if value != "parsed":
            raise _ingest_unsupported(provider, f"reasoning_format = {value!r}", "only 'parsed' maps (Reasoning.summary='auto', MAP-7 rule 7)")
        if effort is None:
            # The documented door: extensions={"reasoning_format": "parsed"}
            # with reasoning absent (the Qwen dial has no levels).
            extensions["reasoning_format"] = value
        else:
            summary = "auto"

    if off:
        return Reasoning(effort="off"), extensions
    if effort is None:
        return None, extensions
    return Reasoning(effort=effort, summary=summary), extensions


def _ingest_cache(
    provider: str, body: Mapping[str, Any], compat: ResolvedOpenAIChatCompat,
    *, system_breakpoint: bool, breakpoint_index: int | None,
) -> CacheConfig | None:
    """Inverse of _cache_common_payload and the breakpoint placement (MAP-6
    on the OpenAI classes).  Under a preset with no OpenAI cache control the
    keys are refused: the server has no such field."""
    keys = {k for k in ("prompt_cache_key", "prompt_cache_retention", "prompt_cache_options") if k in body}
    marked = system_breakpoint or breakpoint_index is not None
    if not keys and not marked:
        return None
    if compat.cache_control not in ("openai", "openai_implicit"):
        what = sorted(keys)[0] if keys else "prompt_cache_breakpoint"
        raise _ingest_unsupported(provider, f"{what!r}", "this server has no OpenAI prompt-cache control (compat.cache_control)")
    if marked and compat.cache_control != "openai":
        raise _ingest_unsupported(provider, "prompt_cache_breakpoint", "this server swallows an explicit breakpoint silently (compat.cache_control=openai_implicit)")
    key = body.get("prompt_cache_key")
    retention = None
    if "prompt_cache_retention" in body:
        if body["prompt_cache_retention"] != "24h":
            raise _ingest_unsupported(provider, f"prompt_cache_retention = {body['prompt_cache_retention']!r}", "only '24h' has a canonical value (CacheConfig.retention='long')")
        retention = "long"
    explicit = False
    if "prompt_cache_options" in body:
        spec = _ingest_object(body["prompt_cache_options"], "prompt_cache_options")
        _ingest_only_keys(provider, spec, frozenset({"mode", "ttl"}), "prompt_cache_options")
        if spec.get("ttl") is not None:
            raise _ingest_unsupported(provider, "prompt_cache_options.ttl", "CacheConfig.retention names 24h only")
        if spec.get("mode") == "explicit":
            explicit = True
        elif spec.get("mode") == "implicit":
            raise _ingest_unsupported(provider, "prompt_cache_options.mode = 'implicit'", "the server default; a canonical CacheConfig names auto or off")
        else:
            raise ValueError(f"prompt_cache_options.mode must be explicit or implicit; got {spec.get('mode')!r}")
    if explicit and not marked:
        # Explicit mode with no mark is the cache-WRITE off switch (MAP-6 rule 2).
        if key is not None or retention is not None:
            raise ValueError("prompt_cache_options.mode=explicit with no breakpoint is the off switch; it cannot carry a key or retention (INV-027)")
        return CacheConfig(mode="off")
    if system_breakpoint:
        return CacheConfig(prefix="stable", key=key, retention=retention)
    if breakpoint_index is not None:
        return CacheConfig(prefix_until_index=breakpoint_index, key=key, retention=retention)
    return CacheConfig(key=key, retention=retention)


def _ingest_config(
    provider: str, body: Mapping[str, Any], compat: ResolvedOpenAIChatCompat,
    *, system_breakpoint: bool, breakpoint_index: int | None,
) -> Config:
    kwargs: dict[str, Any] = {}
    if "max_completion_tokens" in body or "max_tokens" in body:
        values = {k: body[k] for k in ("max_completion_tokens", "max_tokens") if k in body}
        if len(set(map(repr, values.values()))) > 1:
            raise ValueError(f"max_tokens and max_completion_tokens disagree: {values}")
        kwargs["max_tokens"] = next(iter(values.values()))
    for key in ("temperature", "top_p", "service_tier", "store"):
        if key in body:
            kwargs[key] = body[key]
    if "stop" in body:
        kwargs["stop"] = body["stop"]
    if body.get("logprobs") is True:
        top = body.get("top_logprobs", 0)
        kwargs["logprobs"] = top
    elif body.get("logprobs") not in (None, False):
        raise TypeError("logprobs must be a boolean")
    elif "top_logprobs" in body:
        raise ValueError("top_logprobs requires logprobs: true")
    if "response_format" in body:
        kwargs["response_format"] = _ingest_response_format(provider, body["response_format"])
    kwargs["tool_choice"] = _ingest_tool_choice(provider, body.get("tool_choice"), body.get("parallel_tool_calls"))

    user_keys = [k for k in ("user", "safety_identifier", "user_id") if k in body]
    if "user_id" in user_keys and compat.user_field != "user_id":
        raise _ingest_unsupported(provider, "'user_id'", f"this server spells the end-user field {compat.user_field!r}")
    if len(user_keys) > 1:
        raise ValueError(f"one end-user identifier only; got {user_keys}")
    if user_keys:
        kwargs["user_id"] = body[user_keys[0]]

    reasoning, extensions = _ingest_reasoning(provider, body, compat)
    kwargs["reasoning"] = reasoning
    kwargs["cache"] = _ingest_cache(provider, body, compat, system_breakpoint=system_breakpoint, breakpoint_index=breakpoint_index)
    for key in body:
        if key in _INGEST_EXTENSIONS_KEYS:
            extensions[key] = body[key]
    kwargs["extensions"] = extensions or None
    return Config(**kwargs)


def _ingest_openai_chat(provider: str, body: Mapping[str, Any], compat: ResolvedOpenAIChatCompat) -> Request:
    if not isinstance(body, Mapping):
        raise TypeError(f"a Chat Completions request body is a JSON object, got {type(body).__name__}")
    for key in body:
        if key in _INGEST_REFUSED_KEYS:
            raise _ingest_unsupported(provider, f"{key!r}", _INGEST_REFUSED_KEYS[key])
        if key not in _INGEST_CONFIG_KEYS and key not in _INGEST_EXTENSIONS_KEYS and key not in _INGEST_CALL_MODE_KEYS:
            raise _ingest_unsupported(provider, f"{key!r}", "no verdict for this key (lm15-contract/tools/openai-chat-ingest-verdicts.json); lm15 never drops a key silently")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if "messages" not in body:
        raise ValueError("messages is required")
    system, messages, system_breakpoint, breakpoint_index = _ingest_messages(provider, body["messages"], compat)
    tools = _ingest_tools(provider, body.get("tools"), compat)
    config = _ingest_config(provider, body, compat, system_breakpoint=system_breakpoint, breakpoint_index=breakpoint_index)
    return Request(model=model, messages=tuple(messages), system=system, tools=tuple(tools), config=config)


def request_from_openai_chat(body: Mapping[str, Any], *, compat: OpenAIChatCompat | str | None = None) -> Request:
    """A Chat Completions request body → the canonical :class:`Request` (MAP-12).

    ``body`` is the JSON object a client would POST to ``/chat/completions``
    (``model``, ``messages``, ``tools``, generation knobs).  ``compat`` names
    the server dialect whose spellings are read — a preset name
    (``"openai"``, ``"groq"``, ``"deepseek"``, …), an
    :class:`OpenAIChatCompat`, or None for OpenAI's own — the same policy
    :class:`OpenAIChatLM` writes with, so what that adapter emits for a
    Request reads back as that Request wherever the wire can carry it.

    Every key has one verdict: it maps to a canonical field, it passes
    verbatim through ``config.extensions`` (``seed``, ``logit_bias``,
    ``presence_penalty``, ``frequency_penalty``, ``metadata``, …), or it is
    refused with :class:`UnsupportedFeatureError` naming the key (``n``,
    the deprecated ``functions`` shape, a content block with no part, a
    spelling another server owns).  ``stream`` / ``stream_options`` are
    call-mode keys with no place on a Request and are read and dropped.
    Malformed input raises ``ValueError`` / ``TypeError``.

    On the adapter, ``lm.request_from_openai_chat(body)`` is the same
    function under that adapter's compat, including its per-model overrides.
    """
    if isinstance(compat, str):
        partial = OpenAIChatCompat.preset(compat)
    elif isinstance(compat, OpenAIChatCompat):
        partial = compat
    elif compat is None:
        partial = OpenAIChatCompat()
    else:
        raise TypeError("compat must be a preset name, an OpenAIChatCompat, or None")
    model = body.get("model") if isinstance(body, Mapping) else None
    resolved = resolve_openai_chat_compat(partial.for_model(model) if isinstance(model, str) else partial)
    return _ingest_openai_chat("openai-chat", body, resolved)


def _response_from_chat_body(
    provider: str, data: Any, *, model: str | None, choice: int | None,
    on_error: Callable[[str, str], ProviderError],
) -> Response:
    """The one Chat Completions response reader: ``parse_response`` for
    provider traffic and :func:`response_from_openai_chat` for a foreign
    dict share it.  ``choice`` names the choice to read; ``None`` means
    "the only one", and a body with several choices is then refused
    rather than silently reduced to its first (the reading-side twin of
    MAP-12's refusal of ``n``)."""
    if not isinstance(data, Mapping):
        raise TypeError(f"a Chat Completions response body is a JSON object, got {type(data).__name__}")
    resp_error = data.get("error")
    if isinstance(resp_error, dict):
        raise on_error(str(resp_error.get("code") or ""), str(resp_error.get("message") or resp_error))

    unmapped: list[dict[str, str]] = []
    choices = data.get("choices") or []
    if not isinstance(choices, list):
        raise TypeError("choices must be an array")
    if choice is None:
        if len(choices) > 1:
            raise UnsupportedFeatureError(
                f"{provider}: the body carries {len(choices)} choices; a canonical Response is one message — "
                "name the choice to read (choice=i) and read each one, or send no n",
                provider=provider,
            )
        index = 0
    else:
        if choice < 0 or choice >= len(choices):
            raise ValueError(f"choice={choice} but the body carries {len(choices)} choice(s)")
        index = choice
    path = f"choices[{index}]"
    chosen = choices[index] if choices and isinstance(choices[index], dict) else {}
    if choices and not isinstance(choices[index], dict):
        _record_unmapped(unmapped, path, type(choices[index]).__name__)
    message = chosen.get("message") if isinstance(chosen.get("message"), dict) else {}

    parts: list[Any] = []
    reasoning_text = message.get("reasoning_content") or message.get("reasoning")
    if reasoning_text:
        parts.append(ThinkingPart(text=str(reasoning_text)))

    content = message.get("content")
    if isinstance(content, str):
        if content:
            parts.append(TextPart(text=content))
    elif isinstance(content, list):
        for content_index, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(TextPart(text=str(item.get("text") or "")))
            else:
                _record_unmapped(
                    unmapped,
                    f"{path}.message.content[{content_index}]",
                    item.get("type") if isinstance(item, dict) else type(item).__name__,
                )
    elif content is not None:
        _record_unmapped(unmapped, f"{path}.message.content", type(content).__name__)

    refusal = message.get("refusal")
    if refusal:
        parts.append(RefusalPart(text=str(refusal)))

    for call_index, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            _record_unmapped(unmapped, f"{path}.message.tool_calls[{call_index}]", type(call).__name__)
            continue
        call_type = call.get("type") or "function"
        if call_type != "function":
            _record_unmapped(unmapped, f"{path}.message.tool_calls[{call_index}]", call_type)
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        if not function.get("name"):
            raise unnamed_tool_call_error(provider, f"{path}.message.tool_calls[{call_index}]")
        parts.append(
            ToolCallPart(
                id=str(call.get("id") or f"call_{len(parts)}"),
                name=str(function["name"]),
                input=parse_json_object(function.get("arguments")),
            )
        )

    if not parts:
        # MAP-2: a response message is never empty.
        parts = [TextPart(text="")]

    has_tool = any(isinstance(part, ToolCallPart) for part in parts)
    usage = _usage_from_chat(data.get("usage") or {})
    # choices[i].logprobs.content is the message-level token sequence.
    # Refusal logprobs (choices[i].logprobs.refusal) stay in
    # provider_data — no canonical refusal-token concept.
    logprobs_payload = chosen.get("logprobs") if isinstance(chosen.get("logprobs"), dict) else {}
    logprob_seq = openai_token_logprobs(logprobs_payload.get("content"))
    resolved_model = data.get("model") or model
    if not resolved_model:
        raise ValueError("the body carries no model; pass model=")
    return Response(
        id=str(data.get("id")) if data.get("id") else None,
        model=str(resolved_model),
        message=Message(role="assistant", parts=tuple(parts)),
        finish_reason=OpenAIChatLM._finish_reason(chosen.get("finish_reason"), has_tool_call=has_tool, unmapped=unmapped, path=path),
        usage=usage,
        logprobs=logprob_seq or None,
        provider_data=_attach_unmapped(dict(data), unmapped),
    )


def response_from_openai_chat(body: Mapping[str, Any], *, model: str | None = None, choice: int | None = None) -> Response:
    """A Chat Completions response body → the canonical :class:`Response`.

    The reading-side twin of :func:`request_from_openai_chat`: ``body`` is
    the JSON object a Chat Completions server (or a client library that
    imitates one — litellm's ``ModelResponse.model_dump()``) returned.  It
    is the same reader :meth:`OpenAIChatLM.parse_response` runs on provider
    traffic, so what it maps (text, ``reasoning_content`` → thinking, tool
    calls, refusal, usage, logprobs, finish reason) and what it records as
    unmapped (``provider_data["_lm15_unmapped"]``) are pinned by the
    contract's ``response`` direction.  An error envelope raises the typed
    provider error.

    ``model`` fills ``Response.model`` when the body carries none.
    ``choice`` names the choice to read; left unset, a body with more than
    one choice is refused (a Response is one message; read each choice by
    index instead) — the twin of the request side's refusal of ``n``.
    Keys the reader does not know are neither refused nor lost: the whole
    body is ``provider_data``.  No compat is taken: the Chat Completions
    response shape does not vary by server the way the request does.
    """
    def on_error(code: str, message: str) -> ProviderError:
        cls = OpenAILM._response_error_code_map.get(code, ServerError)
        return cls(message or code or "provider error", provider="openai-chat", provider_code=code or None)

    return _response_from_chat_body("openai-chat", body, model=model, choice=choice, on_error=on_error)


def _usage_from_chat(usage_data: dict[str, Any]) -> Usage:
    prompt_details = usage_data.get("prompt_tokens_details") or {}
    completion_details = usage_data.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=usage_data.get("prompt_tokens"),
        output_tokens=usage_data.get("completion_tokens"),
        total_tokens=usage_data.get("total_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        cache_read_tokens=prompt_details.get("cached_tokens"),
        cache_write_tokens=prompt_details.get("cache_write_tokens"),
        input_audio_tokens=prompt_details.get("audio_tokens"),
        output_audio_tokens=completion_details.get("audio_tokens"),
    )


@dataclass(slots=True)
class OpenAIChatLM(BaseProviderLM):
    """Adapter for the OpenAI Chat Completions wire dialect.

    ``compat`` may be an :class:`OpenAIChatCompat`, a preset name
    (``"ollama"``, ``"groq"``, ``"openrouter"``, ``"vllm"``, ``"sglang"``,
    ``"openai"``, …), or None (plain OpenAI policy).  A preset name also
    supplies that server's default ``base_url``; an explicit non-default
    ``base_url`` argument always wins.
    """

    api_key: Credential | None = field(default=None, repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = _DEFAULT_BASE_URL
    compat: OpenAIChatCompat | str | None = None
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)
    settings: "Mapping[str, str] | None" = None
    clock: "Callable[[], datetime] | None" = field(default=None, repr=False)

    provider: str = field(default="openai-chat", init=False)
    account_id: str | None = field(default=None, init=False, repr=False)
    manifest: ClassVar[ProviderManifest] = OPENAI_CHAT_API

    # OpenAI-compatible servers reuse the same error envelope family;
    # share the Responses adapter's mapping verbatim.
    _response_error_code_map: ClassVar[dict[str, type[ProviderError]]] = OpenAILM._response_error_code_map
    _model_error_codes: ClassVar[frozenset[str]] = OpenAILM._model_error_codes
    _stream_error_code_map: ClassVar[dict[str, type[ProviderError]]] = OpenAILM._stream_error_code_map

    _is_model_error = staticmethod(OpenAILM._is_model_error)
    _response_error = OpenAILM._response_error
    _error_detail = OpenAILM._error_detail
    normalize_error = OpenAILM.normalize_error

    def __post_init__(self) -> None:
        self._bind_access(self.access, credentials_path=self.credentials_path, default_base_url=_DEFAULT_BASE_URL, settings=self.settings)
        compat = self.compat if self.compat is not None else self._registry_compat()
        if isinstance(compat, str):
            preset_key = compat.lower().replace("-", "_").replace(" ", "_")
            partial = OpenAIChatCompat.preset(compat)
            if self.base_url == _DEFAULT_BASE_URL:
                self.base_url = OPENAI_CHAT_PRESET_BASE_URLS.get(preset_key, _DEFAULT_BASE_URL)
        elif isinstance(compat, OpenAIChatCompat):
            partial = compat
        else:
            partial = OpenAIChatCompat()
        self._compat_partial = partial
        self._resolved_compat = resolve_openai_chat_compat(partial)

    _resolved_compat: ResolvedOpenAIChatCompat = field(init=False, repr=False, default=ResolvedOpenAIChatCompat())
    _compat_partial: OpenAIChatCompat | None = field(init=False, repr=False, default=None)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}  # auth is applied once, in _emit (AUTH-2)
        headers["Content-Type"] = "application/json"
        for key, static in self.access.headers:
            headers[key] = static
        return headers

    # ─── Live model listing (provisional endpoint) ──────────────────────

    def _models_request(self):
        return self._emit(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/models",
            headers=self._headers(),
            read_timeout=30.0,
        )

    def _models_from_body(self, body: str):
        data = json.loads(body)
        entries = data.get("data") if isinstance(data, dict) else None
        return model_infos_from_entries(
            entries,
            provider=self.provider,
            api_family="openai_chat",
            id_of=lambda entry: entry.get("id"),
        )

    # ─── Request serialization ──────────────────────────────────────

    def _build_messages(self, request: Request, compat: ResolvedOpenAIChatCompat) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            system_text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            if _cache_stable_prefix(request, compat.cache_control):
                # prefix="stable": the mark rides on the system message's
                # text content part (array form; a bare string cannot carry it).
                messages.append({"role": compat.instruction_role, "content": [
                    {"type": "text", "text": system_text, "prompt_cache_breakpoint": {"mode": "explicit"}}
                ]})
            else:
                messages.append({"role": compat.instruction_role, "content": system_text})

        breakpoint_index = _cache_breakpoint_index(request, compat.cache_control)
        for msg_index, msg in enumerate(request.messages):
            if msg_index == breakpoint_index and msg.role in ("assistant", "tool"):
                raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
            if msg.role == "tool":
                for part in msg.parts:
                    if isinstance(part, ToolResultPart):
                        item: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": part.id,
                            "content": _tool_row_content(self.provider, part, compat.tool_result_media),
                        }
                        if compat.tool_result_name == "include" and part.name:
                            item["name"] = part.name
                        messages.append(item)
                continue

            if msg.role == "assistant":
                text_bits: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_bits.append(part.text)
                    elif isinstance(part, RefusalPart) and part.text:
                        text_bits.append(part.text)
                    elif isinstance(part, ThinkingPart) and compat.thinking_replay == "as_text" and part.text:
                        text_bits.append(part.text)
                tool_calls = [
                    {
                        "id": part.id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.input, separators=(",", ":")),
                        },
                    }
                    for part in msg.parts
                    if isinstance(part, ToolCallPart)
                ]
                item = {"role": "assistant", "content": "\n".join(text_bits) if text_bits else None}
                if compat.thinking_replay == "native":
                    thinking = "\n".join(p.text for p in msg.parts if isinstance(p, ThinkingPart) and p.text)
                    if thinking or compat.assistant_reasoning_content == "include_empty":
                        item["reasoning_content"] = thinking
                if tool_calls:
                    item["tool_calls"] = tool_calls
                messages.append(item)
                continue

            role = compat.instruction_role if msg.role == "developer" else msg.role
            at_breakpoint = msg_index == breakpoint_index
            content = _chat_content_parts(msg, force_array=at_breakpoint, provider=self.provider)
            if at_breakpoint:
                # Same rule as the Responses dialect: the breakpoint rides on
                # the last text content block of the prefix message
                # (chat--create.md: ChatCompletionContentPartText carries
                # prompt_cache_breakpoint).
                if not isinstance(content, list) or not content or content[-1].get("type") != "text":
                    raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
                content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
            if content or content == "":
                messages.append({"role": role, "content": content})
        return messages

    def request_from_openai_chat(self, body: Mapping[str, Any]) -> Request:
        """The inverse of :meth:`build_request`'s body under this adapter's
        compat (MAP-12): a Chat Completions request body → canonical
        :class:`Request`.  See :func:`request_from_openai_chat`."""
        model = body.get("model") if isinstance(body, Mapping) else None
        compat = self._compat_for(model) if isinstance(model, str) else self._resolved_compat
        return _ingest_openai_chat(self.provider, body, compat)

    def _builtin_tool_payload(self, tool: BuiltinTool, compat: ResolvedOpenAIChatCompat) -> dict[str, Any]:
        """Map a BuiltinTool for the chat dialect, or raise.

        The base Chat Completions wire carries function/custom tools only,
        and some compat servers silently IGNORE unknown tool types
        (OpenRouter returned 200 with no search, verified live 2026-09-01)
        — passthrough would fabricate wire shapes and silent no-ops, so
        every unproven target raises.
        """
        if compat.builtin_tools == "groq":
            wire_type = _GROQ_BUILTIN_MAP.get(tool.name)
            if wire_type is None:
                raise UnsupportedFeatureError(
                    f"{self.provider}: builtin tool {tool.name!r} has no Groq wire "
                    f"mapping — supported: {sorted(_GROQ_BUILTIN_MAP)}",
                    provider=self.provider,
                )
            entry: dict[str, Any] = {"type": wire_type}
            if tool.config:
                entry.update(tool.config)
            return entry
        raise UnsupportedFeatureError(
            f"{self.provider}: builtin tool {tool.name!r} is not supported on this "
            "server — the Chat Completions wire carries function tools only, and "
            "unproven servers may silently ignore unknown tool types. Use "
            "compat='groq' for Groq's server-executed tools, or the OpenAI "
            "Responses / Anthropic / Gemini providers",
            provider=self.provider,
        )

    def _tool_choice_payload(self, request: Request) -> Any:
        tc = request.config.tool_choice
        if tc is None:
            return None
        if tc.mode == "none":
            return "none"
        if tc.allowed:
            by_name = {t.name: t for t in request.tools}
            entries = [by_name[name] for name in tc.allowed]
            builtins = [t.name for t in entries if isinstance(t, BuiltinTool)]
            if builtins:
                raise UnsupportedFeatureError(
                    f"{self.provider}: cannot force builtin tools {builtins} — the "
                    "Chat Completions wire has no hosted-tool tool_choice form "
                    "(OpenAI Responses and Anthropic carry it)",
                    provider=self.provider,
                )
            if len(entries) == 1 and tc.mode == "required":
                return {"type": "function", "function": {"name": entries[0].name}}
            # mode="auto" restriction or multi-tool subset: the dialect's
            # allowed_tools form (nested spelling, function tools only).
            return {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": tc.mode,
                    "tools": [
                        {"type": "function", "function": {"name": t.name}} for t in entries
                    ],
                },
            }
        if tc.mode == "required":
            return "required"
        return "auto"

    def _compat_for(self, model: str) -> ResolvedOpenAIChatCompat:
        """The resolved compat for this model: the preset's per-family
        overrides applied (OpenAIChatCompat.model_overrides)."""
        partial = self._compat_partial
        if partial is None or not partial.model_overrides:
            return self._resolved_compat
        return resolve_openai_chat_compat(partial.for_model(model))

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        compat = self._compat_for(request.model)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request, compat),
        }
        if stream:
            payload["stream"] = True
            if compat.stream_usage == "include":
                payload["stream_options"] = {"include_usage": True}
        if request.config.max_tokens is not None:
            payload[compat.max_tokens_field] = request.config.max_tokens
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.config.top_k is not None:
            # No wire slot on Chat Completions (port.md rule 4: a raise or an
            # extensions door, never omission).
            raise UnsupportedFeatureError(
                f"{self.provider}: config.top_k has no field on the Chat Completions wire; servers that accept "
                "top_k take it through extensions", provider=self.provider,
            )
        if request.config.stop:
            payload["stop"] = list(request.config.stop)
        if request.config.logprobs is not None:
            # Verified live 2026-09-01: logprobs=true alone returns chosen
            # tokens only; top_logprobs adds the alternatives.
            payload["logprobs"] = True
            if request.config.logprobs > 0:
                payload["top_logprobs"] = request.config.logprobs
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    function_payload: dict[str, Any] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    if compat.strict_tools == "include":
                        function_payload["strict"] = False
                    tools_wire.append({"type": "function", "function": function_payload})
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(self._builtin_tool_payload(tool, compat))
            if tools_wire:
                payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            tc = request.config.tool_choice
            if compat.forced_tool_choice == "reject" and (tc.mode != "auto" or tc.allowed):
                # MAP-8: the server documents tool_choice=auto only and ignores
                # every other form without an error (Z.AI, live 2026-09-03:
                # required → text answer, none → a tool call).  A silent widen
                # is worse than an error; omit tool_choice or use "auto".
                raise UnsupportedFeatureError(
                    f"{self.provider}: tool_choice mode={tc.mode!r}"
                    + (f" allowed={list(tc.allowed)}" if tc.allowed else "")
                    + " is silently ignored by this server (only 'auto' is honoured); "
                    "omit tool_choice, or send only the tools you want callable",
                    provider=self.provider,
                )
            payload["tool_choice"] = tool_choice
        if request.config.tool_choice and request.config.tool_choice.parallel is not None:
            payload["parallel_tool_calls"] = request.config.tool_choice.parallel
        if request.config.response_format:
            if compat.json_schema == "reject" and request.config.response_format["type"] != "json_object":
                # The server accepts response_format.type=json_schema and
                # ignores it (Z.AI, live 2026-09-03: HTTP 200, fenced JSON with
                # keys the schema never named).  json_object is honoured.
                raise UnsupportedFeatureError(
                    f"{self.provider}: response_format type "
                    f"{request.config.response_format['type']!r} is silently ignored by this "
                    "server; use {'type': 'json_object'} and describe the shape in the prompt",
                    provider=self.provider,
                )
            payload["response_format"] = _response_format_to_chat(request.config.response_format)
        if request.config.reasoning:
            reasoning = request.config.reasoning
            if not reasoning.is_off:
                # MAP-7: verbatim effort; no budget on this wire; summary
                # levels are Responses-only; "auto" maps to the dialect's
                # visibility knob where one exists (Groq include_reasoning).
                if reasoning.thinking_budget is not None:
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.thinking_budget is not supported — the Chat "
                        "Completions wire has no thinking token budget; use effort",
                        provider=self.provider,
                    )
                if reasoning.summary in ("concise", "detailed"):
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.summary={reasoning.summary!r} is an OpenAI Responses "
                        "detail level; the Chat Completions wire has none (use 'auto')",
                        provider=self.provider,
                    )
                effort = reasoning.effort
                if compat.reasoning_efforts is not None and effort not in compat.reasoning_efforts:
                    # MAP-7 rule 2: a word with no native level raises here
                    # when the server would not refuse it (Moonshot kimi-k3
                    # answered 200 to `medium` and to `bogus`, live 2026-09-03).
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.effort={effort!r} has no level on this server "
                        f"(it accepts {', '.join(compat.reasoning_efforts)}) and would be accepted silently",
                        provider=self.provider,
                    )
                if compat.builtin_tools == "groq" and reasoning.summary == "auto":
                    # Groq's visibility knob (MAP-7 rule 7): "parsed" returns
                    # the trace as message.reasoning.  Live 2026-09-02: Qwen
                    # 3.6's default leaks a raw <think> block into
                    # message.content; this is the wire's fix.  Qwen's dial
                    # accepts only none|default, so an effort word still
                    # fails loudly there — extensions={"reasoning_format":
                    # "parsed"} with reasoning absent is the documented door.
                    payload["reasoning_format"] = "parsed"
                if compat.thinking_format == "reasoning_effort":
                    payload["reasoning_effort"] = effort
                elif compat.thinking_format == "openrouter":
                    payload["reasoning"] = {"effort": effort}
                elif compat.thinking_format == "deepseek":
                    payload["thinking"] = {"type": "enabled"}
                    payload["reasoning_effort"] = effort
                elif compat.thinking_format == "kimi":
                    # Moonshot: the effort word alone, the field kimi-k3
                    # documents (guide--reasoning-effort.md).  A `thinking`
                    # object beside it is accepted (live 2026-09-03) but the
                    # docs say not to send it, and it would carry nothing.
                    # Stated trade-off: kimi-k2.6 has no levels and ignores
                    # the word silently (live 2026-09-03, still 44 reasoning
                    # tokens at `low`); the adapter does not sniff model
                    # names, so the docs say: do not set effort on K2.x.
                    payload["reasoning_effort"] = effort
                elif compat.thinking_format == "qwen":
                    payload["enable_thinking"] = True
                elif compat.thinking_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    }
            else:
                # Explicit off must reach the wire; omission lets
                # reasoning-by-default models spend hidden reasoning tokens
                # (verified live 2026-09-01 on Groq: gpt-oss-20b spent 45
                # reasoning tokens when the field was omitted).  Servers
                # whose models cannot disable reasoning reject "none" with
                # a clear 400 — loud failure over a silent paid no-op.
                if compat.thinking_format == "reasoning_effort":
                    payload["reasoning_effort"] = "none"
                elif compat.thinking_format == "openrouter":
                    payload["reasoning"] = {"enabled": False}
                elif compat.thinking_format in ("deepseek", "kimi"):
                    # "kimi": the K2.x family's documented off switch; the
                    # docs say kimi-k3 "always reasons", but live 2026-09-03
                    # it honoured the object too (200, no reasoning_content,
                    # no reasoning_tokens) — pinned as moonshotai.reasoning_off.
                    payload["thinking"] = {"type": "disabled"}
                elif compat.thinking_format == "qwen":
                    payload["enable_thinking"] = False
                elif compat.thinking_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Prompt caching (MAP-6): off switch, key, retention, resource.
        _cache_common_payload(request, payload, compat.cache_control, self.provider)

        if compat.routing is not None:
            payload["provider"] = compat.routing

        # Promoted cross-provider knobs (changes/2026-09-01-extensions-burn-down):
        # Chat Completions spellings — user_id rides the dialect's `user`
        # field, or the server's own name when the compat says so (DeepSeek:
        # `user_id`; Meta: `safety_identifier`, which supersedes `user` on
        # that server — protocols--chat-completions.md). Servers that
        # do not implement a field reject it themselves; the dialect adapter
        # cannot know each compat server's support statically.
        if request.config.service_tier is not None:
            payload["service_tier"] = request.config.service_tier
        if request.config.user_id is not None:
            payload[compat.user_field] = request.config.user_id
        if request.config.store is not None:
            payload["store"] = request.config.store

        if request.config.extensions:
            reserved = {
                "prompt_caching",
                "cache",
                "compat",
                "openai_compat",
                "openai_chat_compat",
            }
            passthrough = {k: v for k, v in request.config.extensions.items() if k not in reserved}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return self._emit(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            endpoint="chat/completions",
            stream=stream,
            model=request.model,
            headers=self._headers(),
            payload=self._payload(request, stream=stream),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    @staticmethod
    def _finish_reason(raw: Any, *, has_tool_call: bool, unmapped: list[dict[str, str]], path: str = "choices[0]") -> str:
        if has_tool_call:
            return "tool_call"
        if raw is None or raw == "":
            return "stop"
        mapped = _FINISH_REASON_MAP.get(str(raw))
        if mapped is None:
            _record_unmapped(unmapped, f"{path}.finish_reason", raw)
            return "stop"
        return mapped

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        return _response_from_chat_body(
            self.provider, response.json(), model=request.model,
            choice=None, on_error=self._response_error,
        )

    def response_from_openai_chat(self, body: Mapping[str, Any], *, model: str | None = None, choice: int | None = None) -> Response:
        """A Chat Completions response body → canonical :class:`Response`
        under this adapter's provider name and error mapping; see
        :func:`response_from_openai_chat`."""
        return _response_from_chat_body(self.provider, body, model=model, choice=choice, on_error=self._response_error)

    # ─── Stream parsing ──────────────────────────────────────────────

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        if not raw_event.data:
            return
        if raw_event.data == "[DONE]":
            yield StreamEndEvent()
            return
        payload = json.loads(raw_event.data)
        if not isinstance(payload, dict):
            return

        err = payload.get("error")
        if isinstance(err, dict):
            provider_code = str(err.get("code") or err.get("type") or "provider")
            yield StreamErrorEvent(error=self._error_detail(provider_code, str(err.get("message") or "")))
            return

        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

        reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning_text:
            yield StreamDeltaEvent(delta=ThinkingDelta(text=str(reasoning_text)))

        content = delta.get("content")
        if isinstance(content, str) and content:
            logprobs_payload = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
            yield StreamDeltaEvent(
                delta=TextDelta(
                    text=content,
                    logprobs=openai_token_logprobs(logprobs_payload.get("content")),
                )
            )

        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            yield StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=str(function.get("arguments") or ""),
                    part_index=int(call.get("index", 0) or 0),
                    id=str(call.get("id") or "") or None,
                    name=str(function.get("name") or "") or None,
                )
            )

        # MAP-3 (D9, 2026-09-06): the end event's provider_data is the frame
        # that supplied usage, verbatim, else the frame that supplied
        # finish_reason; [DONE] contributes nothing.  The coalescer keeps the
        # usage frame's when both arrive.
        finish_raw = choice.get("finish_reason")
        usage_data = payload.get("usage")
        if finish_raw:
            yield StreamEndEvent(
                finish_reason=_FINISH_REASON_MAP.get(str(finish_raw), "stop"),
                usage=_usage_from_chat(usage_data) if isinstance(usage_data, dict) else None,
                provider_data=payload,
            )
        elif isinstance(usage_data, dict):
            # Final usage-only chunk (stream_options.include_usage).
            yield StreamEndEvent(usage=_usage_from_chat(usage_data), provider_data=payload)
