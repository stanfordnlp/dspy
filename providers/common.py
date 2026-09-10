from __future__ import annotations

import base64
import json
import urllib.parse
from typing import Any

from ..models import ModelInfo, ModelOrigin
from ..transports import TransportRequest
from ..types import (
    AudioPart,
    BinaryPart,
    CitationPart,
    DocumentPart,
    ImagePart,
    Message,
    Part,
    TextPart,
    ThinkingPart,
    FileReadiness,
    TokenLogprob,
    ToolResultPart,
    TopLogprob,
    VideoPart,
)

JsonPayload = dict[str, Any] | list[Any]


# The FileReadiness fold for every OpenAI-shaped file object (api.openai.com,
# Azure OpenAI v1, Meta): spec/vocabularies.md FileReadiness, ratified
# 2026-09-06 (lm15-contract/changes/2026-09-06-decisions.md D6).  Azure says `pending` after
# a 201 upload where api.openai.com says `uploaded` (live 2026-09-04); the
# `status` field is deprecated upstream, so absent and unknown words read as
# ready.
_OPENAI_FILE_READINESS: dict[str, FileReadiness] = {
    "uploaded": "pending",
    "pending": "pending",
    "error": "failed",
    "failed": "failed",
    "processed": "ready",
}


def openai_file_readiness(status: object) -> FileReadiness:
    """``uploaded | pending -> pending``, ``error | failed -> failed``,
    ``processed | absent | unknown -> ready``."""
    if not isinstance(status, str):
        return "ready"
    return _OPENAI_FILE_READINESS.get(status, "ready")


# The part kinds that carry bytes/addresses rather than words (MAP-10: a
# native block or a raise; never text).
MEDIA_KINDS: frozenset[str] = frozenset({"image", "audio", "video", "document", "binary"})


def parts_to_text(parts: tuple[Part, ...], *, provider: str | None = None,
                  where: str = "a text-only wire field") -> str:
    """Text rendering for wire fields that take text only.

    Renders text-bearing parts (text, thinking, citation). A media part
    RAISES `UnsupportedFeatureError` before any wire (MAP-10 rule 2): the
    field cannot carry it, and rendering a caption or a type name in its
    place is the silent substitution the rule forbids. Callers that own a
    slot for the part must not be here.
    """
    from ..errors import UnsupportedFeatureError

    out: list[str] = []
    for part in parts:
        if part.type in MEDIA_KINDS:
            head = f"{provider}: " if provider else ""
            raise UnsupportedFeatureError(
                f"{head}a {part.type} part cannot reach {where}, which takes text only; "
                "no text rendering of a media part is made (MAP-10)",
                provider=provider,
            )
        if isinstance(part, TextPart):
            out.append(part.text)
        elif isinstance(part, ThinkingPart) and part.text:
            out.append(part.text)
        elif isinstance(part, CitationPart):
            bits = [x for x in (part.title, part.url, part.text) if x]
            if bits:
                out.append(" — ".join(bits))
    return "\n".join(out)


def message_text(msg: Message) -> str:
    return parts_to_text(msg.parts)


def media_base64(part: ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart) -> str:
    """The part's bytes as base64: inline `data`, or the `path` read now.
    A url/file_id-addressed part has no bytes here; the caller maps those."""
    if part.data is not None:
        return part.data
    if part.path is not None:
        return base64.b64encode(part.path.read_bytes()).decode("ascii")
    raise ValueError(f"{part.type} part has no inline data or path")


def media_data_uri(part: ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart) -> str:
    return f"data:{part.media_type};base64,{media_base64(part)}"


# ─── MAP-10: tool-result media policy ──────────────────────────────

# Which part kinds a `tool_result_media` value admits inside a result item.
_TOOL_RESULT_MEDIA_ADMITS: dict[str, frozenset[str]] = {
    "native": frozenset({"image", "document"}),
    "images": frozenset({"image"}),
    "reject": frozenset(),
}

# The door that carries the part, named in the refusal (MAP-10 rule 3).
_MEDIA_DOORS: dict[str, str] = {
    "image": "the OpenAI Responses, Anthropic Messages and Gemini dialects (and the xai/moonshotai/zai chat presets)",
    "document": "the OpenAI Responses, Anthropic Messages and Gemini dialects",
}


def check_tool_result_media(provider: str, part: ToolResultPart, policy: str, *, wire: str) -> None:
    """Raise before any wire when `part.content` carries a part kind the
    preset's `tool_result_media` policy does not admit (MAP-10 rules 1–3).
    `wire` names the field for the message ("a tool row", "function_call_output")."""
    from ..errors import UnsupportedFeatureError

    admits = _TOOL_RESULT_MEDIA_ADMITS[policy]
    for p in part.content:
        if p.type in MEDIA_KINDS and p.type not in admits:
            why = ("this server takes text-only tool results" if policy == "reject"
                   else f"this server carries images but not {p.type} parts in a tool result")
            raise UnsupportedFeatureError(
                f"{provider}: a {p.type} part in tool_result {part.id!r} cannot reach {wire} — {why} "
                f"(compat tool_result_media={policy!r}, measured: lm15-contract/research/tool-result-content/). "
                f"Carried natively by {_MEDIA_DOORS.get(p.type, 'no lm15 door yet')}; "
                "or render the part to text yourself before building the tool result (MAP-10)",
                provider=provider,
            )


def tool_result_error_text(part: ToolResultPart, text: str) -> str:
    """MAP-10 rule 5 on wires with no error flag: the text carries it."""
    return f"[error] {text}" if part.is_error else text


def media_bytes(part: ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart) -> bytes:
    return part.bytes


def extension_config(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def json_dumps(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def path_id(value: str, *, resource_name: bool = False) -> str:
    """A provider id placed in a URL path (docs/mapping-rules.md MAP-11).

    RFC 3986 percent-encoding over UTF-8: every byte outside the unreserved
    set becomes ``%XX``. ``resource_name`` keeps ``/`` literal for wires
    whose ids are resource names (Gemini ``files/abc``); a flat-id wire
    (OpenAI, Anthropic, xAI) encodes ``/`` too, since a literal slash would
    turn one operation into another on the same route table. The id is
    never decoded first: a pre-encoded id is double-encoded and answers 404
    (loud), where a raw reserved byte would misroute (silent).
    """
    return urllib.parse.quote(value, safe="/" if resource_name else "")


def build_url(url: str, params: dict[str, Any] | None = None) -> str:
    if not params:
        return url
    clean = {k: v for k, v in params.items() if v is not None}
    if not clean:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urllib.parse.urlencode(clean)}"


def make_json_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
    params: dict[str, Any] | None = None,
    payload: JsonPayload | None = None,
    body: bytes | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
) -> TransportRequest:
    hdrs = list(headers.items()) if isinstance(headers, dict) else list(headers or [])
    if payload is not None:
        body = json_dumps(payload)
        if not any(k.lower() == "content-type" for k, _ in hdrs):
            hdrs.append(("Content-Type", "application/json"))
    return TransportRequest(
        method=method,
        url=build_url(url, params),
        headers=hdrs,
        body=body or b"",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
    )


def model_infos_from_entries(
    entries: Any,
    *,
    provider: str,
    api_family: str,
    id_of: "callable",
) -> tuple[ModelInfo, ...]:
    """Map a provider's list-models entries to canonical ModelInfo.

    ``id`` is the usable Request.model string (``id_of`` extracts it); the
    verbatim wire entry is preserved under ``origin.provider_data`` (opaque,
    never cleaned).  Entries without a usable id are skipped.
    """
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = id_of(entry)
        if not isinstance(model_id, str) or not model_id:
            continue
        out.append(
            ModelInfo(
                id=model_id,
                provider=provider,
                api_family=api_family,
                origin=ModelOrigin(type="provider", provider_data=entry),
            )
        )
    return tuple(out)


def part_to_openai_input(part: Part, *, provider: str | None = None) -> dict[str, Any]:
    """One prompt part → one Responses input block. Every media source form
    maps (url, file_id, inline data, a path read now); a part with no slot
    RAISES (MAP-10) instead of becoming empty text."""
    from ..errors import UnsupportedFeatureError

    if isinstance(part, TextPart):
        return {"type": "input_text", "text": part.text}

    if isinstance(part, ImagePart):
        if part.file_id is not None:
            return {"type": "input_image", "file_id": part.file_id}
        payload = {"type": "input_image", "image_url": part.url if part.url is not None else media_data_uri(part)}
        if part.detail:
            payload["detail"] = part.detail
        return payload

    if isinstance(part, AudioPart):
        if part.url is not None:
            return {"type": "input_audio", "audio_url": part.url}
        if part.file_id is not None:
            return {"type": "input_audio", "file_id": part.file_id}
        media = (part.media_type or "audio/wav").split("/", 1)[-1]
        if media in {"mpeg", "mp3"}:
            media = "mp3"
        return {"type": "input_audio", "audio": media_base64(part), "format": media}

    if isinstance(part, (DocumentPart, BinaryPart)):
        if part.url is not None:
            return {"type": "input_file", "file_url": part.url}
        if part.file_id is not None:
            return {"type": "input_file", "file_id": part.file_id}
        # OpenAI requires a filename alongside inline file_data (live
        # 2026-06-11: 400 missing_required_parameter without one); derive
        # a deterministic name from the media-type subtype.
        ext = (part.media_type or "application/octet-stream").split("/", 1)[-1].split("+", 1)[0] or "bin"
        return {"type": "input_file", "filename": f"file.{ext}", "file_data": media_data_uri(part)}

    if isinstance(part, VideoPart):
        if part.url is not None:
            return {"type": "input_video", "video_url": part.url}
        if part.file_id is not None:
            return {"type": "input_video", "file_id": part.file_id}
        return {"type": "input_video", "video_data": media_data_uri(part)}

    if isinstance(part, (CitationPart, ThinkingPart)):
        return {"type": "input_text", "text": parts_to_text((part,))}

    head = f"{provider}: " if provider else ""
    raise UnsupportedFeatureError(
        f"{head}a {part.type} part has no input block on the Responses wire (MAP-10)", provider=provider,
    )


def tool_result_output_openai(provider: str, part: ToolResultPart, policy: str) -> str | list[dict[str, Any]]:
    """`function_call_output.output` (MAP-10): a string when the content is
    text-only, the documented array of input_text/input_image/input_file
    blocks otherwise; a media part the preset does not admit raises first.
    `is_error` rides as an `[error] ` prefix on the text (rule 5)."""
    check_tool_result_media(provider, part, policy, wire="function_call_output")
    if all(p.type not in MEDIA_KINDS for p in part.content):
        return tool_result_error_text(part, parts_to_text(part.content, provider=provider, where="function_call_output"))
    blocks: list[dict[str, Any]] = []
    for p in part.content:
        block = part_to_openai_input(p, provider=provider)
        blocks.append(block)
    if part.is_error:
        first = next((b for b in blocks if b.get("type") == "input_text"), None)
        if first is None:
            blocks.insert(0, {"type": "input_text", "text": "[error]"})
        else:
            first["text"] = "[error] " + first["text"]
    return blocks


def message_to_openai_input(msg: Message) -> dict[str, Any]:
    return {"role": msg.role, "content": [part_to_openai_input(p) for p in msg.parts]}


def anthropic_source(part: ImagePart | DocumentPart | BinaryPart) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "url", "url": part.url}
    if part.file_id is not None:
        return {"type": "file", "file_id": part.file_id}
    if part.data is not None:
        return {"type": "base64", "media_type": part.media_type, "data": part.data}
    if part.path is not None:
        data = base64.b64encode(part.path.read_bytes()).decode("ascii")
        return {"type": "base64", "media_type": part.media_type, "data": data}
    raise ValueError(f"{part.type} part has no usable source")


def openai_token_logprobs(entries: Any) -> tuple[TokenLogprob, ...]:
    """Map OpenAI-style logprob entries to canonical TokenLogprob tuples.

    Both OpenAI wire dialects share the entry shape (verified live
    2026-09-01): ``{token, logprob, bytes, top_logprobs: [{token, logprob,
    bytes}]}``.  Non-list input and malformed entries yield ().
    """
    if not isinstance(entries, list):
        return ()
    out: list[TokenLogprob] = []
    for entry in entries:
        if not isinstance(entry, dict) or "token" not in entry or "logprob" not in entry:
            continue
        top: list[TopLogprob] = []
        for alt in entry.get("top_logprobs") or []:
            if not isinstance(alt, dict) or "token" not in alt or "logprob" not in alt:
                continue
            alt_bytes = alt.get("bytes")
            top.append(
                TopLogprob(
                    token=str(alt["token"]),
                    logprob=float(alt["logprob"]),
                    bytes=tuple(alt_bytes) if isinstance(alt_bytes, list) else None,
                )
            )
        entry_bytes = entry.get("bytes")
        out.append(
            TokenLogprob(
                token=str(entry["token"]),
                logprob=float(entry["logprob"]),
                bytes=tuple(entry_bytes) if isinstance(entry_bytes, list) else None,
                top=tuple(top),
            )
        )
    return tuple(out)


def unnamed_tool_call_error(provider: str, path: str) -> "ProviderError":
    """MAP-9 on the complete path (changes/2026-09-07-complete-tool-call-no-guess.md):
    a tool call the provider sent without a name is not actionable, and lm15
    never guesses which tool the model meant."""
    from ..errors import ProviderError

    return ProviderError(
        f"{provider}: {path} is a tool call with no name; lm15 does not guess "
        "which tool the model meant (MAP-9)",
        provider=provider,
    )


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return {"partial_json": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def iso_utc(value: object) -> str | None:
    """Normalize a provider timestamp (unix epoch or ISO-8601 string) to
    canonical ``YYYY-MM-DDTHH:MM:SSZ``. Returns None when unparseable —
    the raw value stays available in provider_data."""
    import datetime

    try:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            dt = datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
        elif isinstance(value, str) and value:
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            # datetime.fromisoformat rejects >6 fractional digits (Gemini
            # emits nanoseconds); trim the fraction to microseconds.
            import re

            text = re.sub(r"\.(\d{6})\d+", r".\1", text)
            dt = datetime.datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            dt = dt.astimezone(datetime.timezone.utc)
        else:
            return None
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OverflowError, OSError):
        return None


def multipart_form_body(
    *,
    fields: list[tuple[str, str]] | None = None,
    files: list[tuple[str, str, str, bytes]] | None = None,
) -> tuple[str, bytes]:
    """Build a multipart/form-data body (OpenAI and Anthropic file uploads).

    ``files`` entries are ``(field_name, filename, content_type, data)``.
    Returns ``(content_type_header_value, body)``.
    """
    import uuid

    boundary = f"lm15-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields or []:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{value}\r\n".encode("utf-8"))
    for name, filename, content_type, data in files or []:
        safe_filename = filename.replace('"', "%22")
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{safe_filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def multipart_related_body(
    *,
    metadata: dict[str, Any],
    media_type: str,
    data: bytes,
) -> tuple[str, bytes]:
    """Build a multipart/related body (Gemini media upload: JSON metadata
    part followed by one media part).  Returns ``(content_type, body)``."""
    import uuid

    boundary = f"lm15-{uuid.uuid4().hex}"
    chunks: list[bytes] = [
        f"--{boundary}\r\n".encode("utf-8"),
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
        json_dumps(metadata),
        b"\r\n",
        f"--{boundary}\r\n".encode("utf-8"),
        f"Content-Type: {media_type}\r\n\r\n".encode("utf-8"),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return f"multipart/related; boundary={boundary}", b"".join(chunks)


# ─── MAP-7: the one effort→budget grading table ──────────────────────
#
# Budget-only model classes (Anthropic 4.5 and earlier, Gemini 2.5) have
# no effort level on the wire; the universal dial is expressed as a
# thinking-token budget through this single, stated table.  Receipted on
# both providers 2026-09-02 (research/reasoning).  Provider floors and
# ceilings are the provider's: Anthropic rejects < 1024, Gemini 2.5 Flash
# rejects > 24576 — loudly.
EFFORT_THINKING_BUDGETS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 24576,
    "max": 32768,
}
