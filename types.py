"""
lm15.types — Core vocabulary for foundation model interaction.

The fundamental unit is the Part: an atomic, typed block of content.
Parts compose into Messages (attributed to a speaker).
Messages compose into Requests (sent to a model).
Models produce Responses (containing a Message).

Streams reveal Responses incrementally through Deltas — typed fragments
of streamable response parts.  Not every Part is streamable; use
StreamablePart / NonStreamablePart to make that boundary explicit.

Design principles:

1. Parts, Deltas, and Events are proper discriminated unions.  Each
   variant is an independent frozen dataclass.  Fields that don't belong
   to a variant don't exist on it — accessing them raises AttributeError.
   Check .type or use isinstance() before accessing variant-specific fields.

2. One representation per concept.  A Delta is always a typed Delta
   object, never a dict.  Tool call arguments are called "input"
   everywhere — in memory, in deltas, in serialization.

3. Frozen + slotted dataclasses throughout.  Dataclass attributes are
   shallowly immutable after construction: fields cannot be rebound, while
   caller-provided JSON containers remain ordinary mutable Python containers.

4. Universal structure, provider-specific values.  The shape of a
   Request is universal.  Provider-specific configuration flows
   through Config.extensions — clearly separated from universal knobs.
   Provider-specific response metadata lives in Response.provider_data.

5. Runtime validation is deliberately narrow.  Constructors enforce the
   invariants that make objects meaningful (required identities, one media
   source, non-negative token counts, JSON-serializable extension fields)
   while LMs remain responsible for normalizing provider quirks before
   constructing these types.
"""

from __future__ import annotations

import base64 as _base64
import binascii as _binascii
import json as _json
import math as _math
import mimetypes as _mimetypes
import re as _re
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, get_args


# ─── Literal vocabularies ────────────────────────────────────────────

Role = Literal["user", "assistant", "tool", "developer"]

PartType = Literal[
    "text",
    "image",
    "audio",
    "video",
    "document",
    "binary",
    "tool_call",
    "tool_result",
    "thinking",
    "refusal",
    "citation",
]

ContinuationKind: TypeAlias = str

# FinishReason values are a separate namespace from PartType values even when
# a token such as "tool_call" appears in both.
FinishReason = Literal["stop", "length", "tool_call", "content_filter", "error"]

ReasoningEffort = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max"
]
ReasoningSummary = Literal["auto", "concise", "detailed"]

ErrorCode = Literal[
    "auth",
    "billing",
    "rate_limit",
    "invalid_request",
    "context_length",
    "timeout",
    "server",
    "unsupported_model",
    "unsupported_feature",
    "not_configured",
    "unknown_model",
    "ambiguous_model",
    "transport",
    "lock_timeout",
    "stream_assembly",
    "provider",
]
ERROR_CODES = frozenset(get_args(ErrorCode))

StreamEventType = Literal["start", "delta", "end", "error"]
BatchStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled", "expired"]
VideoStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
BatchOutcome = Literal["succeeded", "errored", "cancelled", "expired"]
FileReadiness = Literal["pending", "ready", "failed"]
AudioEncoding = Literal["pcm16", "opus", "mp3", "aac"]
ToolChoiceMode = Literal["auto", "required", "none"]
LiveClientEventType = Literal["turn", "audio", "image", "text", "tool_result", "interrupt", "end_audio"]
LiveServerEventType = Literal["audio", "text", "tool_call", "tool_call_delta", "interrupted", "turn_end", "usage", "error"]

ROLE_VALUES = frozenset(get_args(Role))
FINISH_REASONS = frozenset(get_args(FinishReason))
REASONING_EFFORTS = frozenset(get_args(ReasoningEffort))
REASONING_SUMMARIES = frozenset(get_args(ReasoningSummary))
BATCH_STATUSES = frozenset(get_args(BatchStatus))
BATCH_OUTCOMES = frozenset(get_args(BatchOutcome))
# Job statuses in which a batch will make no further progress.
BATCH_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})
VIDEO_STATUSES = frozenset(get_args(VideoStatus))
# Job statuses in which a video job will make no further progress.
VIDEO_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
FILE_READINESS_VALUES = frozenset(get_args(FileReadiness))
AUDIO_ENCODINGS = frozenset(get_args(AudioEncoding))
TOOL_CHOICE_MODES = frozenset(get_args(ToolChoiceMode))

_P = TypeVar("_P")
_MISSING: Any = object()

# JSON-compatible values used for model inputs, tool schemas, provider
# extensions, and provider metadata.  Keep these small and boring: the
# LM layer can map richer Python objects into this vocabulary before
# they enter the core model.
JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonArray: TypeAlias = list[JsonValue]
JsonObject: TypeAlias = dict[str, JsonValue]


def _is_json_value(value: Any) -> bool:
    """Return True if value is made only of strict JSON containers.

    Complete JSON validation is necessarily proportional to payload size, so
    keep it iterative and simple: no recursion, no copying, and no coercion of
    tuples or non-string dict keys into something different on the wire.
    """
    stack = [value]
    while stack:
        item = stack.pop()
        kind = type(item)
        if item is None or kind is bool or kind is int or kind is str:
            continue
        if kind is float:
            if not _math.isfinite(item):
                return False
            continue
        if kind is list:
            stack.extend(item)
            continue
        if kind is dict:
            for key, child in item.items():
                if not isinstance(key, str):
                    return False
                stack.append(child)
            continue
        # Keep compatibility with scalar subclasses without paying
        # ``isinstance`` costs for the common exact built-in types above.
        if isinstance(item, (int, str)):
            continue
        if isinstance(item, float):
            if not _math.isfinite(item):
                return False
            continue
        return False
    return True


def _check_json_object(value: Any, *, field_name: str, required: bool = False) -> None:
    """Validate a JSON object field without copying or wrapping it."""
    if value is None:
        if required:
            raise TypeError(f"{field_name} must be a JSON object")
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    if not _is_json_value(value):
        raise TypeError(f"{field_name} must contain only JSON-compatible values")


def _validate_json_field(obj: object, field_name: str, *, required: bool = False) -> None:
    """Validate a JSON object field on a dataclass."""
    _check_json_object(getattr(obj, field_name), field_name=field_name, required=required)


def _validate_extensions_field(obj: object) -> None:
    """Validate extensions, normalizing an empty mapping to None."""
    if getattr(obj, "extensions") == {}:
        object.__setattr__(obj, "extensions", None)
        return
    _validate_json_field(obj, "extensions")


@dataclass(frozen=True, slots=True, repr=False)
class ContinuationState:
    """Opaque provider-owned state needed to continue/replay a transcript.

    Continuation state is not visible model content.  It travels with the
    Message or Part it describes so provider adapters can reconstruct future
    provider requests without relying on detached response metadata.
    """

    provider: str
    kind: ContinuationKind
    data: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text(self.provider, field_name="ContinuationState.provider", allow_empty=False)
        _validate_text(self.kind, field_name="ContinuationState.kind", allow_empty=False)
        _validate_json_field(self, "data", required=True)

    def __repr__(self) -> str:
        return (
            "ContinuationState("
            f"provider={self.provider!r}, "
            f"kind={self.kind!r}, "
            f"data=<dict: {len(self.data)} keys>)"
        )


def _normalize_continuation(value: Any, *, field_name: str = "continuation") -> tuple[ContinuationState, ...]:
    if value is None:
        return ()
    if isinstance(value, ContinuationState):
        return (value,)
    if isinstance(value, str):
        raise TypeError(f"{field_name} must contain ContinuationState objects")
    try:
        states = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of ContinuationState objects") from exc
    if not all(isinstance(item, ContinuationState) for item in states):
        raise TypeError(f"{field_name} must contain ContinuationState objects")
    return states


def _validate_continuation_field(obj: object, field_name: str = "continuation") -> None:
    object.__setattr__(
        obj,
        field_name,
        _normalize_continuation(getattr(obj, field_name), field_name=field_name),
    )


def _continuation_repr(value: tuple[ContinuationState, ...]) -> str | None:
    if not value:
        return None
    return f"<{len(value)} continuation state{'s' if len(value) != 1 else ''}>"


def _continuation_by_kind(
    value: tuple[ContinuationState, ...], provider: str, kind: str
) -> JsonObject | None:
    for state in value:
        if state.provider == provider and state.kind == kind:
            return state.data
    return None


def continuation_data(
    value: Message | Part | tuple[ContinuationState, ...], provider: str, kind: str
) -> JsonObject | None:
    """Return provider-owned continuation data from a Message or Part.

    This is a convenience helper for provider adapters and advanced callers.
    It returns the opaque JSON object attached to the first matching
    ContinuationState, or None when no match exists.
    """
    if isinstance(value, tuple):
        states = value
    else:
        states = getattr(value, "continuation", ())
    return _continuation_by_kind(states, provider, kind)


def _coerce_int_field(obj: object, field_name: str) -> None:
    """Number rule (docs/serde-rules.md): int fields have ONE wire form.

    Same-valued float input is coerced (2.0 -> 2); non-integral floats are
    rejected; bool never coerces (``type is float`` excludes bool, and the
    int validators reject bool downstream).
    """
    value = getattr(obj, field_name)
    if type(value) is float:
        if not value.is_integer():
            raise TypeError(f"{field_name} must be an int")
        object.__setattr__(obj, field_name, int(value))


def _coerce_float_field(obj: object, field_name: str) -> None:
    """Number rule (docs/serde-rules.md): float fields have ONE wire form.

    Same-valued int input is coerced (1 -> 1.0); bool never coerces
    (``type is int`` excludes bool, leaving it for downstream rejection).
    """
    value = getattr(obj, field_name)
    if type(value) is int:
        object.__setattr__(obj, field_name, float(value))


def _validate_int(value: Any, *, field_name: str) -> None:
    """Reject non-int values, including bool (which subclasses int)."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")


def _validate_positive(value: int | None, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_non_negative(value: int | None, *, field_name: str) -> None:
    _validate_int(value, field_name=field_name)
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_part_index(part_index: int) -> None:
    _validate_int(part_index, field_name="part_index")
    if part_index < 0:
        raise ValueError("part_index must be >= 0")


def _validate_text(value: Any, *, field_name: str, allow_empty: bool = True) -> None:
    """Validate that value is a string (and optionally non-empty)."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not allow_empty and value == "":
        raise ValueError(f"{field_name} cannot be empty")


def _validate_optional_text(
    value: Any, *, field_name: str, allow_empty: bool = True
) -> None:
    """Validate that an optional value is either None or a string."""
    if value is not None:
        _validate_text(value, field_name=field_name, allow_empty=allow_empty)


def _validate_bool(value: Any, *, field_name: str) -> None:
    """Validate that value is exactly a bool."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")


def _validate_optional_bool(value: Any, *, field_name: str) -> None:
    """Validate that an optional value is either None or exactly a bool."""
    if value is not None:
        _validate_bool(value, field_name=field_name)


# ─── Media helpers ────────────────────────────────────────────────────


def _validate_media(
    part_type: str,
    data: str | None,
    url: str | None,
    file_id: str | None,
    path: Path | None,
) -> None:
    """Validate that exactly one non-empty media source is set."""
    count = (
        (data is not None)
        + (url is not None)
        + (file_id is not None)
        + (path is not None)
    )
    if count != 1:
        raise ValueError(f"{part_type} requires exactly one of data, url, file_id, or path")
    if path is not None:
        if not isinstance(path, Path):
            raise TypeError(f"{part_type} path must be a pathlib.Path")
        return
    name, value = (
        ("data", data)
        if data is not None
        else ("url", url)
        if url is not None
        else ("file_id", file_id)
    )
    if not isinstance(value, str):
        raise TypeError(f"{part_type} {name} must be a string")
    if value == "":
        raise ValueError(f"{part_type} {name} cannot be empty")


def _validate_media_delta_addresses(
    part_type: str, data: str | None, url: str | None, file_id: str | None
) -> None:
    """Validate media-delta address shape without requiring a final address."""
    if (data is not None) + (url is not None) + (file_id is not None) > 1:
        raise ValueError(f"{part_type} can include at most one of data, url, or file_id")


_BASE64_RE = _re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def _base64_payload(part_type: str, data: str | None) -> str:
    """Return the base64 payload from a raw base64 string or data URI."""
    if data is None:
        raise ValueError(
            f"{part_type} has no inline data; fetch url/file_id-addressed media before decoding"
        )
    if not isinstance(data, str):
        raise TypeError(f"{part_type}.data must be a base64 string")
    if data == "":
        raise ValueError(f"{part_type}.data cannot be empty")
    if data.startswith("data:") and ";base64," in data:
        data = data.split(";base64,", 1)[1]
    stripped = data.strip()
    if (
        stripped != data
        or " " in stripped
        or "\n" in stripped
        or "\r" in stripped
        or "\t" in stripped
        or "\v" in stripped
        or "\f" in stripped
    ):
        return "".join(stripped.split())
    return data


def _validate_base64_data(part_type: str, data: str | None) -> None:
    """Validate base64 shape without eagerly decoding large payloads."""
    payload = _base64_payload(part_type, data)
    if len(payload) % 4 != 0 or not _BASE64_RE.fullmatch(payload):
        raise ValueError(f"{part_type}.data must be a valid base64 string")


def _decode_data(part_type: str, data: str | None) -> bytes:
    """Decode base64 data to bytes."""
    payload = _base64_payload(part_type, data)
    try:
        return _base64.b64decode(payload, validate=True)
    except (_binascii.Error, ValueError) as e:
        raise ValueError(f"{part_type}.data must be a valid base64 string") from e


def _base64_summary(data: str | None) -> str | None:
    """Return a short repr-safe summary for base64 data."""
    if data is None:
        return None
    payload = _base64_payload("media", data)
    return f"<base64: {len(payload)} chars>"


def _base64_chunk_summary(data: str | None) -> str | None:
    """Return a repr-safe summary for a possibly partial base64 chunk."""
    if data is None:
        return None
    if data.startswith("data:") and ";base64," in data:
        data = data.split(";base64,", 1)[1]
    stripped = data.strip()
    if (
        stripped != data
        or " " in stripped
        or "\n" in stripped
        or "\r" in stripped
        or "\t" in stripped
        or "\v" in stripped
        or "\f" in stripped
    ):
        stripped = "".join(stripped.split())
    return f"<base64: {len(stripped)} chars>"


def _bytes_summary(data: bytes | bytearray | None) -> str | None:
    """Return a short repr-safe summary for raw bytes."""
    if data is None:
        return None
    return f"<bytes: {len(data)} bytes>"


@dataclass(frozen=True, slots=True, repr=False)
class _MediaMixin:
    """Shared fields, validation, repr, and byte access for media parts."""

    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    path: Path | None = None
    continuation: tuple[ContinuationState, ...] = ()

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, Path):
            if str(self.path) == "":
                raise ValueError(f"{self.__class__.__name__} path cannot be empty")
            object.__setattr__(self, "path", Path(self.path))
        if not isinstance(self.media_type, str) or self.media_type == "":
            raise ValueError(f"{self.__class__.__name__} requires media_type")
        _validate_media(self.__class__.__name__, self.data, self.url, self.file_id, self.path)
        if self.data is not None:
            _validate_base64_data(self.__class__.__name__, self.data)
        _validate_continuation_field(self)

    def __repr__(self) -> str:
        fields = [
            ("media_type", self.media_type),
            ("data", _base64_summary(self.data)),
            ("url", self.url),
            ("file_id", self.file_id),
            ("path", self.path),
        ]
        detail = getattr(self, "detail", None)
        if detail is not None:
            fields.append(("detail", detail))
        continuation = _continuation_repr(self.continuation)
        if continuation is not None:
            fields.append(("continuation", continuation))
        args = ", ".join(f"{name}={value!r}" for name, value in fields)
        return f"{self.__class__.__name__}({args})"

    @property
    def bytes(self) -> bytes:
        if self.data is not None:
            return _decode_data(self.__class__.__name__, self.data)
        if self.path is not None:
            return self.path.read_bytes()
        raise ValueError(
            f"{self.__class__.__name__} has no inline data or path; "
            "fetch url/file_id-addressed media before decoding"
        )


# ─── Parts ───────────────────────────────────────────────────────────
#
# The atoms of content.  A discriminated union — check .type or use
# isinstance(), then access the fields that belong to that variant.


@dataclass(frozen=True, slots=True)
class TextPart:
    """A block of text content."""

    text: str
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="TextPart.text")
        _validate_continuation_field(self)


@dataclass(frozen=True, slots=True, repr=False)
class ImagePart(_MediaMixin):
    """An image, addressed by exactly one of data/url/file_id/path."""

    media_type: str = "image/png"
    detail: Literal["low", "high", "auto"] | None = None
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        _MediaMixin.__post_init__(self)
        _validate_optional_text(self.detail, field_name="ImagePart.detail", allow_empty=False)
        if self.detail is not None and self.detail not in {"low", "high", "auto"}:
            raise ValueError(f"unsupported ImagePart.detail: {self.detail}")


@dataclass(frozen=True, slots=True, repr=False)
class AudioPart(_MediaMixin):
    """Audio content, addressed by exactly one of data/url/file_id/path."""

    media_type: str = "audio/wav"
    type: Literal["audio"] = field(default="audio", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class VideoPart(_MediaMixin):
    """Video content, addressed by exactly one of data/url/file_id/path."""

    media_type: str = "video/mp4"
    type: Literal["video"] = field(default="video", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class DocumentPart(_MediaMixin):
    """A document (PDF, etc.), addressed by exactly one of data/url/file_id/path."""

    media_type: str = "application/pdf"
    type: Literal["document"] = field(default="document", init=False)


@dataclass(frozen=True, slots=True, repr=False)
class BinaryPart(_MediaMixin):
    """Arbitrary binary content, addressed by exactly one media source."""

    media_type: str = "application/octet-stream"
    type: Literal["binary"] = field(default="binary", init=False)


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """The model requests an external computation."""

    id: str
    name: str
    input: JsonObject
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="ToolCallPart.id", allow_empty=False)
        _validate_text(self.name, field_name="ToolCallPart.name", allow_empty=False)
        _validate_json_field(self, "input", required=True)
        _validate_continuation_field(self)


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The result of an external computation, sent back to the model."""

    id: str
    content: tuple[ToolResultContentPart, ...]
    name: str | None = None
    is_error: bool = False
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="ToolResultPart.id", allow_empty=False)
        _validate_optional_text(self.name, field_name="ToolResultPart.name", allow_empty=False)
        object.__setattr__(self, "content", tuple(self.content))
        if not self.content:
            raise ValueError("ToolResultPart requires content")
        if not all(_is_part(p) for p in self.content):
            raise TypeError("ToolResultPart.content must contain Part objects")
        _validate_bool(self.is_error, field_name="ToolResultPart.is_error")
        # Tool results may not contain protocol parts: tool calls, nested
        # tool results, model reasoning traces, or refusals.  Only the
        # presentational variants from ToolResultContentPart are allowed.
        if any(isinstance(p, _TOOL_RESULT_FORBIDDEN_PARTS) for p in self.content):
            raise TypeError(
                "ToolResultPart.content cannot contain tool calls, nested tool "
                "results, thinking parts, or refusals"
            )
        _validate_continuation_field(self)


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """Model reasoning trace.

    Hidden thinking (Anthropic ``redacted_thinking``, an OpenAI reasoning
    item with no summary) is a ThinkingPart with empty ``text`` and its
    replay state in ``continuation``.  There is no flag and no placeholder
    text (MAP-7 rule 11, ratified 2026-09-06).
    """

    text: str
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["thinking"] = field(default="thinking", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="ThinkingPart.text")
        _validate_continuation_field(self)


@dataclass(frozen=True, slots=True)
class RefusalPart:
    """Model explicitly refused to respond.

    Refusals require non-empty text because they are final semantic content;
    empty ``TextPart``/``ThinkingPart`` values remain allowed for streaming
    reassembly and provider redaction edge cases.
    """

    text: str
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["refusal"] = field(default="refusal", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="RefusalPart.text", allow_empty=False)
        _validate_continuation_field(self)


@dataclass(frozen=True, slots=True)
class CitationPart:
    """A reference to source material."""

    url: str | None = None
    title: str | None = None
    text: str | None = None
    continuation: tuple[ContinuationState, ...] = ()
    type: Literal["citation"] = field(default="citation", init=False)

    def __post_init__(self) -> None:
        _validate_optional_text(self.url, field_name="CitationPart.url", allow_empty=False)
        _validate_optional_text(self.title, field_name="CitationPart.title", allow_empty=False)
        _validate_optional_text(self.text, field_name="CitationPart.text", allow_empty=False)
        if self.url is None and self.title is None and self.text is None:
            raise ValueError("CitationPart requires at least one of url, title, or text")
        _validate_continuation_field(self)


_TOOL_RESULT_FORBIDDEN_PARTS: tuple[type, ...] = (
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
)


# The union type.  This IS the vocabulary of content.
Part: TypeAlias = (
    TextPart
    | ImagePart
    | AudioPart
    | VideoPart
    | DocumentPart
    | BinaryPart
    | ToolCallPart
    | ToolResultPart
    | ThinkingPart
    | RefusalPart
    | CitationPart
)

# Runtime dispatch table, derived from the union so adding a Part variant has
# one source of truth: the variant class plus the Part union.
def _variant_type(cls: type) -> str:
    return cls.__dataclass_fields__["type"].default  # type: ignore[attr-defined]


PART_CLASSES: tuple[type, ...] = get_args(Part)
PART_TYPES: dict[str, type] = {_variant_type(cls): cls for cls in PART_CLASSES}


def _is_part(value: object) -> bool:
    """Return True if value is one of lm15's concrete Part variants."""
    return isinstance(value, PART_CLASSES)


MediaPart: TypeAlias = ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart
MEDIA_TYPES: tuple[type, ...] = get_args(MediaPart)

# Shared endpoint metadata/content aliases
Extensions: TypeAlias = JsonObject
ProviderData: TypeAlias = JsonObject

# Parts allowed in tool result content: presentational variants only.
# (ToolCallPart, ToolResultPart, ThinkingPart, and RefusalPart are excluded
# both at the type level and in ``ToolResultPart.__post_init__``.)
ToolResultContentPart: TypeAlias = (
    TextPart | ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart | CitationPart
)
ToolResultContent: TypeAlias = str | ToolResultContentPart | Sequence[str | ToolResultContentPart]

# Parts allowed in prompts (user/developer messages and system content).
# Excludes model/tool protocol parts which are produced by the model or
# tool runtime, never authored by the caller.
PromptPart: TypeAlias = (
    TextPart | ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart
)
PromptContent: TypeAlias = str | PromptPart | Sequence[str | PromptPart]
SystemContent: TypeAlias = PromptContent

# Parts allowed in assistant messages.  This intentionally includes
# model-emitted protocol/artifact parts while still excluding ToolResultPart,
# which belongs only in tool messages.
AssistantPart: TypeAlias = (
    TextPart
    | ImagePart
    | AudioPart
    | VideoPart
    | DocumentPart
    | BinaryPart
    | ToolCallPart
    | ThinkingPart
    | RefusalPart
    | CitationPart
)
AssistantContent: TypeAlias = str | AssistantPart | Sequence[str | AssistantPart]

# Broad content alias used internally by the normalizer; role-specific
# constructors expose narrower aliases above.
PartInput: TypeAlias = str | Part | Sequence[str | Part]

# Parts forbidden in prompts (user/developer messages and system content).
# Defined once and reused by every prompt-side validator.
_PROMPT_FORBIDDEN_PARTS: tuple[type, ...] = (
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    RefusalPart,
    CitationPart,
)


# ─── Part constructors ───────────────────────────────────────────────
#
# Factory functions for the common construction patterns.  These live
# at module level — there's no base class to hang them on.


def text(content: str, *, continuation: Sequence[ContinuationState] | ContinuationState | None = None) -> TextPart:
    """Create a text part."""
    return TextPart(text=content, continuation=_normalize_continuation(continuation))


def thinking(
    content: str,
    *,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> ThinkingPart:
    return ThinkingPart(text=content, continuation=_normalize_continuation(continuation))


def refusal(content: str, *, continuation: Sequence[ContinuationState] | ContinuationState | None = None) -> RefusalPart:
    return RefusalPart(text=content, continuation=_normalize_continuation(continuation))


def citation(
    *,
    url: str | None = None,
    title: str | None = None,
    text: str | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> CitationPart:
    return CitationPart(url=url, title=title, text=text, continuation=_normalize_continuation(continuation))


def _encode_data(data: bytes | str) -> str:
    """Ensure data is a base64 string."""
    if isinstance(data, bytes):
        return _base64.b64encode(data).decode("ascii")
    return data


def _prepare_media_factory_input(
    part_type: str,
    *,
    url: str | None,
    data: bytes | str | None,
    file_id: str | None,
    path: str | PathLike[str] | None,
    media_type: str | None,
    default_media_type: str,
) -> tuple[str | None, Path | None, str]:
    count = (
        (data is not None)
        + (url is not None)
        + (file_id is not None)
        + (path is not None)
    )
    if count != 1:
        raise ValueError(
            f"{part_type} requires exactly one of data, url, file_id, or path"
        )
    media_path: Path | None = None
    if path is not None:
        if str(path) == "":
            raise ValueError(f"{part_type} path cannot be empty")
        media_path = Path(path)
        media_type = media_type or _mimetypes.guess_type(str(media_path))[0]
    encoded = _encode_data(data) if data is not None else None
    return encoded, media_path, media_type or default_media_type


def image(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    detail: Literal["low", "high", "auto"] | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> ImagePart:
    encoded_data, media_path, resolved_media_type = _prepare_media_factory_input(
        "ImagePart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="image/png",
    )
    return ImagePart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        path=media_path,
        detail=detail,
        continuation=_normalize_continuation(continuation),
    )


def audio(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> AudioPart:
    encoded_data, media_path, resolved_media_type = _prepare_media_factory_input(
        "AudioPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="audio/wav",
    )
    return AudioPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        path=media_path,
        continuation=_normalize_continuation(continuation),
    )


def video(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> VideoPart:
    encoded_data, media_path, resolved_media_type = _prepare_media_factory_input(
        "VideoPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="video/mp4",
    )
    return VideoPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        path=media_path,
        continuation=_normalize_continuation(continuation),
    )


def document(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> DocumentPart:
    encoded_data, media_path, resolved_media_type = _prepare_media_factory_input(
        "DocumentPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="application/pdf",
    )
    return DocumentPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        path=media_path,
        continuation=_normalize_continuation(continuation),
    )


def binary(
    *,
    url: str | None = None,
    data: bytes | str | None = None,
    path: str | PathLike[str] | None = None,
    file_id: str | None = None,
    media_type: str | None = None,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> BinaryPart:
    encoded_data, media_path, resolved_media_type = _prepare_media_factory_input(
        "BinaryPart",
        url=url,
        data=data,
        file_id=file_id,
        path=path,
        media_type=media_type,
        default_media_type="application/octet-stream",
    )
    return BinaryPart(
        media_type=resolved_media_type,
        data=encoded_data,
        url=url,
        file_id=file_id,
        path=media_path,
        continuation=_normalize_continuation(continuation),
    )


def tool_call(
    id: str,
    name: str,
    input: JsonObject,
    *,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> ToolCallPart:
    return ToolCallPart(id=id, name=name, input=input, continuation=_normalize_continuation(continuation))


def tool_result(
    id: str,
    content: ToolResultContent,
    *,
    name: str | None = None,
    is_error: bool = False,
    continuation: Sequence[ContinuationState] | ContinuationState | None = None,
) -> ToolResultPart:
    """Create a tool result part.

    content can be a sequence of parts, a single part, or a string
    (which becomes a TextPart).  Raw bytes should be wrapped in a media part;
    use BinaryPart for arbitrary bytes that are not image/audio/video/document.
    """
    parts = _normalize_parts(content)  # type: ignore[arg-type]
    return ToolResultPart(
        id=id,
        content=parts,
        name=name,
        is_error=is_error,
        continuation=_normalize_continuation(continuation),
    )


# ─── Messages ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Message:
    """A contribution to a conversation, attributed to a speaker.

    A message is a sequence of typed Parts.  Roles:

    - ``user``      — end-user input
    - ``assistant``  — model output
    - ``tool``      — tool execution results
    - ``developer`` — high-authority instructions from the application
                      developer.  Can appear mid-conversation to inject
                      new instructions without invalidating the KV-cache
                      prefix.  On OpenAI this maps to the native
                      ``developer`` role; on other providers the LM
                      converts it to a user message with a clear prefix.
    """

    role: Role
    parts: tuple[Part, ...]
    continuation: tuple[ContinuationState, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLE_VALUES:
            raise ValueError(f"unsupported role: {self.role}")
        if isinstance(self.parts, str):
            raise TypeError("Message.parts must be Part objects; use Message.user('text') for strings")
        parts = (self.parts,) if _is_part(self.parts) else tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        if not self.parts:
            raise ValueError("Message requires at least one part")
        if not all(_is_part(p) for p in self.parts):
            raise TypeError("Message.parts must contain Part objects")
        _validate_continuation_field(self)
        _validate_message_parts(self.role, self.parts)

    @staticmethod
    def user(content: PromptContent) -> "Message":
        return Message(role="user", parts=_normalize_parts(content))

    @staticmethod
    def assistant(content: AssistantContent) -> "Message":
        return Message(role="assistant", parts=_normalize_parts(content))

    @staticmethod
    def developer(content: PromptContent) -> "Message":
        """Create a developer message.

        Developer messages carry instructions with higher authority than
        user messages (equivalent to OpenAI's ``developer`` role).  They
        can appear anywhere in the conversation — including mid-conversation
        — which is useful for injecting new instructions without
        invalidating the KV-cache prefix built from earlier turns.

        On providers that don't natively support a developer role
        (Anthropic, Gemini), the LM converts these to user messages
        with a clear ``[developer]`` prefix so the model still sees the
        instruction boundary.
        """
        return Message(role="developer", parts=_normalize_parts(content))

    @staticmethod
    def tool(
        results: "str | ToolResultPart | Sequence[ToolResultPart] | dict[str, ToolResultContent]",
        output: "ToolResultContent | None" = None,
        *,
        is_error: bool = False,
    ) -> "Message":
        """Create a tool message.  Three spellings:

        - ``Message.tool(call_id, output, is_error=False)`` — one result.
        - ``Message.tool({call_id: output, ...})`` — answer several calls
          at once (cannot express is_error; use the other forms).
        - ``Message.tool(part_or_parts)`` — explicit ToolResultPart(s),
          e.g. from :func:`lm15.tool_result`.
        """
        if isinstance(results, str):
            if output is None:
                raise TypeError(
                    "Message.tool(call_id) is missing the output. Accepted "
                    "spellings: Message.tool(call_id, output, is_error=...), "
                    "Message.tool({call_id: output, ...}), or "
                    "Message.tool(ToolResultPart(...))."
                )
            return Message(
                role="tool",
                parts=(tool_result(results, output, is_error=is_error),),
            )
        if output is not None:
            raise TypeError(
                "Message.tool() takes an output only with a call-id string: "
                "Message.tool(call_id, output, is_error=...)."
            )
        if is_error:
            raise TypeError(
                "Message.tool({...}, is_error=True) is ambiguous — the dict "
                "form cannot say which result errored. Use "
                "Message.tool(call_id, output, is_error=True) or "
                "lm15.tool_result(call_id, output, is_error=True)."
            )
        if isinstance(results, dict):
            parts: list[Part] = []
            for call_id, value in results.items():
                parts.append(tool_result(call_id, value))
            return Message(role="tool", parts=tuple(parts))
        parts = (results,) if isinstance(results, ToolResultPart) else tuple(results)
        if not all(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError(
                "Message.tool() requires ToolResultPart objects. Accepted "
                "spellings: Message.tool(call_id, output, is_error=...), "
                "Message.tool({call_id: output, ...}), or "
                "Message.tool(ToolResultPart(...))."
            )
        return Message(role="tool", parts=parts)

    def parts_of(self, cls: type[_P]) -> list[_P]:
        """Return all parts that are instances of ``cls``."""
        return [p for p in self.parts if isinstance(p, cls)]

    def first(self, cls: type[_P]) -> _P | None:
        """Return the first part that is an instance of ``cls``, if any."""
        return next((p for p in self.parts if isinstance(p, cls)), None)

    @property
    def text(self) -> str | None:
        """Text only when the message contains text and nothing else."""
        if not all(isinstance(p, TextPart) for p in self.parts):
            return None
        return "\n".join(p.text for p in self.parts)


def _normalize_parts(content: PartInput) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (TextPart(text=content),)
    if _is_part(content):
        return (content,)
    if isinstance(content, Sequence):
        raw_parts = tuple(content)
        if not raw_parts:
            raise ValueError("content sequence cannot be empty")
        parts: list[Part] = []
        for part in raw_parts:
            if isinstance(part, str):
                parts.append(TextPart(text=part))
            elif _is_part(part):
                parts.append(part)
            else:
                raise TypeError("content sequence must contain strings or Part objects")
        return tuple(parts)
    raise TypeError("content must be a string, Part, or sequence of Parts")


def _validate_message_parts(role: Role, parts: tuple[Part, ...]) -> None:
    if role == "tool":
        if not all(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError("tool messages may only contain ToolResultPart objects")
        return
    if role == "assistant":
        if any(isinstance(p, ToolResultPart) for p in parts):
            raise TypeError(
                "assistant messages cannot contain ToolResultPart objects"
            )
        return
    # User and developer messages carry prompts/instructions, not model/tool
    # protocol parts and not model-emitted artifacts like citations.
    if any(isinstance(p, _PROMPT_FORBIDDEN_PARTS) for p in parts):
        raise TypeError(f"{role} messages cannot contain model/tool protocol parts")


def _normalize_system(system: SystemContent | None) -> str | tuple[PromptPart, ...] | None:
    if system is None:
        return None
    if isinstance(system, str):
        if system == "":
            raise ValueError("system cannot be empty")
        return system
    parts = _normalize_parts(system)
    if any(isinstance(p, _PROMPT_FORBIDDEN_PARTS) for p in parts):
        raise TypeError("system parts cannot contain model/tool protocol parts")
    return parts


# ─── Deltas ──────────────────────────────────────────────────────────
#
# A Delta is a typed fragment of a Part being assembled during
# streaming.  Like Part, Delta is a discriminated union: fields that
# don't belong to a variant don't exist on it.

DeltaType = Literal["text", "thinking", "audio", "image", "tool_call", "citation", "continuation"]


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A text fragment arriving during streaming.

    ``logprobs`` carries the token logprobs for exactly the tokens in this
    fragment, when the request asked for them (``Config.logprobs``) and the
    provider streams them per chunk (verified live for OpenAI Responses,
    2026-09-01).  Materialization concatenates fragment logprobs in arrival
    order into ``Response.logprobs``.
    """

    text: str
    part_index: int = 0
    logprobs: tuple[TokenLogprob, ...] = ()
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_text(self.text, field_name="TextDelta.text")
        _validate_logprobs_field(self, "logprobs", allow_none=False)


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """A reasoning/thinking fragment arriving during streaming."""

    text: str
    part_index: int = 0
    type: Literal["thinking"] = field(default="thinking", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_text(self.text, field_name="ThinkingDelta.text")


@dataclass(frozen=True, slots=True, repr=False)
class AudioDelta:
    """An audio fragment arriving during streaming.

    Media deltas are partial stream chunks, not final media parts.  They may
    carry unaligned base64 data, a URL/file id update, or metadata only; final
    media validation happens when the stream is assembled into an AudioPart.
    """

    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    part_index: int = 0
    media_type: str | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_optional_text(self.data, field_name="AudioDelta.data")
        _validate_optional_text(self.url, field_name="AudioDelta.url")
        _validate_optional_text(self.file_id, field_name="AudioDelta.file_id")
        _validate_media_delta_addresses("AudioDelta", self.data, self.url, self.file_id)
        _validate_optional_text(self.media_type, field_name="AudioDelta.media_type", allow_empty=False)

    def __repr__(self) -> str:
        return (
            "AudioDelta("
            f"data={_base64_chunk_summary(self.data)!r}, "
            f"url={self.url!r}, "
            f"file_id={self.file_id!r}, "
            f"part_index={self.part_index!r}, "
            f"media_type={self.media_type!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ImageDelta:
    """An image fragment, addressed by exactly one of data/url/file_id."""

    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    part_index: int = 0
    media_type: str | None = None
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_optional_text(self.data, field_name="ImageDelta.data")
        _validate_optional_text(self.url, field_name="ImageDelta.url")
        _validate_optional_text(self.file_id, field_name="ImageDelta.file_id")
        _validate_media_delta_addresses("ImageDelta", self.data, self.url, self.file_id)
        _validate_optional_text(self.media_type, field_name="ImageDelta.media_type", allow_empty=False)

    def __repr__(self) -> str:
        return (
            "ImageDelta("
            f"data={_base64_chunk_summary(self.data)!r}, "
            f"url={self.url!r}, "
            f"file_id={self.file_id!r}, "
            f"part_index={self.part_index!r}, "
            f"media_type={self.media_type!r})"
        )


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """A tool-call input fragment, optionally carrying call identity."""

    input: str
    part_index: int = 0
    id: str | None = None
    name: str | None = None
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_text(self.input, field_name="ToolCallDelta.input")
        _validate_optional_text(self.id, field_name="ToolCallDelta.id", allow_empty=False)
        _validate_optional_text(self.name, field_name="ToolCallDelta.name", allow_empty=False)


@dataclass(frozen=True, slots=True)
class CitationDelta:
    """A citation fragment arriving during streaming."""

    text: str | None = None
    url: str | None = None
    title: str | None = None
    part_index: int = 0
    type: Literal["citation"] = field(default="citation", init=False)

    def __post_init__(self) -> None:
        _coerce_int_field(self, "part_index")
        _validate_part_index(self.part_index)
        _validate_optional_text(self.text, field_name="CitationDelta.text")
        _validate_optional_text(self.url, field_name="CitationDelta.url")
        _validate_optional_text(self.title, field_name="CitationDelta.title")
        if self.text is None and self.url is None and self.title is None:
            raise ValueError("CitationDelta requires at least one of text, url, or title")


@dataclass(frozen=True, slots=True)
class ContinuationDelta:
    """Opaque provider continuation state arriving during streaming.

    part_index=None attaches the state to the assistant message; otherwise it
    attaches to the completed part with that index.
    """

    provider: str
    kind: ContinuationKind
    data: JsonObject = field(default_factory=dict)
    part_index: int | None = None
    type: Literal["continuation"] = field(default="continuation", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.provider, field_name="ContinuationDelta.provider", allow_empty=False)
        _validate_text(self.kind, field_name="ContinuationDelta.kind", allow_empty=False)
        _validate_json_field(self, "data", required=True)
        if self.part_index is not None:
            _coerce_int_field(self, "part_index")
            _validate_part_index(self.part_index)

    def to_state(self) -> ContinuationState:
        return ContinuationState(provider=self.provider, kind=self.kind, data=self.data)


Delta: TypeAlias = (
    TextDelta
    | ThinkingDelta
    | AudioDelta
    | ImageDelta
    | ToolCallDelta
    | CitationDelta
    | ContinuationDelta
)

DELTA_CLASSES: tuple[type, ...] = get_args(Delta)
DELTA_TYPES: dict[str, type] = {_variant_type(cls): cls for cls in DELTA_CLASSES}

# Streaming boundary: content Delta variants assemble into Parts.  Some Delta
# variants (currently ContinuationDelta) carry stream metadata that attaches to
# a Message or Part rather than materializing as content.
StreamablePart: TypeAlias = TextPart | ThinkingPart | ImagePart | AudioPart | ToolCallPart | CitationPart
NonStreamablePart: TypeAlias = VideoPart | DocumentPart | BinaryPart | ToolResultPart | RefusalPart

_STREAMABLE_PART_CLASSES: tuple[type, ...] = get_args(StreamablePart)
_NON_STREAMABLE_PART_CLASSES: tuple[type, ...] = get_args(NonStreamablePart)
_STATE_DELTA_TYPES = frozenset({"continuation"})


def _check_streamable_partition() -> None:
    streamable = {_variant_type(cls) for cls in _STREAMABLE_PART_CLASSES}
    non_streamable = {_variant_type(cls) for cls in _NON_STREAMABLE_PART_CLASSES}
    delta_types = set(DELTA_TYPES) - set(_STATE_DELTA_TYPES)
    part_types = set(PART_TYPES)

    overlap = streamable & non_streamable
    if overlap:
        raise RuntimeError(
            f"StreamablePart and NonStreamablePart overlap on: {sorted(overlap)}"
        )
    union = streamable | non_streamable
    if union != part_types:
        missing = part_types - union
        extra = union - part_types
        raise RuntimeError(
            "StreamablePart ∪ NonStreamablePart must equal Part. "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if delta_types != streamable:
        raise RuntimeError(
            "Delta variants must match StreamablePart. "
            f"delta_types={sorted(delta_types)} streamable={sorted(streamable)}"
        )


_check_streamable_partition()


# ─── Stream Events ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """Structured error information.  A dataclass, not a dict."""

    code: ErrorCode
    message: str
    provider_code: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unsupported error code: {self.code}")
        _validate_text(self.message, field_name="ErrorDetail.message")
        _validate_optional_text(self.provider_code, field_name="ErrorDetail.provider_code", allow_empty=False)


@dataclass(frozen=True, slots=True)
class StreamStartEvent:
    """The response stream has started."""

    id: str | None = None
    model: str | None = None
    type: Literal["start"] = field(default="start", init=False)

    def __post_init__(self) -> None:
        _validate_optional_text(self.id, field_name="StreamStartEvent.id", allow_empty=False)
        _validate_optional_text(self.model, field_name="StreamStartEvent.model", allow_empty=False)


@dataclass(frozen=True, slots=True)
class StreamDeltaEvent:
    """A typed content delta arrived."""

    delta: Delta
    type: Literal["delta"] = field(default="delta", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.delta, DELTA_CLASSES):
            raise TypeError("StreamDeltaEvent.delta must be a Delta")


@dataclass(frozen=True, slots=True, repr=False)
class StreamEndEvent:
    """The response stream completed."""

    finish_reason: FinishReason | None = None
    usage: "Usage | None" = None
    provider_data: ProviderData | None = None
    type: Literal["end"] = field(default="end", init=False)

    def __post_init__(self) -> None:
        if self.finish_reason is not None and self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"unsupported finish reason: {self.finish_reason}")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("StreamEndEvent.usage must be a Usage")
        _validate_json_field(self, "provider_data")

    def __repr__(self) -> str:
        fields = []
        if self.finish_reason is not None:
            fields.append(("finish_reason", repr(self.finish_reason)))
        if self.usage is not None:
            fields.append(("usage", repr(self.usage)))
        if self.provider_data is not None:
            if isinstance(self.provider_data, dict):
                fields.append(("provider_data", f"<dict: {len(self.provider_data)} keys>"))
            else:
                fields.append(("provider_data", "<present>"))
        fields.append(("type", repr(self.type)))

        body = ",\n".join(f"    {name}={value}" for name, value in fields)
        return f"StreamEndEvent(\n{body},\n)"


@dataclass(frozen=True, slots=True)
class StreamErrorEvent:
    """The stream failed."""

    error: ErrorDetail
    type: Literal["error"] = field(default="error", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.error, ErrorDetail):
            raise TypeError("StreamErrorEvent.error must be an ErrorDetail")


StreamEvent: TypeAlias = StreamStartEvent | StreamDeltaEvent | StreamEndEvent | StreamErrorEvent
STREAM_EVENT_CLASSES: tuple[type, ...] = get_args(StreamEvent)


# ─── Tools ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Serializable function tool specification sent to the model."""

    name: str
    description: str | None = None
    parameters: JsonObject = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    type: Literal["function"] = field(default="function", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.name, field_name="FunctionTool.name", allow_empty=False)
        _validate_optional_text(self.description, field_name="FunctionTool.description")
        _validate_json_field(self, "parameters", required=True)

@dataclass(frozen=True, slots=True)
class BuiltinTool:
    """A provider-native tool (web search, code execution, etc.)."""

    name: str
    config: JsonObject | None = None
    type: Literal["builtin"] = field(default="builtin", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.name, field_name="BuiltinTool.name", allow_empty=False)
        _validate_json_field(self, "config")


Tool: TypeAlias = FunctionTool | BuiltinTool


# ─── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Reasoning:
    """How much hidden thinking the model does before it answers (MAP-7).

    ``effort`` is the one dial and is required.  Vocabulary: ``off``,
    ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``, ``max``.  Every
    provider has this dial with these words; each model accepts a subset
    and rejects the rest loudly.  Adapters send the word verbatim where
    the provider has levels; budget-only model classes (Anthropic 4.5 and
    earlier, Gemini 2.5) express it through one documented grading table.
    Words with no native level on a provider RAISE rather than downgrade.

    There is no "adaptive" value: leaving ``config.reasoning`` unset is
    how you let the model decide, on every provider.  ``effort="off"``
    reaches the wire as the provider's disable or fails loudly where the
    provider cannot disable (MAP-5).

    ``thinking_budget`` is a token cap on providers that count thinking
    separately (Anthropic manual class, Gemini); it RAISES where the wire
    has no budget.  On budget-only classes the budget is the spelling on
    the wire and ``effort`` is the intent.  ``total_budget`` was removed
    2026-09-02: ``Config.max_tokens`` is the ceiling.

    ``summary`` is visibility: ``None`` = provider default, ``"auto"`` =
    show the thinking where a knob exists (satisfied silently where the
    provider always shows it), ``"concise"``/``"detailed"`` = OpenAI's
    detail levels (RAISE elsewhere).
    """

    effort: ReasoningEffort
    thinking_budget: int | None = None
    summary: ReasoningSummary | None = None

    def __post_init__(self) -> None:
        if self.effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {self.effort}")
        if self.summary is not None and self.summary not in REASONING_SUMMARIES:
            raise ValueError(f"unsupported reasoning summary: {self.summary}")
        _coerce_int_field(self, "thinking_budget")
        _validate_positive(self.thinking_budget, field_name="thinking_budget")
        if self.effort == "off" and (self.thinking_budget is not None or self.summary is not None):
            raise ValueError(
                "Reasoning(effort='off') cannot specify thinking_budget or summary"
            )

    @property
    def is_off(self) -> bool:
        return self.effort == "off"


# ─── Cache / Prompt Caching ──────────────────────────────────

CacheRetention = Literal["short", "long"]
CacheMode = Literal["auto", "off"]
CachePrefix = Literal["stable", "history"]
CACHE_MODES = frozenset(get_args(CacheMode))
CACHE_RETENTIONS = frozenset(get_args(CacheRetention))
CACHE_PREFIXES = frozenset(get_args(CachePrefix))


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """
    Universal prompt cache configuration (MAP-6, docs/mapping-rules.md).

    Every provider caches in up to three tiers: automatic (nothing to
    send, best-effort), breakpoint (a mark on a block, guaranteed above a
    per-model minimum), resource (a stored, named, billed object).  The
    fields name INTENTS; each adapter maps an intent to the best tier it
    has, and the result is always visible in ``Usage.cache_read_tokens``.

    - mode="auto": the cheapest safe instruction (a mark on the stable
      beginning where marks exist; nothing elsewhere).
    - mode="off": send nothing; where a provider has a real off switch
      for cache WRITES, send it.
    - prefix="stable": mark the end of system + tools.
      prefix="history": mark the end of the last message (a growing
      chat).  Providers without marks fall back to the automatic tier:
      that fallback spends nothing and its outcome is observable, the
      two conditions that permit it.
    - prefix_until_index: the same intent with a precise position (the
      last block of that message).  Mutually exclusive with ``prefix``.
    - retention: "long" asks for the longer lifetime at extra cost;
      providers without one RAISE.
    - key: a best-effort routing/affinity hint; providers without one RAISE.
    - resource: the id of a stored cache object (``CacheInfo.id``) to
      reference; providers without the resource tier RAISE.
    """

    mode: CacheMode = "auto"
    retention: CacheRetention | None = None
    key: str | None = None
    prefix_until_index: int | None = None
    prefix: CachePrefix | None = None
    resource: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in CACHE_MODES:
            raise ValueError(f"unsupported cache mode: {self.mode}")
        if self.retention is not None and self.retention not in CACHE_RETENTIONS:
            raise ValueError(f"unsupported cache retention: {self.retention}")
        if self.prefix is not None and self.prefix not in CACHE_PREFIXES:
            raise ValueError(f"unsupported cache prefix: {self.prefix}")
        _validate_optional_text(self.key, field_name="CacheConfig.key", allow_empty=False)
        _validate_optional_text(self.resource, field_name="CacheConfig.resource", allow_empty=False)
        if self.mode == "off" and (
            self.retention is not None or self.key is not None
            or self.prefix is not None or self.prefix_until_index is not None or self.resource is not None
        ):
            raise ValueError("CacheConfig(mode='off') cannot specify retention, key, prefix, prefix_until_index, or resource")
        if self.prefix is not None and self.prefix_until_index is not None:
            raise ValueError("CacheConfig cannot specify both prefix and prefix_until_index")
        if self.prefix_until_index is not None:
            _coerce_int_field(self, "prefix_until_index")
            _validate_non_negative(self.prefix_until_index, field_name="prefix_until_index")


# ─── Tool Choice ───────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How the model should use tools."""

    mode: ToolChoiceMode = "auto"
    allowed: tuple[str, ...] = ()
    parallel: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in TOOL_CHOICE_MODES:
            raise ValueError(f"unsupported tool choice mode: {self.mode}")
        allowed = (self.allowed,) if isinstance(self.allowed, str) else tuple(self.allowed)
        object.__setattr__(self, "allowed", allowed)
        if any(not isinstance(name, str) or not name for name in self.allowed):
            raise ValueError("ToolChoice.allowed must contain non-empty tool names")
        _validate_optional_bool(self.parallel, field_name="ToolChoice.parallel")
        if self.mode == "none" and (self.allowed or self.parallel is not None):
            raise ValueError("ToolChoice(mode='none') cannot specify allowed or parallel")

    @classmethod
    def from_tools(
        cls,
        allowed: Tool | Sequence[Tool | str],
        *,
        mode: ToolChoiceMode = "auto",
        parallel: bool | None = None,
    ) -> "ToolChoice":
        """Create a choice by explicitly converting Tool objects to names."""
        if isinstance(allowed, str):
            names = (allowed,)
        elif isinstance(allowed, (FunctionTool, BuiltinTool)):
            names = (allowed.name,)
        else:
            names = tuple(item.name if isinstance(item, (FunctionTool, BuiltinTool)) else item for item in allowed)
        return cls(mode=mode, allowed=names, parallel=parallel)


@dataclass(frozen=True, slots=True)
class Config:
    """Generation parameters.

    Universal fields are typed.  Provider-specific settings go in
    `extensions` — a clearly-separated namespace that never pretends
    to be part of the universal schema.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    response_format: JsonObject | None = None
    tool_choice: ToolChoice | None = None
    reasoning: Reasoning | None = None
    cache: CacheConfig | None = None
    # Cross-provider knobs promoted from extensions (2026-09-01 burn-down):
    # service_tier is an OPEN string namespace — the tier concept is shared
    # (OpenAI, Anthropic), the value vocabulary is provider-owned, like
    # `voice`. user_id is an opaque end-user identifier for abuse
    # attribution (OpenAI safety_identifier, Anthropic metadata.user_id).
    # store opts a request out of (or into) provider-side response storage
    # (OpenAI and Gemini, same wire key). Providers whose wire cannot carry
    # a set knob RAISE — never silently drop.
    service_tier: str | None = None
    user_id: str | None = None
    store: bool | None = None
    # logprobs: request token log probabilities. None = do not request;
    # 0 = chosen tokens only; n > 0 = also the top-n alternatives per
    # position. Provider caps (currently 0–20) are provider-owned — lm15
    # does not hard-code them; an over-cap value maps to the provider's
    # own InvalidRequestError. Providers without logprobs (Anthropic)
    # RAISE — never silently drop.
    logprobs: int | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop or ())
        object.__setattr__(self, "stop", stop)
        _coerce_int_field(self, "max_tokens")
        _coerce_int_field(self, "top_k")
        _validate_positive(self.max_tokens, field_name="max_tokens")
        _validate_positive(self.top_k, field_name="top_k")
        for field_name in ("temperature", "top_p"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{field_name} must be numeric")
            _coerce_float_field(self, field_name)
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        if any(not isinstance(s, str) or not s for s in self.stop):
            raise ValueError("stop must contain non-empty strings")
        if self.tool_choice is not None and not isinstance(self.tool_choice, ToolChoice):
            raise TypeError("tool_choice must be a ToolChoice")
        if self.reasoning is not None and not isinstance(self.reasoning, Reasoning):
            raise TypeError("reasoning must be a Reasoning")
        if self.cache is not None and not isinstance(self.cache, CacheConfig):
            raise TypeError("cache must be a CacheConfig")
        _validate_optional_text(self.service_tier, field_name="Config.service_tier", allow_empty=False)
        _validate_optional_text(self.user_id, field_name="Config.user_id", allow_empty=False)
        if self.store is not None and not isinstance(self.store, bool):
            raise TypeError("Config.store must be a bool or None")
        _coerce_int_field(self, "logprobs")
        _validate_non_negative(self.logprobs, field_name="logprobs")
        _validate_json_field(self, "response_format")
        _validate_response_format_shape(self.response_format)
        _validate_extensions_field(self)


_RESPONSE_FORMAT_TYPES = ("json_object", "json_schema")


def _validate_response_format_shape(value: JsonObject | None) -> None:
    """INV-050 (MAP-8, 2026-09-02): response_format has exactly two shapes.

    ``{"type": "json_object"}`` — any valid JSON;
    ``{"type": "json_schema", "schema": <JSON Schema>, "name"?: str,
    "strict"?: bool}`` — this schema.  Provider-native spellings
    (``{"format": ...}``, ``{"response_mime_type": ...}``,
    ``{"json_schema": {...}}``, a bare schema) belong in ``extensions``.
    ``schema`` is opaque and verbatim: lm15 never rewrites a keyword to
    make a request pass; the provider's 400 is the contract.
    """
    if value is None:
        return
    fmt = value.get("type")
    if fmt not in _RESPONSE_FORMAT_TYPES:
        raise ValueError(
            "response_format must be {'type': 'json_object'} or {'type': 'json_schema', "
            "'schema': {...}, 'name'?: str, 'strict'?: bool}; provider-native shapes go in "
            f"Config.extensions (got keys {sorted(value)})"
        )
    allowed = {"type"} if fmt == "json_object" else {"type", "schema", "name", "strict"}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"response_format {fmt!r} does not take keys {sorted(extra)}; provider-native shapes go in Config.extensions")
    if fmt == "json_schema":
        if not isinstance(value.get("schema"), dict):
            raise ValueError("response_format json_schema requires a 'schema' object")
        if "name" in value and (not isinstance(value["name"], str) or not value["name"]):
            raise ValueError("response_format name must be a non-empty string")
        if "strict" in value and not isinstance(value["strict"], bool):
            raise TypeError("response_format strict must be a bool")


# ─── Request ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ModelRequest:
    """Base for endpoint requests/configs that require a model."""

    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model is required")


@dataclass(frozen=True, slots=True)
class Request(_ModelRequest):
    """A complete request to a foundation model.

    The composed artifact sent to the model — conversation history,
    system instructions, available tools, and generation config.
    """

    messages: tuple[Message, ...]
    system: str | tuple[PromptPart, ...] | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        messages = (self.messages,) if isinstance(self.messages, Message) else tuple(self.messages)
        object.__setattr__(self, "messages", messages)
        tools = (self.tools,) if isinstance(self.tools, (FunctionTool, BuiltinTool)) else tuple(self.tools)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "system", _normalize_system(self.system))
        if not self.messages:
            raise ValueError("at least one message is required")
        if not all(isinstance(m, Message) for m in self.messages):
            raise TypeError(
                'Request.messages must contain Message objects — wrap plain '
                'text with Message.user("...").'
            )
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("Request.tools must contain Tool objects")
        tool_names = [t.name for t in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("Request.tools cannot contain duplicate tool names")
        if not isinstance(self.config, Config):
            raise TypeError("Request.config must be a Config")
        if self.config.tool_choice is not None and self.config.tool_choice.allowed:
            missing = set(self.config.tool_choice.allowed) - set(tool_names)
            if missing:
                raise ValueError(
                    f"ToolChoice.allowed contains tools not present in Request.tools: {sorted(missing)}"
                )


# ─── Logprobs ────────────────────────────────────────────────────────
#
# Decoding telemetry: for each generated token, the chosen token's log
# probability plus a ranked list of top alternative candidates.  Two flat
# types instead of one recursive type — the wire never nests alternatives
# inside alternatives, and the type system should say so.


@dataclass(frozen=True, slots=True)
class TopLogprob:
    """One scored alternative token at a decoding step.

    ``bytes`` is the token's UTF-8 byte sequence when the provider reports
    it (OpenAI); ``token_id`` is the vocabulary id when the provider
    reports it (Gemini).  Absence means "not reported", never zero.
    """

    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None
    token_id: int | None = None

    def __post_init__(self) -> None:
        _validate_text(self.token, field_name="token")
        _coerce_float_field(self, "logprob")
        if not isinstance(self.logprob, float):
            raise TypeError("logprob must be a float")
        _validate_logprob_bytes(self)
        _coerce_int_field(self, "token_id")
        _validate_int(self.token_id, field_name="token_id")


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    """The chosen token at one decoding step, with ranked alternatives.

    ``top`` is the provider-reported ranked candidate list (descending
    logprob).  Providers differ on whether the chosen token appears in
    ``top`` — no guarantee either way, and providers also differ on
    whether the requested alternative count includes the chosen token
    (Gemini documents that it does; OpenAI counts alternatives only).
    lm15 preserves the provider-reported list as-is.
    """

    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None
    token_id: int | None = None
    top: tuple[TopLogprob, ...] = ()

    def __post_init__(self) -> None:
        _validate_text(self.token, field_name="token")
        _coerce_float_field(self, "logprob")
        if not isinstance(self.logprob, float):
            raise TypeError("logprob must be a float")
        _validate_logprob_bytes(self)
        _coerce_int_field(self, "token_id")
        _validate_int(self.token_id, field_name="token_id")
        top = (self.top,) if isinstance(self.top, TopLogprob) else tuple(self.top)
        object.__setattr__(self, "top", top)
        if not all(isinstance(t, TopLogprob) for t in self.top):
            raise TypeError("TokenLogprob.top must contain TopLogprob objects")


def _validate_logprob_bytes(obj: object) -> None:
    value = getattr(obj, "bytes")
    if value is None:
        return
    coerced = tuple(value)
    for b in coerced:
        if isinstance(b, bool) or not isinstance(b, int) or b < 0:
            raise TypeError("bytes must contain non-negative ints")
    object.__setattr__(obj, "bytes", coerced)


def _validate_logprobs_field(obj: object, field_name: str, *, allow_none: bool) -> None:
    value = getattr(obj, field_name)
    if value is None:
        if allow_none:
            return
        raise TypeError(f"{type(obj).__name__}.{field_name} must be a tuple, not None")
    coerced = (value,) if isinstance(value, TokenLogprob) else tuple(value)
    object.__setattr__(obj, field_name, coerced)
    if not all(isinstance(t, TokenLogprob) for t in coerced):
        raise TypeError(f"{type(obj).__name__}.{field_name} must contain TokenLogprob objects")


# ─── Usage ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage.

    ``input_tokens`` and ``output_tokens`` are common dimensions.
    ``total_tokens`` is the provider-reported or billed total when present;
    providers may include reasoning, audio, cache, or future token classes in
    totals differently, so it is not forced to equal input + output.

    ``reasoning_tokens`` is populated only when the provider reports an exact
    separate reasoning/thinking token count. Some providers, notably Anthropic,
    can return ``ThinkingPart`` content while reporting only combined
    ``output_tokens``; in that case ``reasoning_tokens`` remains ``None``.

    Every counter is ``int | None``: ``None`` means "the provider did not
    report this dimension", which is distinct from a reported ``0``.
    ``Usage()`` with no arguments therefore means "nothing reported" and
    serializes to ``{}`` (omitted entirely by enclosing serializers, per
    docs/serde-rules.md).

    ``total_tokens``: an explicit value is preserved as provider telemetry.
    When omitted, it auto-computes as ``input_tokens + output_tokens`` only
    when BOTH are present; if either is ``None`` the total stays ``None``.

    Arithmetic over usage (e.g. ``InferencePricing.estimate``) must treat
    ``None`` as "unknown", never as zero.

    Counters are provider-verbatim (spec/types.md, Usage, 2026-09-02): the
    same field means different things across providers. ``input_tokens``
    includes cached tokens on OpenAI, Gemini, and xAI but not on Anthropic
    (whose cache counters are disjoint); ``output_tokens`` includes
    reasoning on OpenAI and Anthropic but not on Gemini and xAI. lm15 does
    not normalise: that is what a bill reconciles against. A consumer
    comparing across providers must apply the provider's rule from the
    table in the spec; ``Response.provider_data`` and the router's
    resolution say which provider answered.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "input_audio_tokens",
            "output_audio_tokens",
        ):
            _coerce_int_field(self, field_name)
            _validate_non_negative(getattr(self, field_name), field_name=field_name)
        _coerce_int_field(self, "total_tokens")
        _validate_non_negative(self.total_tokens, field_name="total_tokens")
        if (
            self.total_tokens is None
            and self.input_tokens is not None
            and self.output_tokens is not None
        ):
            object.__setattr__(self, "total_tokens", self.input_tokens + self.output_tokens)


# ─── Response ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, repr=False)
class Response:
    """The composed artifact returned by a foundation model.

    ``Response`` keeps only minimal convenience properties.  Use
    ``response.message.first(...)`` and ``response.message.parts_of(...)``
    for variant-specific content access.
    """

    id: str | None
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    # Decoding telemetry, same category as usage/finish_reason. None =
    # provider did not report logprobs (the Usage convention). When a
    # response holds several text blocks, per-block lists concatenate in
    # document order — the block boundary survives in the stream
    # (TextDelta.part_index) and in provider_data, not here.
    logprobs: tuple[TokenLogprob, ...] | None = None
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        _validate_optional_text(self.id, field_name="Response.id", allow_empty=False)
        _validate_text(self.model, field_name="Response.model", allow_empty=False)
        if not isinstance(self.message, Message):
            raise TypeError("Response.message must be a Message")
        if self.message.role != "assistant":
            raise ValueError("Response.message must have role 'assistant'")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"unsupported finish reason: {self.finish_reason}")
        if not isinstance(self.usage, Usage):
            raise TypeError("Response.usage must be a Usage")
        _validate_logprobs_field(self, "logprobs", allow_none=True)
        _validate_json_field(self, "provider_data")

    def __repr__(self) -> str:
        display_text = self.text
        citations = self.citations

        fields = [
            ("text", repr(display_text)) if display_text is not None else ("message", repr(self.message)),
            ("model", repr(self.model)),
            ("finish_reason", repr(self.finish_reason)),
            ("usage", repr(self.usage)),
        ]
        if citations:
            fields.append(("citations", repr(citations)))
        if self.logprobs is not None:
            fields.append(("logprobs", f"<{len(self.logprobs)} tokens>"))
        if self.id is not None:
            fields.append(("id", repr(self.id)))
        if self.provider_data is not None:
            if isinstance(self.provider_data, dict):
                fields.append(("provider_data", f"<dict: {len(self.provider_data)} keys>"))
            else:
                fields.append(("provider_data", "<present>"))

        body = ",\n".join(f"    {name}={value}" for name, value in fields)
        return f"Response(\n{body},\n)"

    @property
    def text(self) -> str | None:
        """Concatenated assistant text, when the response has text.

        ``Message.text`` is intentionally strict and only returns text for
        pure-text messages.  For model responses, citation and thinking parts
        are metadata around the visible answer, so they do not make
        ``Response.text`` unavailable.
        """
        text = self.message.text
        if text is not None:
            return text

        if all(isinstance(p, (TextPart, CitationPart, ThinkingPart)) for p in self.message.parts):
            text_parts = [p.text for p in self.message.parts if isinstance(p, TextPart)]
            if text_parts:
                return "\n".join(text_parts)
        return None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.message.parts_of(ToolCallPart)

    @property
    def citations(self) -> list[CitationPart]:
        return self.message.parts_of(CitationPart)

    def parse_json(self, *, default: Any = _MISSING) -> Any:
        """Parse the response text as exact JSON."""
        t = self.text
        if t is None:
            if default is not _MISSING:
                return default
            raise ValueError(
                "Cannot parse response as JSON: response is not pure text. "
                f"Parts: {[p.type for p in self.message.parts]}"
            )
        stripped = t.strip()
        try:
            return _json.loads(stripped)
        except _json.JSONDecodeError as e:
            if default is not _MISSING:
                return default
            preview = stripped[:200] + ("..." if len(stripped) > 200 else "")
            raise ValueError(
                f"Cannot parse response as JSON: {e}\nRaw text: {preview}"
            ) from e

    @property
    def json(self) -> Any:
        """Parsed exact JSON text, or ``None`` when parsing fails.

        Valid JSON ``null`` also returns ``None``; use ``parse_json()`` when
        parse failures should be reported distinctly.
        """
        return self.parse_json(default=None)


# ─── File Upload ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True, repr=False)
class FileUploadRequest:
    """A file upload request.

    Files are an ACCOUNT-scoped resource on every provider, so there is
    no ``model`` field.  Uploads can be backed by in-memory bytes or by a
    local path.  Path-backed uploads are lazy: LMs can stream from disk
    instead of forcing the whole file into memory at construction time,
    and they are runtime-only — the canonical serializer rejects them
    because a local path is meaningless off this machine.
    """

    filename: str
    bytes_data: bytes | None = None
    media_type: str = "application/octet-stream"
    extensions: Extensions | None = None
    path: Path | None = None

    def __post_init__(self) -> None:
        _validate_text(self.filename, field_name="FileUploadRequest.filename", allow_empty=False)
        if self.path is not None and not isinstance(self.path, Path):
            if str(self.path) == "":
                raise ValueError("path cannot be empty")
            object.__setattr__(self, "path", Path(self.path))
        if self.bytes_data is None and self.path is None:
            raise TypeError("FileUploadRequest requires bytes_data or path")
        if self.bytes_data is not None and self.path is not None:
            raise ValueError("FileUploadRequest requires exactly one of bytes_data or path")
        if self.bytes_data is not None:
            if not isinstance(self.bytes_data, (bytes, bytearray)):
                raise TypeError("bytes_data must be bytes")
            if not self.bytes_data:
                raise ValueError("bytes_data is required")
            if isinstance(self.bytes_data, bytearray):
                object.__setattr__(self, "bytes_data", bytes(self.bytes_data))
        if self.path is not None and str(self.path) == "":
            raise ValueError("path cannot be empty")
        _validate_text(self.media_type, field_name="FileUploadRequest.media_type", allow_empty=False)
        _validate_extensions_field(self)

    def __repr__(self) -> str:
        return (
            "FileUploadRequest("
            f"filename={self.filename!r}, "
            f"bytes_data={_bytes_summary(self.bytes_data)!r}, "
            f"media_type={self.media_type!r}, "
            f"extensions={self.extensions!r}, "
            f"path={self.path!r})"
        )

    @property
    def bytes(self) -> bytes:
        if self.bytes_data is not None:
            return self.bytes_data
        if self.path is not None:
            return self.path.read_bytes()
        raise ValueError("FileUploadRequest has neither bytes_data nor path")


@dataclass(frozen=True, slots=True)
class FileInfo:
    """A snapshot of one provider-side stored file.

    ``id`` is the canonical reference for this provider: paste it into a
    media Part's ``file_id`` and the chat mapping places it on the wire
    (OpenAI/Anthropic file ids verbatim; Gemini the file URI, because
    Gemini requests address files by URI, not resource name).

    Optional fields stay ``None`` when the provider does not report the
    value (OpenAI reports no MIME type; only Gemini reports expiry).
    ``downloadable`` is tri-state: ``True``/``False`` when the provider
    states the capability, ``None`` when it does not say.
    """

    id: str
    filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    created_at: str | None = None
    expires_at: str | None = None
    readiness: FileReadiness = "ready"
    downloadable: bool | None = None
    provider_data: ProviderData | None = None

    @property
    def ready(self) -> bool:
        return self.readiness == "ready"

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="FileInfo.id", allow_empty=False)
        _validate_optional_text(self.filename, field_name="FileInfo.filename", allow_empty=False)
        _validate_optional_text(self.media_type, field_name="FileInfo.media_type", allow_empty=False)
        _coerce_int_field(self, "size_bytes")
        _validate_non_negative(self.size_bytes, field_name="FileInfo.size_bytes")
        _validate_optional_text(self.created_at, field_name="FileInfo.created_at", allow_empty=False)
        _validate_optional_text(self.expires_at, field_name="FileInfo.expires_at", allow_empty=False)
        if self.readiness not in FILE_READINESS_VALUES:
            raise ValueError(f"unsupported file readiness: {self.readiness}")
        if self.downloadable is not None and not isinstance(self.downloadable, bool):
            raise TypeError("FileInfo.downloadable must be a bool or None")
        _validate_json_field(self, "provider_data")


@dataclass(frozen=True, slots=True)
class FilePage:
    """One page of stored files plus an opaque continuation cursor.

    ``next_cursor`` is provider-issued and opaque: pass it back to
    ``file_list`` to fetch the next page; ``None`` means the listing is
    complete.
    """

    items: tuple[FileInfo, ...] = ()
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not all(isinstance(item, FileInfo) for item in self.items):
            raise TypeError("FilePage.items must contain FileInfo objects")
        _validate_optional_text(self.next_cursor, field_name="FilePage.next_cursor", allow_empty=False)


# ─── Cache resources (the stored tier of MAP-6) ──────────────────────
#
# A stored cache object belongs to one model on every provider that has
# the tier (a KV state is model-specific by nature).  ``CacheInfo`` is the
# snapshot; ``CachedPrefix`` is the ergonomic that pairs a prefix Request
# with its optional object and builds requests against it.  Nothing here
# names a provider: id formats, TTL spellings, and what the wire does with
# ``CacheConfig.resource`` are the adapter's business.


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """A snapshot of one provider-side stored cache object.

    ``id`` is the provider's reference verbatim; place it in
    ``CacheConfig.resource``.  ``tokens`` is the stored token count when
    reported (storage cost = tokens x hours x the provider's rate).
    ``expires_at`` is ISO-8601 UTC, normalized from the provider's form.
    """

    id: str
    model: str
    tokens: int | None = None
    created_at: str | None = None
    expires_at: str | None = None
    label: str | None = None
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="CacheInfo.id", allow_empty=False)
        _validate_text(self.model, field_name="CacheInfo.model", allow_empty=False)
        _coerce_int_field(self, "tokens")
        _validate_non_negative(self.tokens, field_name="CacheInfo.tokens")
        _validate_optional_text(self.created_at, field_name="CacheInfo.created_at", allow_empty=False)
        _validate_optional_text(self.expires_at, field_name="CacheInfo.expires_at", allow_empty=False)
        _validate_optional_text(self.label, field_name="CacheInfo.label", allow_empty=False)
        _validate_json_field(self, "provider_data")


@dataclass(frozen=True, slots=True)
class CachePage:
    """One page of stored cache objects plus an opaque continuation cursor."""

    items: tuple[CacheInfo, ...] = ()
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        if not all(isinstance(item, CacheInfo) for item in self.items):
            raise TypeError("CachePage.items must contain CacheInfo objects")
        _validate_optional_text(self.next_cursor, field_name="CachePage.next_cursor", allow_empty=False)


@dataclass(frozen=True, slots=True)
class CachedPrefix:
    """A reusable prompt beginning: ``cached = lm.cache(prefix)``.

    ``prefix`` is a Request whose model, system, tools, and messages form
    the reusable part (its config must be default: a cached object has no
    temperature).  ``resource`` is the stored object on providers with
    that tier, ``None`` on providers that mark blocks or cache
    automatically.  ``request(messages)`` builds a Request that appends
    ``messages`` and sets the cache boundary at the seam; ``cached +
    messages`` is Python sugar for it.  The built Request and its wire
    bytes are what the contract pins; the sugar is per-language.
    """

    prefix: Request
    resource: CacheInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, Request):
            raise TypeError("CachedPrefix.prefix must be a Request")
        if self.prefix.config != Config():
            raise ValueError("CachedPrefix.prefix must carry a default Config: a cached object has no generation settings")
        if self.resource is not None:
            if not isinstance(self.resource, CacheInfo):
                raise TypeError("CachedPrefix.resource must be a CacheInfo")
            if self.resource.model != self.prefix.model:
                raise ValueError("CachedPrefix.resource.model must equal the prefix model (a stored cache belongs to one model)")

    @property
    def id(self) -> str | None:
        return self.resource.id if self.resource is not None else None

    @property
    def expires_at(self) -> str | None:
        return self.resource.expires_at if self.resource is not None else None

    def cache_config(self) -> CacheConfig:
        """The CacheConfig that marks the seam between prefix and suffix."""
        return CacheConfig(
            prefix_until_index=len(self.prefix.messages) - 1,
            resource=self.id,
        )

    def request(
        self,
        messages: "str | Message | Sequence[Message] | Request",
        *,
        config: Config | None = None,
    ) -> Request:
        """Append ``messages`` to the prefix and set the cache boundary.

        ``messages`` may be a string (one user message), a Message, a
        sequence of Messages, or a Request (same model, no system, no
        tools: the prefix owns those).  ``config`` supplies generation
        settings; its ``cache`` must be unset (the prefix decides).
        """
        suffix: tuple[Message, ...]
        if isinstance(messages, Request):
            if messages.model != self.prefix.model:
                raise ValueError("suffix Request model must equal the prefix model")
            if messages.system is not None or messages.tools:
                raise ValueError("suffix Request cannot redefine system or tools: the prefix owns them")
            if config is None and messages.config != Config():
                config = messages.config
            suffix = messages.messages
        elif isinstance(messages, str):
            suffix = (Message.user(messages),)
        elif isinstance(messages, Message):
            suffix = (messages,)
        else:
            suffix = tuple(messages)
            if not suffix or not all(isinstance(m, Message) for m in suffix):
                raise TypeError("messages must be a Message or a non-empty sequence of Messages")
        base = config if config is not None else Config()
        if base.cache is not None:
            raise ValueError("config.cache is decided by the CachedPrefix; leave it unset")
        from dataclasses import replace as _replace

        return Request(
            model=self.prefix.model,
            system=self.prefix.system,
            tools=self.prefix.tools,
            messages=self.prefix.messages + suffix,
            config=_replace(base, cache=self.cache_config()),
        )

    def __add__(self, messages: "str | Message | Sequence[Message] | Request") -> Request:
        return self.request(messages)


# ─── Batch ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """A batch of model requests.

    Each nested Request carries its own model.  The optional top-level model is
    only a routing/default convenience; when omitted, it is inferred from the
    first nested request and does not constrain the rest of the batch.
    """

    model: str | None = None
    requests: tuple[Request, ...] = ()
    label: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests:
            raise ValueError("requests cannot be empty")
        if not all(isinstance(r, Request) for r in self.requests):
            raise TypeError("BatchRequest.requests must contain Request objects")
        _validate_optional_text(self.model, field_name="BatchRequest.model", allow_empty=False)
        if self.model is None:
            object.__setattr__(self, "model", self.requests[0].model)
        _validate_optional_text(self.label, field_name="BatchRequest.label", allow_empty=False)
        _validate_extensions_field(self)


@dataclass(frozen=True, slots=True)
class BatchJobInfo:
    """A snapshot of a provider-side batch job — the ticket.

    ``id`` is the provider's job id, a plain string; store it anywhere.
    ``created_at`` is normalized to an ISO-8601 UTC string when the
    provider reports a creation time. Raw job state stays verbatim in
    ``provider_data``.
    """

    id: str
    status: BatchStatus
    label: str | None = None
    created_at: str | None = None
    provider_data: ProviderData | None = None

    @property
    def done(self) -> bool:
        return self.status in BATCH_TERMINAL_STATUSES

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="BatchJobInfo.id", allow_empty=False)
        if self.status not in BATCH_STATUSES:
            raise ValueError(f"unsupported batch status: {self.status}")
        _validate_optional_text(self.label, field_name="BatchJobInfo.label", allow_empty=False)
        _validate_optional_text(self.created_at, field_name="BatchJobInfo.created_at", allow_empty=False)
        _validate_json_field(self, "provider_data")


@dataclass(frozen=True, slots=True)
class BatchEntry:
    """The fate of one request in a batch, in submission order.

    Partial failure is a first-class outcome: a ``completed`` job may mix
    ``succeeded`` and ``errored`` entries. ``succeeded`` carries a full
    canonical Response; ``errored`` carries a canonical ErrorDetail;
    ``cancelled`` and ``expired`` carry neither.
    """

    index: int
    outcome: BatchOutcome
    response: Response | None = None
    error: ErrorDetail | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == "succeeded"

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("BatchEntry.index must be a non-negative int")
        if self.outcome not in BATCH_OUTCOMES:
            raise ValueError(f"unsupported batch outcome: {self.outcome}")
        if self.outcome == "succeeded":
            if not isinstance(self.response, Response) or self.error is not None:
                raise ValueError("succeeded entries carry a Response and no error")
        elif self.outcome == "errored":
            if not isinstance(self.error, ErrorDetail) or self.response is not None:
                raise ValueError("errored entries carry an ErrorDetail and no response")
        elif self.response is not None or self.error is not None:
            raise ValueError(f"{self.outcome} entries carry neither response nor error")


@dataclass(frozen=True, slots=True)
class _PromptRequest(_ModelRequest):
    """Base for endpoint requests that require a non-empty text prompt."""

    prompt: str

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if self.prompt == "":
            raise ValueError("prompt is required")


# ─── Image Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest(_PromptRequest):
    """Text (and optionally input images, for edits) in; images out.

    ``size`` takes the provider's own sizing vocabulary — like ``model``
    and ``voice``, the field is portable but the values are not (OpenAI:
    pixels; Gemini: aspect ratios).  ``images`` are ordinary ImageParts
    (inline data / url / file_id / path); adapters route them to the
    provider's real edit door and raise where the wire has none.
    """

    size: str | None = None
    images: tuple[ImagePart, ...] = ()
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        _validate_optional_text(self.size, field_name="size", allow_empty=False)
        object.__setattr__(self, "images", tuple(self.images))
        if not all(isinstance(img, ImagePart) for img in self.images):
            raise TypeError("ImageGenerationRequest.images must contain ImagePart objects")
        _validate_extensions_field(self)


@dataclass(frozen=True, slots=True)
class ImageGenerationResponse:
    """Generated images, plus any text the model said while drawing
    (Gemini routinely narrates; dropping it would be silent data loss)."""

    images: tuple[ImagePart, ...]
    text: str | None = None
    id: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        _validate_optional_text(self.text, field_name="ImageGenerationResponse.text", allow_empty=False)
        _validate_optional_text(self.id, field_name="ImageGenerationResponse.id", allow_empty=False)
        _validate_optional_text(self.model, field_name="ImageGenerationResponse.model", allow_empty=False)
        if not isinstance(self.usage, Usage):
            raise TypeError("ImageGenerationResponse.usage must be a Usage")
        object.__setattr__(self, "images", tuple(self.images))
        if not self.images:
            raise ValueError("ImageGenerationResponse requires at least one image")
        if not all(isinstance(img, ImagePart) for img in self.images):
            raise TypeError("images must contain ImagePart objects")
        _validate_json_field(self, "provider_data")


# ─── Speech Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpeechGenerationRequest(_PromptRequest):
    """Text-to-speech.  Named for what every wire actually sells: speech.

    ``voice`` and ``format`` take the provider's own vocabularies.  An
    omitted field means the SERVER decides (OpenAI's real default format
    is MP3); lm15 injects no defaults of its own.  ``format`` raises on
    providers whose wire has no slot for it (Gemini: always PCM).
    """

    voice: str | None = None
    format: str | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        _validate_optional_text(self.voice, field_name="voice", allow_empty=False)
        _validate_optional_text(self.format, field_name="format", allow_empty=False)
        _validate_extensions_field(self)


@dataclass(frozen=True, slots=True)
class SpeechGenerationResponse:
    audio: AudioPart
    id: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider_data: ProviderData | None = None

    def __post_init__(self) -> None:
        _validate_optional_text(self.id, field_name="SpeechGenerationResponse.id", allow_empty=False)
        _validate_optional_text(self.model, field_name="SpeechGenerationResponse.model", allow_empty=False)
        if not isinstance(self.usage, Usage):
            raise TypeError("SpeechGenerationResponse.usage must be a Usage")
        if not isinstance(self.audio, AudioPart):
            raise TypeError("audio must be an AudioPart")
        _validate_json_field(self, "provider_data")


# ─── Video Generation ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class VideoGenerationRequest(_PromptRequest):
    """Text (and optionally an input image) in; a video JOB out.

    Video is job-shaped on every wire that sells it (Sora, Veo,
    grok-imagine): submission returns a ticket, not bytes.  ``seconds``
    maps to the provider's duration knob and raises where the wire has
    none; ``images`` are input frames (image-to-video), ordinary
    ImageParts.
    """

    seconds: int | None = None
    images: tuple[ImagePart, ...] = ()
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _PromptRequest.__post_init__(self)
        if self.seconds is not None and (
            not isinstance(self.seconds, int) or isinstance(self.seconds, bool) or self.seconds <= 0
        ):
            raise ValueError("VideoGenerationRequest.seconds must be a positive int")
        object.__setattr__(self, "images", tuple(self.images))
        if not all(isinstance(img, ImagePart) for img in self.images):
            raise TypeError("VideoGenerationRequest.images must contain ImagePart objects")
        _validate_extensions_field(self)


@dataclass(frozen=True, slots=True)
class VideoJobInfo:
    """A snapshot of a provider-side video job — the ticket.

    ``id`` is the provider's job id, a plain string; store it anywhere
    (on xAI it is the ONLY copy — the wire has no list endpoint).
    ``progress`` is a percentage when the provider reports one.  Raw job
    state stays verbatim in ``provider_data``.
    """

    id: str
    status: VideoStatus
    progress: int | None = None
    created_at: str | None = None
    model: str | None = None
    provider_data: ProviderData | None = None

    @property
    def done(self) -> bool:
        return self.status in VIDEO_TERMINAL_STATUSES

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="VideoJobInfo.id", allow_empty=False)
        if self.status not in VIDEO_STATUSES:
            raise ValueError(f"unsupported video status: {self.status}")
        if self.progress is not None and (
            not isinstance(self.progress, int) or isinstance(self.progress, bool)
            or not 0 <= self.progress <= 100
        ):
            raise ValueError("VideoJobInfo.progress must be an int percentage 0-100")
        _validate_optional_text(self.created_at, field_name="VideoJobInfo.created_at", allow_empty=False)
        _validate_optional_text(self.model, field_name="VideoJobInfo.model", allow_empty=False)
        _validate_json_field(self, "provider_data")


EndpointRequest: TypeAlias = (
    Request
    | FileUploadRequest
    | BatchRequest
    | ImageGenerationRequest
    | SpeechGenerationRequest
    | VideoGenerationRequest
)

EndpointResponse: TypeAlias = (
    Response
    | FileInfo
    | BatchJobInfo
    | ImageGenerationResponse
    | SpeechGenerationResponse
    | VideoJobInfo
)


# ─── Audio Format ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AudioFormat:
    encoding: AudioEncoding
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if self.encoding not in AUDIO_ENCODINGS:
            raise ValueError(f"unsupported audio encoding: {self.encoding}")
        _coerce_int_field(self, "sample_rate")
        _coerce_int_field(self, "channels")
        _validate_positive(self.sample_rate, field_name="sample_rate")
        _validate_positive(self.channels, field_name="channels")


# ─── Live (Realtime) ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LiveConfig(_ModelRequest):
    system: str | tuple[PromptPart, ...] | None = None
    tools: tuple[Tool, ...] = ()
    voice: str | None = None
    input_format: AudioFormat | None = None
    output_format: AudioFormat | None = None
    extensions: Extensions | None = None

    def __post_init__(self) -> None:
        _ModelRequest.__post_init__(self)
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "system", _normalize_system(self.system))
        if not all(isinstance(t, (FunctionTool, BuiltinTool)) for t in self.tools):
            raise TypeError("LiveConfig.tools must contain Tool objects")
        tool_names = [t.name for t in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("LiveConfig.tools cannot contain duplicate tool names")
        if self.input_format is not None and not isinstance(self.input_format, AudioFormat):
            raise TypeError("input_format must be an AudioFormat")
        if self.output_format is not None and not isinstance(self.output_format, AudioFormat):
            raise TypeError("output_format must be an AudioFormat")
        _validate_optional_text(self.voice, field_name="LiveConfig.voice", allow_empty=False)
        _validate_extensions_field(self)


@dataclass(frozen=True, slots=True)
class LiveClientTurnEvent:
    parts: tuple[PromptPart, ...]
    turn_complete: bool = True
    type: Literal["turn"] = field(default="turn", init=False)

    def __post_init__(self) -> None:
        parts = (self.parts,) if _is_part(self.parts) else tuple(self.parts)
        object.__setattr__(self, "parts", parts)
        if not self.parts:
            raise ValueError("LiveClientTurnEvent requires at least one part")
        if not all(_is_part(p) for p in self.parts):
            raise TypeError("LiveClientTurnEvent.parts must contain Part objects")
        if any(isinstance(p, _PROMPT_FORBIDDEN_PARTS) for p in self.parts):
            raise TypeError("LiveClientTurnEvent.parts cannot contain model/tool protocol parts")
        _validate_bool(self.turn_complete, field_name="LiveClientTurnEvent.turn_complete")


@dataclass(frozen=True, slots=True)
class LiveClientAudioEvent:
    data: str
    media_type: str = "audio/pcm;rate=16000"
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.data, field_name="LiveClientAudioEvent.data", allow_empty=False)
        _validate_base64_data("LiveClientAudioEvent", self.data)
        _validate_text(self.media_type, field_name="LiveClientAudioEvent.media_type", allow_empty=False)
        if not self.media_type.startswith("audio/"):
            raise ValueError("LiveClientAudioEvent.media_type must start with 'audio/'")


@dataclass(frozen=True, slots=True)
class LiveClientImageEvent:
    data: str
    media_type: str = "image/jpeg"
    type: Literal["image"] = field(default="image", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.data, field_name="LiveClientImageEvent.data", allow_empty=False)
        _validate_base64_data("LiveClientImageEvent", self.data)
        _validate_text(self.media_type, field_name="LiveClientImageEvent.media_type", allow_empty=False)
        if not self.media_type.startswith("image/"):
            raise ValueError("LiveClientImageEvent.media_type must start with 'image/'")


@dataclass(frozen=True, slots=True)
class LiveClientTextEvent:
    text: str
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="LiveClientTextEvent.text")


@dataclass(frozen=True, slots=True)
class LiveClientToolResultEvent:
    id: str
    content: tuple[ToolResultContentPart, ...]
    type: Literal["tool_result"] = field(default="tool_result", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="LiveClientToolResultEvent.id", allow_empty=False)
        object.__setattr__(self, "content", tuple(self.content))
        if not self.content:
            raise ValueError("LiveClientToolResultEvent requires content")
        if not all(_is_part(p) for p in self.content):
            raise TypeError("LiveClientToolResultEvent.content must contain Part objects")
        if any(isinstance(p, _TOOL_RESULT_FORBIDDEN_PARTS) for p in self.content):
            raise TypeError("LiveClientToolResultEvent.content cannot contain model or protocol parts")


@dataclass(frozen=True, slots=True)
class LiveClientInterruptEvent:
    type: Literal["interrupt"] = field(default="interrupt", init=False)


@dataclass(frozen=True, slots=True)
class LiveClientEndAudioEvent:
    type: Literal["end_audio"] = field(default="end_audio", init=False)


LiveClientEvent: TypeAlias = (
    LiveClientTurnEvent
    | LiveClientAudioEvent
    | LiveClientImageEvent
    | LiveClientTextEvent
    | LiveClientToolResultEvent
    | LiveClientInterruptEvent
    | LiveClientEndAudioEvent
)
LIVE_CLIENT_EVENT_CLASSES: tuple[type, ...] = get_args(LiveClientEvent)


@dataclass(frozen=True, slots=True)
class LiveServerAudioEvent:
    data: str
    media_type: str | None = None
    type: Literal["audio"] = field(default="audio", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.data, field_name="LiveServerAudioEvent.data", allow_empty=False)
        _validate_base64_data("LiveServerAudioEvent", self.data)
        _validate_optional_text(self.media_type, field_name="LiveServerAudioEvent.media_type", allow_empty=False)
        if self.media_type is not None and not self.media_type.startswith("audio/"):
            raise ValueError("LiveServerAudioEvent.media_type must start with 'audio/'")


@dataclass(frozen=True, slots=True)
class LiveServerTextEvent:
    text: str
    type: Literal["text"] = field(default="text", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.text, field_name="LiveServerTextEvent.text")


@dataclass(frozen=True, slots=True)
class LiveServerToolCallEvent:
    id: str
    name: str
    input: JsonObject
    type: Literal["tool_call"] = field(default="tool_call", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="LiveServerToolCallEvent.id", allow_empty=False)
        _validate_text(self.name, field_name="LiveServerToolCallEvent.name", allow_empty=False)
        _validate_json_field(self, "input", required=True)


@dataclass(frozen=True, slots=True)
class LiveServerToolCallDeltaEvent:
    input_delta: str
    id: str | None = None
    name: str | None = None
    type: Literal["tool_call_delta"] = field(default="tool_call_delta", init=False)

    def __post_init__(self) -> None:
        _validate_text(self.input_delta, field_name="LiveServerToolCallDeltaEvent.input_delta")
        _validate_optional_text(self.id, field_name="LiveServerToolCallDeltaEvent.id", allow_empty=False)
        _validate_optional_text(self.name, field_name="LiveServerToolCallDeltaEvent.name", allow_empty=False)


@dataclass(frozen=True, slots=True)
class LiveServerInterruptedEvent:
    type: Literal["interrupted"] = field(default="interrupted", init=False)


@dataclass(frozen=True, slots=True)
class LiveServerTurnEndEvent:
    usage: Usage
    type: Literal["turn_end"] = field(default="turn_end", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.usage, Usage):
            raise TypeError("LiveServerTurnEndEvent.usage must be a Usage")


@dataclass(frozen=True, slots=True)
class LiveServerUsageEvent:
    """Billed usage for a response that did not end the turn.

    A function-call response (the model waits for results; the turn stays
    open) and a cancelled response (barge-in) both consume tokens the
    provider reports. ``turn_end`` carries the usage of a completed turn;
    this event carries the usage of everything else, so a session's bill
    is the sum of every ``usage`` and ``turn_end`` event. Never a turn
    boundary; dispatch loops ignore it.
    """

    usage: Usage
    type: Literal["usage"] = field(default="usage", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.usage, Usage):
            raise TypeError("LiveServerUsageEvent.usage must be a Usage")


@dataclass(frozen=True, slots=True)
class LiveServerErrorEvent:
    error: ErrorDetail
    type: Literal["error"] = field(default="error", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.error, ErrorDetail):
            raise TypeError("LiveServerErrorEvent.error must be an ErrorDetail")


LiveServerEvent: TypeAlias = (
    LiveServerAudioEvent
    | LiveServerTextEvent
    | LiveServerToolCallEvent
    | LiveServerToolCallDeltaEvent
    | LiveServerInterruptedEvent
    | LiveServerTurnEndEvent
    | LiveServerUsageEvent
    | LiveServerErrorEvent
)
LIVE_SERVER_EVENT_CLASSES: tuple[type, ...] = get_args(LiveServerEvent)


# ─── ToolCallInfo (for callbacks) ────────────────────────────────────
#
# Lightweight callback payload: same identity/input shape as ToolCallPart,
# but without the content-part discriminator.


@dataclass(frozen=True, slots=True)
class ToolCallInfo:
    """Callback view of a tool call without the part discriminator."""

    id: str
    name: str
    input: JsonObject

    def __post_init__(self) -> None:
        _validate_text(self.id, field_name="ToolCallInfo.id", allow_empty=False)
        _validate_text(self.name, field_name="ToolCallInfo.name", allow_empty=False)
        _validate_json_field(self, "input", required=True)

    @classmethod
    def from_part(cls, part: ToolCallPart) -> "ToolCallInfo":
        if not isinstance(part, ToolCallPart):
            raise TypeError("ToolCallInfo.from_part() requires a ToolCallPart")
        return cls(id=part.id, name=part.name, input=part.input)

    def to_part(self) -> ToolCallPart:
        return ToolCallPart(id=self.id, name=self.name, input=self.input)


def _check_literal_vocabularies() -> None:
    checks = (
        ("PartType", set(get_args(PartType)), set(PART_TYPES)),
        ("DeltaType", set(get_args(DeltaType)), set(DELTA_TYPES)),
        ("ErrorCode", set(get_args(ErrorCode)), set(ERROR_CODES)),
        ("FinishReason", set(get_args(FinishReason)), set(FINISH_REASONS)),
        ("Role", set(get_args(Role)), set(ROLE_VALUES)),
        ("StreamEventType", set(get_args(StreamEventType)), {_variant_type(cls) for cls in STREAM_EVENT_CLASSES}),
        ("BatchStatus", set(get_args(BatchStatus)), set(BATCH_STATUSES)),
        ("BatchOutcome", set(get_args(BatchOutcome)), set(BATCH_OUTCOMES)),
        ("FileReadiness", set(get_args(FileReadiness)), set(FILE_READINESS_VALUES)),
        ("AudioEncoding", set(get_args(AudioEncoding)), set(AUDIO_ENCODINGS)),
        ("ToolChoiceMode", set(get_args(ToolChoiceMode)), set(TOOL_CHOICE_MODES)),
        ("ReasoningEffort", set(get_args(ReasoningEffort)), set(REASONING_EFFORTS)),
        ("ReasoningSummary", set(get_args(ReasoningSummary)), set(REASONING_SUMMARIES)),
        ("LiveClientEventType", set(get_args(LiveClientEventType)), {_variant_type(cls) for cls in LIVE_CLIENT_EVENT_CLASSES}),
        ("LiveServerEventType", set(get_args(LiveServerEventType)), {_variant_type(cls) for cls in LIVE_SERVER_EVENT_CLASSES}),
    )
    for name, literal_values, runtime_values in checks:
        if literal_values != runtime_values:
            raise RuntimeError(
                f"{name} literal values must match runtime vocabulary: "
                f"literal={sorted(literal_values)} runtime={sorted(runtime_values)}"
            )


_check_literal_vocabularies()
