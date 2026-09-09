"""
lm15.result — Stream materialization.

One engine and two skins, all speaking the canonical StreamEvent
vocabulary:

- ``StreamAccumulator`` — the push-based engine.  Feed it events, ask it
  for the ``Response``.  It is the shape every port shares (Rust/Go/TS
  have no generators) and the only place accumulation logic lives.
- ``ResponseStream`` / ``AsyncResponseStream`` — lazy iteration sugar
  over a live stream: iterate for text as it arrives, ``.events()`` for
  the canonical typed events, ``.response`` afterwards for the same
  Response a non-streaming call returns.
- ``materialize_response`` / ``amaterialize_response`` — the one-shot
  functional form.

Nothing here executes anything: tool calls are surfaced as data, and
any execute-tools-until-done loop belongs to the layer above lm15.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Iterator

from .errors import LM15Error, error_class_for_code
from .types import (
    AudioDelta,
    AudioPart,
    CitationDelta,
    CitationPart,
    ContinuationDelta,
    ContinuationState,
    ErrorDetail,
    ImageDelta,
    ImagePart,
    JsonObject,
    Message,
    Part,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    TokenLogprob,
    ToolCallDelta,
    ToolCallPart,
    Usage,
)

__all__ = [
    "StreamAccumulator",
    "ResponseStream",
    "AsyncResponseStream",
    "materialize_response",
    "amaterialize_response",
    "response_to_events",
    "coalesce_stream",
    "acoalesce_stream",
]


@dataclass(slots=True)
class StreamAccumulator:
    """Accumulates canonical stream events into a complete Response.

    Push-based so that sync iteration, async iteration, and one-shot
    materialization all share the same accumulation logic:

        acc = StreamAccumulator(request)
        for event in lm.stream(request):
            acc.push(event)
        response = acc.response()

    ``push`` ignores error events — deciding whether to raise is the
    caller's job (``ResponseStream`` raises; a resumption layer might
    not).  ``response()`` materializes whatever has been accumulated so
    far; callers normally push through the end event first.
    """

    request: Request
    started_id: str | None = None
    started_model: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    text_parts: dict[int, list[str]] = field(default_factory=dict)
    thinking_parts: dict[int, list[str]] = field(default_factory=dict)
    audio_chunks: dict[int, list[str]] = field(default_factory=dict)
    audio_media_types: dict[int, str | None] = field(default_factory=dict)
    image_parts: dict[int, ImagePart] = field(default_factory=dict)
    citation_parts: dict[int, list[CitationPart]] = field(default_factory=dict)
    tool_call_raw: dict[int, str] = field(default_factory=dict)
    tool_call_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    message_continuation: list[ContinuationState] = field(default_factory=list)
    part_continuation: dict[int, list[ContinuationState]] = field(default_factory=dict)
    logprob_seq: list[TokenLogprob] = field(default_factory=list)
    provider_data: dict[str, Any] | None = None

    def push(self, event: StreamEvent) -> None:
        """Fold one canonical stream event into the accumulated state."""
        if event.type == "start":
            self.started_id = event.id or self.started_id
            self.started_model = event.model or self.started_model
            return

        if event.type == "end":
            self.finish_reason = event.finish_reason or self.finish_reason
            self.usage = event.usage or self.usage
            if event.provider_data is not None:
                self.provider_data = event.provider_data
            return

        if event.type != "delta" or event.delta is None:
            return

        delta = event.delta

        if delta.type == "text":
            self.text_parts.setdefault(delta.part_index, []).append(delta.text or "")
            if delta.logprobs:
                self.logprob_seq.extend(delta.logprobs)

        elif delta.type == "thinking":
            self.thinking_parts.setdefault(delta.part_index, []).append(delta.text or "")

        elif delta.type == "audio":
            self.audio_chunks.setdefault(delta.part_index, []).append(delta.data or "")
            self.audio_media_types.setdefault(delta.part_index, delta.media_type)

        elif delta.type == "tool_call":
            idx = delta.part_index
            meta = self.tool_call_meta.setdefault(idx, {})
            if delta.id is not None:
                meta["id"] = str(delta.id)
            if delta.name is not None:
                meta["name"] = str(delta.name)
            aggregate = self.tool_call_raw.get(idx, "") + delta.input
            self.tool_call_raw[idx] = aggregate
            meta["input"] = _parse_json_best_effort(aggregate)

        elif delta.type == "image":
            mt = delta.media_type or "image/png"
            if delta.data is not None:
                part = ImagePart(media_type=mt, data=str(delta.data))
            elif delta.url is not None:
                part = ImagePart(media_type=mt, url=str(delta.url))
            elif delta.file_id is not None:
                part = ImagePart(media_type=mt, file_id=str(delta.file_id))
            else:
                return
            self.image_parts[delta.part_index] = part

        elif delta.type == "citation":
            self.citation_parts.setdefault(delta.part_index, []).append(CitationPart(
                text=delta.text, url=delta.url, title=delta.title,
            ))

        elif delta.type == "continuation":
            state = delta.to_state()
            if delta.part_index is None:
                self.message_continuation.append(state)
            else:
                self.part_continuation.setdefault(delta.part_index, []).append(state)

    def response(self) -> Response:
        """Build a complete Response from the accumulated state.

        Raises :class:`StreamAssemblyError` when a tool call's fragments
        never carried a name (MAP-9): the error carries everything else
        that did assemble as ``partial``.
        """
        unnamed = sorted(
            idx for idx, meta in self.tool_call_meta.items() if not meta.get("name")
        )
        if unnamed:
            partial = self._assemble(skip=frozenset(unnamed))
            from .errors import StreamAssemblyError

            raise StreamAssemblyError(
                f"tool call at part {unnamed[0]} arrived without a name; the adapter "
                "that produced this stream must set ToolCallDelta.name on the call's "
                "first fragment (MAP-9: lm15 does not guess which tool the model meant)",
                partial=partial,
                part_index=unnamed[0],
            )
        return self._assemble(skip=frozenset())

    def _assemble(self, *, skip: frozenset[int]) -> Response:
        parts: list[Part] = []

        part_indexes = sorted(
            set(self.thinking_parts)
            | set(self.text_parts)
            | set(self.image_parts)
            | set(self.audio_chunks)
            | set(self.citation_parts)
            | set(self.tool_call_meta)
            | set(self.part_continuation)
        )

        for idx in part_indexes:
            continuation = tuple(self.part_continuation.get(idx, ()))
            has_tool = idx in self.tool_call_meta and idx not in skip
            if idx in self.thinking_parts:
                parts.append(ThinkingPart(text="".join(self.thinking_parts[idx]), continuation=continuation))
            if idx in self.text_parts:
                parts.append(TextPart(text="".join(self.text_parts[idx]), continuation=continuation))
            if idx in self.image_parts:
                parts.append(replace(self.image_parts[idx], continuation=continuation))
            if idx in self.audio_chunks:
                raw_data = _concat_b64_chunks(self.audio_chunks[idx])
                media_type = self.audio_media_types.get(idx)
                from .types import audio as make_audio
                if media_type in (None, "audio/pcm", "audio/pcm16"):
                    parts.append(make_audio(data=_pcm_to_wav(raw_data), media_type="audio/wav", continuation=continuation))
                else:
                    parts.append(make_audio(data=raw_data, media_type=media_type, continuation=continuation))
            if idx in self.citation_parts:
                parts.extend(replace(part, continuation=continuation) for part in self.citation_parts[idx])
            if has_tool:
                meta = self.tool_call_meta[idx]
                payload = meta.get("input")
                if not isinstance(payload, dict):
                    payload = _parse_json_best_effort(self.tool_call_raw.get(idx, ""))
                # A missing id gets an lm15-minted correlator (Gemini sends
                # none); a missing name is never minted — see response().
                tc_id = str(meta.get("id") or f"tool_call_{idx}")
                parts.append(ToolCallPart(id=tc_id, name=str(meta["name"]), input=payload, continuation=continuation))
            elif (
                idx not in skip
                and idx not in self.thinking_parts
                and idx not in self.text_parts
                and idx not in self.image_parts
                and idx not in self.audio_chunks
                and idx not in self.citation_parts
            ):
                parts.append(TextPart(text="", continuation=continuation))

        if not parts:
            parts = [TextPart(text="")]

        finish = self.finish_reason
        has_tool_calls = any(isinstance(p, ToolCallPart) for p in parts)
        if finish is None:
            finish = "tool_call" if has_tool_calls else "stop"
        elif finish == "stop" and has_tool_calls:
            finish = "tool_call"

        return Response(
            id=self.started_id,
            model=self.started_model or self.request.model,
            message=Message(role="assistant", parts=tuple(parts), continuation=tuple(self.message_continuation)),
            finish_reason=finish,
            usage=self.usage or Usage(),
            logprobs=tuple(self.logprob_seq) if self.logprob_seq else None,
            provider_data=self.provider_data,
        )


class ResponseStream:
    """Lazy stream-backed response assembler.

        rs = ResponseStream(lm.stream(request), request)
        for text in rs:              # text as it arrives
            print(text, end="")
        rs.response                  # the same Response complete() returns

    ``rs.events()`` yields the canonical StreamEvents instead — one
    vocabulary for raw and assembled streaming.  Accessors mirror
    Response's own minimal set; everything richer is
    ``rs.response.message.first(...)`` / ``.parts_of(...)``.

    Tool calls are surfaced as data only; ResponseStream never executes
    anything.
    """

    def __init__(self, events: Iterator[StreamEvent], request: Request) -> None:
        self._accumulator = StreamAccumulator(request)
        self._source = events
        self._response: Response | None = None
        self._failure: Exception | None = None
        self._done = False
        self._event_iter = self._pump()

    def __iter__(self) -> Iterator[str]:
        for event in self.events():
            if event.type == "delta" and event.delta is not None:
                if event.delta.type == "text" and event.delta.text is not None:
                    yield event.delta.text

    def events(self) -> Iterator[StreamEvent]:
        """Canonical stream events, teed through the accumulator."""
        while True:
            try:
                yield next(self._event_iter)
            except StopIteration:
                return

    @property
    def text(self) -> str | None:
        return self.response.text

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return self.response.tool_calls

    @property
    def citations(self) -> list[CitationPart]:
        return self.response.citations

    @property
    def usage(self) -> Usage:
        return self.response.usage

    @property
    def finish_reason(self) -> str:
        return self.response.finish_reason

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def json(self) -> Any:
        return self.response.json

    @property
    def response(self) -> Response:
        if self._failure is not None:
            raise self._failure
        if not self._done:
            for _ in self.events():
                pass
            if self._failure is not None:  # pragma: no cover - pump raises first
                raise self._failure
        assert self._response is not None
        return self._response

    def _pump(self) -> Iterator[StreamEvent]:
        try:
            for event in self._source:
                if event.type == "error":
                    self._failure = _exception_from_error(event)
                    raise self._failure
                self._accumulator.push(event)
                yield event
                if event.type == "end":
                    break
            self._response = self._accumulator.response()
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            self._done = True


class AsyncResponseStream:
    """Async mirror of :class:`ResponseStream`, same accumulator engine.

        rs = AsyncResponseStream(lm.stream(request), request)
        async for text in rs:
            print(text, end="")
        response = await rs.response()

    ``response()`` is a method (it may need to consume the stream, which
    is an awaitable operation in async code).
    """

    def __init__(self, events: AsyncIterator[StreamEvent], request: Request) -> None:
        self._accumulator = StreamAccumulator(request)
        self._source = events
        self._response: Response | None = None
        self._failure: Exception | None = None
        self._done = False
        self._event_gen = self._pump()

    def __aiter__(self) -> AsyncIterator[str]:
        return self._text_iter()

    async def _text_iter(self) -> AsyncIterator[str]:
        async for event in self.events():
            if event.type == "delta" and event.delta is not None:
                if event.delta.type == "text" and event.delta.text is not None:
                    yield event.delta.text

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Canonical stream events, teed through the accumulator."""
        async for event in self._event_gen:
            yield event

    async def response(self) -> Response:
        if self._failure is not None:
            raise self._failure
        if not self._done:
            async for _ in self.events():
                pass
            if self._failure is not None:  # pragma: no cover - pump raises first
                raise self._failure
        assert self._response is not None
        return self._response

    async def _pump(self) -> AsyncIterator[StreamEvent]:
        try:
            async for event in self._source:
                if event.type == "error":
                    self._failure = _exception_from_error(event)
                    raise self._failure
                self._accumulator.push(event)
                yield event
                if event.type == "end":
                    break
            self._response = self._accumulator.response()
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            self._done = True


# ─── One-shot materialization ────────────────────────────────────────

def materialize_response(events: Iterator[StreamEvent], request: Request) -> Response:
    """Consume stream events and build a complete Response."""
    accumulator = StreamAccumulator(request)
    for event in events:
        if event.type == "error":
            raise _exception_from_error(event)
        accumulator.push(event)
        if event.type == "end":
            break
    return accumulator.response()


async def amaterialize_response(events: AsyncIterator[StreamEvent], request: Request) -> Response:
    """Async mirror of :func:`materialize_response`."""
    accumulator = StreamAccumulator(request)
    async for event in events:
        if event.type == "error":
            raise _exception_from_error(event)
        accumulator.push(event)
        if event.type == "end":
            break
    return accumulator.response()


# ─── Conversion utilities ────────────────────────────────────────────

def response_to_events(response: Response) -> Iterator[StreamEvent]:
    """Convert a complete Response to stream events.

    The conversion is intentionally lossless for the Delta vocabulary.  If a
    response contains a valid Part that has no Delta representation, this
    function raises instead of silently dropping content.
    """
    yield StreamStartEvent(id=response.id, model=response.model)
    # Response.logprobs is message-level; the delta vocabulary carries
    # logprobs on text deltas.  Emitting the whole sequence on the first
    # text delta makes Response -> events -> Response lossless.
    pending_logprobs = response.logprobs or ()
    for idx, part in enumerate(response.message.parts):
        if isinstance(part, TextPart):
            yield StreamDeltaEvent(delta=TextDelta(text=part.text, part_index=idx, logprobs=pending_logprobs))
            pending_logprobs = ()
        elif isinstance(part, ThinkingPart):
            yield StreamDeltaEvent(delta=ThinkingDelta(text=part.text, part_index=idx))
        elif isinstance(part, ToolCallPart):
            yield StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=json.dumps(part.input),
                    part_index=idx,
                    id=part.id,
                    name=part.name,
                )
            )
        elif isinstance(part, ImagePart):
            yield StreamDeltaEvent(
                delta=ImageDelta(
                    part_index=idx,
                    data=part.data,
                    url=part.url,
                    file_id=part.file_id,
                    media_type=part.media_type,
                )
            )
        elif isinstance(part, AudioPart):
            if part.data is None:
                _raise_non_streamable_part(
                    part,
                    reason="AudioDelta only supports inline data",
                )
            yield StreamDeltaEvent(
                delta=AudioDelta(
                    data=part.data,
                    part_index=idx,
                    media_type=part.media_type,
                )
            )
        elif isinstance(part, CitationPart):
            yield StreamDeltaEvent(
                delta=CitationDelta(
                    text=part.text,
                    url=part.url,
                    title=part.title,
                    part_index=idx,
                )
            )
        else:
            _raise_non_streamable_part(part)
        for state in part.continuation:
            yield StreamDeltaEvent(
                delta=ContinuationDelta(
                    provider=state.provider,
                    kind=state.kind,
                    data=state.data,
                    part_index=idx,
                )
            )
    for state in response.message.continuation:
        yield StreamDeltaEvent(
            delta=ContinuationDelta(
                provider=state.provider,
                kind=state.kind,
                data=state.data,
                part_index=None,
            )
        )
    yield StreamEndEvent(
        finish_reason=response.finish_reason,
        usage=response.usage,
        provider_data=response.provider_data,
    )


class _EndProviderData:
    """MAP-3 / D9: which adapter end event's ``provider_data`` the merged
    end carries.  Rank 2: a frame that supplied usage.  Rank 1: a frame that
    supplied finish_reason.  Rank 0: anything else that carried data.  A
    later frame replaces an earlier one only at the same or a higher rank,
    so the usage frame wins over a finish-only frame in either order."""

    __slots__ = ("value", "rank")

    def __init__(self) -> None:
        self.value: dict[str, Any] | None = None
        self.rank = -1

    def absorb(self, event: StreamEndEvent) -> None:
        if event.provider_data is None:
            return
        rank = 2 if event.usage is not None else 1 if event.finish_reason is not None else 0
        if rank >= self.rank:
            self.value = event.provider_data
            self.rank = rank


def coalesce_stream(
    events: Iterator[StreamEvent], *, model: str | None = None
) -> Iterator[StreamEvent]:
    """Enforce MAP-3 and MAP-4: one final StreamEndEvent, one leading StreamStartEvent.

    Adapters are stateless and may emit one end event per provider terminal
    frame (finish_reason chunk, usage-only chunk, ``[DONE]``,
    ``message_delta`` + ``message_stop``).  This wrapper passes delta and
    error events through unchanged, absorbs every end event's fields —
    a later non-None field replaces the accumulated value, a None field never
    erases one — and emits the single merged end event once the underlying
    iterator is exhausted.  If no end event was seen (e.g. the stream errored
    or was truncated), no end event is fabricated.  ``provider_data`` follows
    D9 (2026-09-06): the merged end carries the frame that supplied usage,
    else the frame that supplied finish_reason; a later usage-bearing frame
    replaces an earlier one, a finish-only frame never displaces a usage
    frame.

    Dialects without a start frame (chat completions, gemini SSE) get a
    synthesized ``StreamStartEvent`` before the first delta or end event, so
    every successful stream reads start → deltas → end (MAP-4).  A provider
    start passes through; duplicates after the first are dropped.  Error
    events never force a start: a stream that fails to open has no start.

    See docs/mapping-rules.md MAP-3 and MAP-4.
    """
    started = False
    saw_end = False
    finish_reason = None
    usage: Usage | None = None
    end_data = _EndProviderData()
    for event in events:
        if event.type == "start":
            if started:
                continue
            started = True
            yield event
            continue
        if event.type == "end":
            saw_end = True
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
            if event.usage is not None:
                usage = event.usage
            end_data.absorb(event)
            continue
        if not started and event.type == "delta":
            started = True
            yield StreamStartEvent(model=model)
        yield event
    if saw_end:
        if not started:
            yield StreamStartEvent(model=model)
        yield StreamEndEvent(
            finish_reason=finish_reason,
            usage=usage,
            provider_data=end_data.value,
        )


async def acoalesce_stream(
    events: "AsyncIterator[StreamEvent]", *, model: str | None = None
) -> "AsyncIterator[StreamEvent]":
    """Async mirror of :func:`coalesce_stream` — same MAP-3/MAP-4 semantics.

    Passes delta/error events through unchanged, absorbs every end
    event's fields (later non-None replaces, None never erases), and emits
    exactly one merged final StreamEndEvent once the source is exhausted.
    No end event is fabricated if none was seen.  Synthesizes one leading
    StreamStartEvent for dialects without a start frame; duplicate starts
    are dropped; error events never force a start.
    """
    started = False
    saw_end = False
    finish_reason = None
    usage: Usage | None = None
    end_data = _EndProviderData()
    async for event in events:
        if event.type == "start":
            if started:
                continue
            started = True
            yield event
            continue
        if event.type == "end":
            saw_end = True
            if event.finish_reason is not None:
                finish_reason = event.finish_reason
            if event.usage is not None:
                usage = event.usage
            end_data.absorb(event)
            continue
        if not started and event.type == "delta":
            started = True
            yield StreamStartEvent(model=model)
        yield event
    if saw_end:
        if not started:
            yield StreamStartEvent(model=model)
        yield StreamEndEvent(
            finish_reason=finish_reason,
            usage=usage,
            provider_data=end_data.value,
        )


# ─── Internal helpers ────────────────────────────────────────────────

def _raise_non_streamable_part(part: Part, *, reason: str | None = None) -> None:
    detail = reason or f"no {part.type!r} Delta variant exists"
    raise TypeError(f"Cannot convert {type(part).__name__} to StreamEvent: {detail}")


def _exception_from_error(event: StreamEvent) -> Exception:
    err = event.error or ErrorDetail(code="provider", message="stream error")
    code = err.code
    message = err.message
    exc_cls = error_class_for_code(code)
    if issubclass(exc_cls, LM15Error):
        return exc_cls(message, provider_code=err.provider_code)
    return exc_cls(message)


def _concat_b64_chunks(chunks: list[str]) -> bytes:
    """Decode each base64 chunk and concatenate raw bytes."""
    import base64
    raw = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        try:
            raw.extend(base64.b64decode(chunk))
        except Exception:
            padded = chunk + "=" * (-len(chunk) % 4)
            try:
                raw.extend(base64.b64decode(padded))
            except Exception:
                pass
    return bytes(raw)


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    import struct
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
        channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm


def _parse_json_best_effort(raw: str | None) -> JsonObject:
    if not raw:
        return {}
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
        return value if isinstance(value, dict) else {"value": value}
    except Exception:
        return {"partial_json": raw}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")
