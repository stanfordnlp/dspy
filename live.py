"""
lm15.live — WebSocket live session wrapper.

Provider-agnostic session around a WebSocket connection for realtime
(live) interactions with foundation models.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Deque

from .types import (
    ErrorDetail,
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientImageEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveServerEvent,
    PART_CLASSES,
    Part,
    PartInput,
    TextPart,
    ToolCallInfo,
    Usage,
    _normalize_parts,
)

EncodeEventFn = Callable[[LiveClientEvent], list[dict[str, Any]]]
DecodeEventFn = Callable[[str | bytes], list[LiveServerEvent]]


def require_websocket_sync_connect():
    """Return `websockets.sync.client.connect` or raise a helpful ImportError."""
    try:
        from websockets.sync.client import connect  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Live sessions require the optional 'websockets' dependency.\n\n"
            "  Install it with:\n"
            "    pip install lm15[live]\n"
        ) from exc
    return connect


def require_websocket_async_connect():
    """Return `websockets.asyncio.client.connect` or raise a helpful ImportError."""
    try:
        from websockets.asyncio.client import connect  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Live sessions require the optional 'websockets' dependency.\n\n"
            "  Install it with:\n"
            "    pip install lm15[live]\n"
        ) from exc
    return connect


# ─── Turn: half-duplex ergonomics over the event stream ─────────────
#
# A live session is FULL-duplex: with voice-activity detection the model
# can speak spontaneously and turns can overlap after interruptions.
# Plain session iteration is the primary surface for that. `turn()` and
# `Turn` serve the half-duplex idiom (send, then listen until the turn
# ends) — the shape every scripted recipe and turn-based voice app has.
# They are per-language ergonomics like BatchJob, not canonical wire
# types: ports choose their own idiom; the transcript harness pins the
# event stream, not this sugar.

_TURN_TERMINAL = frozenset({"turn_end", "interrupted", "error"})


@dataclass(frozen=True)
class Turn:
    """One materialized turn.

    ``ended_by`` is one of ``turn_end`` / ``interrupted`` / ``error`` /
    ``tool_call``. ``usage`` is the field-wise sum of every ``usage`` and
    ``turn_end`` event the turn saw: a tool-call response's tokens arrive
    as a ``usage`` event at the start of the continuation turn, and a
    cancelled response's tokens precede its ``interrupted``. A ``tool_call`` ending mirrors the non-live
    ``finish_reason="tool_call"`` contract: the model is waiting for
    YOUR result — answer with ``send_tool_result()`` and materialize the
    next turn. Materializing buffers text and audio in memory until the
    turn ends; for latency-sensitive playback iterate events instead.
    """

    ended_by: str
    text: str = ""
    audio: bytes = b""
    audio_media_type: str | None = None
    tool_calls: tuple[ToolCallInfo, ...] = ()
    usage: "Usage | None" = None
    error: "ErrorDetail | None" = None
    events: tuple[LiveServerEvent, ...] = ()

    @property
    def ok(self) -> bool:
        return self.ended_by == "turn_end"


def _sum_usage(acc: "Usage | None", more: Usage) -> Usage:
    """Field-wise sum of two Usage values from one session (same provider,
    same taxonomy). INV-029: a counter absent on either side is unknown
    in the sum, never zero."""
    if acc is None:
        return more
    fields = ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens",
              "reasoning_tokens", "input_audio_tokens", "output_audio_tokens")
    values = {}
    for name in fields:
        a, b = getattr(acc, name), getattr(more, name)
        values[name] = a + b if a is not None and b is not None else None
    return Usage(**values)


def _materialize_turn(events: tuple[LiveServerEvent, ...]) -> Turn:
    text_parts: list[str] = []
    audio = bytearray()
    audio_media_type: str | None = None
    tool_calls: list[ToolCallInfo] = []
    usage: Usage | None = None
    error: ErrorDetail | None = None
    for event in events:
        if event.type == "text":
            text_parts.append(event.text)
        elif event.type == "audio":
            audio.extend(base64.b64decode(event.data))
            if audio_media_type is None and event.media_type is not None:
                audio_media_type = event.media_type
        elif event.type == "tool_call":
            tool_calls.append(ToolCallInfo(id=event.id, name=event.name, input=event.input))
        elif event.type in ("turn_end", "usage"):
            # A turn's bill is every usage-bearing event it saw: the
            # usage event of a tool-call response (which arrives after the
            # tool_call that ended the previous result(), i.e. at the start
            # of the continuation turn — the semantic turn stayed open) and
            # the usage of a cancelled response before its interrupted.
            usage = _sum_usage(usage, event.usage)
        elif event.type == "error":
            error = event.error
    ended_by = events[-1].type if events and events[-1].type in (_TURN_TERMINAL | {"tool_call"}) else "error"
    return Turn(
        ended_by=ended_by,
        text="".join(text_parts),
        audio=bytes(audio),
        audio_media_type=audio_media_type,
        tool_calls=tuple(tool_calls),
        usage=usage,
        error=error,
        events=events,
    )


class TurnView:
    """Iterator over one turn's server events.

    Ends itself after yielding the terminal event (``turn_end`` /
    ``interrupted`` / ``error``) — the same self-ending idiom as
    ``stream()``. Tool calls are yielded mid-iteration (you hold the
    session, so you can answer and keep iterating); ``result()`` cannot
    answer for you, so it returns at a ``tool_call`` instead of
    deadlocking against a model that is waiting for your result.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._done = False

    def __iter__(self):
        return self

    def __next__(self) -> LiveServerEvent:
        if self._done:
            raise StopIteration
        event = self._session.recv()
        if event.type in _TURN_TERMINAL:
            self._done = True
        return event

    def result(self) -> Turn:
        events: list[LiveServerEvent] = []
        for event in self:
            events.append(event)
            if event.type == "tool_call":
                break
        return _materialize_turn(tuple(events))


class AsyncTurnView:
    """Async twin of :class:`TurnView`."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> LiveServerEvent:
        if self._done:
            raise StopAsyncIteration
        event = await self._session.recv()
        if event.type in _TURN_TERMINAL:
            self._done = True
        return event

    async def result(self) -> Turn:
        events: list[LiveServerEvent] = []
        async for event in self:
            events.append(event)
            if event.type == "tool_call":
                break
        return _materialize_turn(tuple(events))


class WebSocketLiveSession:
    """Provider-agnostic session wrapper around a WebSocket connection."""

    def __init__(
        self,
        *,
        ws: Any,
        encode_event: EncodeEventFn,
        decode_event: DecodeEventFn,
    ) -> None:
        self._ws = ws
        self._encode_event = encode_event
        self._decode_event = decode_event
        self._pending: Deque[LiveServerEvent] = deque()
        self._send_lock = threading.Lock()
        self._closed = False

    def send(
        self,
        event: LiveClientEvent | None = None,
        *,
        audio: bytes | str | None = None,
        audio_media_type: str = "audio/pcm;rate=16000",
        image: bytes | str | None = None,
        image_media_type: str = "image/jpeg",
        text: str | None = None,
        turn: PartInput | None = None,
        tool_result: dict[str, Any] | None = None,
        interrupt: bool = False,
        end_audio: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("live session is closed")

        if event is not None:
            has_payload = any(x is not None for x in (audio, image, text, turn, tool_result))
            if has_payload or interrupt or end_audio:
                raise ValueError("pass either `event` or keyword payload, not both")
            events = [event]
        else:
            events = _events_from_kwargs(
                audio=audio,
                audio_media_type=audio_media_type,
                image=image,
                image_media_type=image_media_type,
                text=text,
                turn=turn,
                tool_result=tool_result,
                interrupt=interrupt,
                end_audio=end_audio,
            )

        with self._send_lock:
            for evt in events:
                payloads = self._encode_event(evt)
                for payload in payloads:
                    self._ws.send(json.dumps(payload))

    def send_turn(self, content: PartInput, *, turn_complete: bool = True) -> None:
        self.send(LiveClientTurnEvent(parts=_normalize_parts(content), turn_complete=turn_complete))

    def send_audio(self, data: bytes | str, *, media_type: str = "audio/pcm;rate=16000") -> None:
        self.send(LiveClientAudioEvent(data=_to_base64_str(data), media_type=media_type))

    def send_image(self, data: bytes | str, *, media_type: str = "image/jpeg") -> None:
        self.send(LiveClientImageEvent(data=_to_base64_str(data), media_type=media_type))

    def send_text(self, text: str) -> None:
        self.send(LiveClientTextEvent(text=text))

    def send_tool_result(self, results: dict[str, Any]) -> None:
        self.send(tool_result=results)

    def interrupt(self) -> None:
        self.send(interrupt=True)

    def end_audio(self) -> None:
        self.send(end_audio=True)

    def recv(self) -> LiveServerEvent:
        if self._closed:
            raise RuntimeError("live session is closed")

        while True:
            if self._pending:
                return self._pending.popleft()

            raw = self._ws.recv()
            decoded = self._decode_event(raw)
            if not decoded:
                continue

            for event in decoded:
                self._pending.append(event)

    def turn(self) -> TurnView:
        """Iterate one turn; see :class:`TurnView` and :class:`Turn`."""
        return TurnView(self)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            return

    def __iter__(self):
        return self

    def __next__(self) -> LiveServerEvent:
        if self._closed:
            raise StopIteration
        try:
            return self.recv()
        except RuntimeError:
            raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _events_from_kwargs(
    *,
    audio: bytes | str | None,
    audio_media_type: str,
    image: bytes | str | None,
    image_media_type: str,
    text: str | None,
    turn: PartInput | None,
    tool_result: dict[str, Any] | None,
    interrupt: bool,
    end_audio: bool,
) -> list[LiveClientEvent]:
    events: list[LiveClientEvent] = []

    if audio is not None:
        events.append(LiveClientAudioEvent(data=_to_base64_str(audio), media_type=audio_media_type))
    if image is not None:
        events.append(LiveClientImageEvent(data=_to_base64_str(image), media_type=image_media_type))
    if turn is not None:
        events.append(LiveClientTurnEvent(parts=_normalize_parts(turn)))
    if text is not None:
        events.append(LiveClientTextEvent(text=text))

    if tool_result:
        for call_id, value in tool_result.items():
            content = tuple(_tool_result_parts(value))
            events.append(LiveClientToolResultEvent(id=call_id, content=content))

    if interrupt:
        events.append(LiveClientInterruptEvent())
    if end_audio:
        events.append(LiveClientEndAudioEvent())

    if not events:
        raise ValueError("nothing to send")
    return events

class AsyncWebSocketLiveSession:
    """Native async live session over `websockets.asyncio`.

    NOT a thread wrapper: a blocked sync ``recv()`` inside a worker
    thread cannot be cancelled from the event loop, and cancellation
    (barge-in, hangup) is the heart of realtime. Here ``recv()`` is a
    real awaitable — cancelling the task cancels the read. The pure
    encode/decode codecs are shared with the sync session verbatim.
    """

    def __init__(
        self,
        *,
        ws: Any,
        encode_event: EncodeEventFn,
        decode_event: DecodeEventFn,
    ) -> None:
        self._ws = ws
        self._encode_event = encode_event
        self._decode_event = decode_event
        self._pending: Deque[LiveServerEvent] = deque()
        self._send_lock = asyncio.Lock()
        self._closed = False

    async def send(
        self,
        event: LiveClientEvent | None = None,
        *,
        audio: bytes | str | None = None,
        audio_media_type: str = "audio/pcm;rate=16000",
        image: bytes | str | None = None,
        image_media_type: str = "image/jpeg",
        text: str | None = None,
        turn: PartInput | None = None,
        tool_result: dict[str, Any] | None = None,
        interrupt: bool = False,
        end_audio: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("live session is closed")

        if event is not None:
            has_payload = any(x is not None for x in (audio, image, text, turn, tool_result))
            if has_payload or interrupt or end_audio:
                raise ValueError("pass either `event` or keyword payload, not both")
            events = [event]
        else:
            events = _events_from_kwargs(
                audio=audio,
                audio_media_type=audio_media_type,
                image=image,
                image_media_type=image_media_type,
                text=text,
                turn=turn,
                tool_result=tool_result,
                interrupt=interrupt,
                end_audio=end_audio,
            )

        async with self._send_lock:
            for evt in events:
                for payload in self._encode_event(evt):
                    await self._ws.send(json.dumps(payload))

    async def send_turn(self, content: PartInput, *, turn_complete: bool = True) -> None:
        await self.send(LiveClientTurnEvent(parts=_normalize_parts(content), turn_complete=turn_complete))

    async def send_audio(self, data: bytes | str, *, media_type: str = "audio/pcm;rate=16000") -> None:
        await self.send(LiveClientAudioEvent(data=_to_base64_str(data), media_type=media_type))

    async def send_image(self, data: bytes | str, *, media_type: str = "image/jpeg") -> None:
        await self.send(LiveClientImageEvent(data=_to_base64_str(data), media_type=media_type))

    async def send_text(self, text: str) -> None:
        await self.send(LiveClientTextEvent(text=text))

    async def send_tool_result(self, results: dict[str, Any]) -> None:
        await self.send(tool_result=results)

    async def interrupt(self) -> None:
        await self.send(interrupt=True)

    async def end_audio(self) -> None:
        await self.send(end_audio=True)

    async def recv(self) -> LiveServerEvent:
        if self._closed:
            raise RuntimeError("live session is closed")

        while True:
            if self._pending:
                return self._pending.popleft()

            raw = await self._ws.recv()
            decoded = self._decode_event(raw)
            if not decoded:
                continue

            for event in decoded:
                self._pending.append(event)

    def turn(self) -> AsyncTurnView:
        """Iterate one turn; see :class:`AsyncTurnView` and :class:`Turn`."""
        return AsyncTurnView(self)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._ws.close()
        except Exception:
            return

    def __aiter__(self):
        return self

    async def __anext__(self) -> LiveServerEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self.recv()
        except RuntimeError:
            raise StopAsyncIteration

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        await self.close()


def _tool_result_parts(value: Any) -> list[Part]:
    if isinstance(value, str):
        return [TextPart(text=value)]
    if isinstance(value, PART_CLASSES):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
        if all(isinstance(part, PART_CLASSES) for part in parts):
            return parts
    return [TextPart(text=str(value))]


def _to_base64_str(data: bytes | str) -> str:
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("ascii")
    return data
