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

from .errors import TransportError
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
# Their shared behavior is specified in the contract's
# 2026-09-11-shared-handles-and-profile-migration decision. They remain
# provisional collectors, not new canonical wire types.

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
            if audio_media_type is not None and event.media_type is not None and audio_media_type != event.media_type:
                raise ValueError("a Turn cannot concatenate different audio media types; consume raw events")
            audio.extend(base64.b64decode(event.data, validate=True))
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
    ended_by = events[-1].type if events and events[-1].type in (_TURN_TERMINAL | {"tool_call"}) else "incomplete"
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


class _TurnState:
    def __init__(self, session: Any) -> None:
        self._session = session
        self._done = False
        self._events: list[LiveServerEvent] = []
        self._failure: Exception | None = None
        self._result: Turn | None = None
        self._reading = False

    def snapshot(self) -> Turn:
        """Collected data so far; incomplete is not a successful turn."""
        return _materialize_turn(tuple(self._events))

    def close(self) -> None:
        """Stop this view, not the underlying live session."""
        if self._reading:
            raise RuntimeError("stop the active turn reader before closing its view")
        self._done = True

    def _accept(self, event: LiveServerEvent | None) -> LiveServerEvent:
        if event is None:
            raise TransportError("live session closed before the turn reached a boundary")
        self._events.append(event)
        if event.type in _TURN_TERMINAL:
            self._done = True
        return event

    def _seal(self) -> Turn:
        if self._failure is not None:
            raise self._failure
        result = self.snapshot()
        if result.ended_by == "incomplete":
            raise TransportError("turn view closed before the turn reached a boundary; inspect snapshot()")
        self._done = True
        self._result = result
        return result


class TurnView(_TurnState):
    """One buffered half-duplex view. Iteration stops on terminal events;
    result also stops at a tool call. Already-yielded events stay in the result.
    Use raw session iteration when buffering is not wanted.
    """

    def __iter__(self):
        return self

    def __next__(self) -> LiveServerEvent:
        if self._reading:
            raise RuntimeError("turn view already has an active reader")
        if self._failure is not None:
            raise self._failure
        if self._done:
            raise StopIteration
        self._reading = True
        try:
            return self._accept(self._session.recv())
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            self._reading = False

    def result(self) -> Turn:
        if self._result is not None:
            return self._result
        # A tool call just yielded by manual iteration must not be read past
        # by result(): the application may still owe the model an answer.
        if not self._events or self._events[-1].type != "tool_call":
            for event in self:
                if event.type == "tool_call":
                    break
        return self._seal()


class AsyncTurnView(_TurnState):
    """Native async twin; task cancellation propagates without inventing a turn."""

    def __aiter__(self):
        return self

    async def __anext__(self) -> LiveServerEvent:
        if self._reading:
            raise RuntimeError("turn view already has an active reader")
        if self._failure is not None:
            raise self._failure
        if self._done:
            raise StopAsyncIteration
        self._reading = True
        try:
            return self._accept(await self._session.recv())
        except Exception as exc:
            self._failure = exc
            raise
        finally:
            self._reading = False

    async def result(self) -> Turn:
        if self._result is not None:
            return self._result
        if not self._events or self._events[-1].type != "tool_call":
            async for event in self:
                if event.type == "tool_call":
                    break
        return self._seal()


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
