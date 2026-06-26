"""DSPy callbacks that emit experiment telemetry events."""

from __future__ import annotations

import contextvars
import sys
from typing import Any

from dr_dspy.event_log import EventWriter
from dr_dspy.serialization import to_jsonable
from dspy.utils.callback import BaseCallback

_current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dspy_call_id", default=None
)


class EventLogCallback(BaseCallback):
    """BaseCallback that writes every DSPy hook to an EventWriter."""

    def __init__(self, writer: EventWriter) -> None:
        super().__init__()
        self._writer = writer
        self._tokens: dict[str, contextvars.Token[str | None]] = {}

    def _safe_log_start(
        self, event_type: str, call_id: str, payload: dict[str, Any]
    ) -> None:
        try:
            parent = _current_call_id.get()
            token = _current_call_id.set(call_id)
            self._tokens[call_id] = token
            self._writer.put_event(
                event_type,
                payload=payload,
                call_id=call_id,
                parent_call_id=parent,
            )
        except Exception as e:
            print(f"[EventLogCallback {event_type}] {e!r}", file=sys.stderr)

    def _safe_log_end(
        self,
        event_type: str,
        call_id: str,
        outputs: Any,
        exception: BaseException | None,
    ) -> None:
        try:
            token = self._tokens.pop(call_id, None)
            payload = {"outputs": to_jsonable(outputs)}
            err = None
            if exception is not None:
                payload["exception"] = repr(exception)
                err = repr(exception)
            self._writer.put_event(
                event_type, payload=payload, call_id=call_id, error=err
            )
            if token is not None:
                _current_call_id.reset(token)
        except Exception as e:
            print(f"[EventLogCallback {event_type}] {e!r}", file=sys.stderr)

    def on_module_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "module.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_module_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("module.end", call_id, outputs, exception)

    def on_lm_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "lm.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("lm.end", call_id, outputs, exception)

    def on_adapter_format_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "adapter.format.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("adapter.format.end", call_id, outputs, exception)

    def on_adapter_parse_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "adapter.parse.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("adapter.parse.end", call_id, outputs, exception)

    def on_tool_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "tool.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_tool_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("tool.end", call_id, outputs, exception)

    def on_evaluate_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        self._safe_log_start(
            "evaluate.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any,
        exception: BaseException | None = None,
    ) -> None:
        self._safe_log_end("evaluate.end", call_id, outputs, exception)
