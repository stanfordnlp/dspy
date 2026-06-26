"""Flow labels for experiment event logs."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Iterator

from dr_dspy.event_log import DEFAULT_FLOW, EventWriter

_current_flow: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dspy_flow", default=DEFAULT_FLOW
)
_thread_fallback_flow = DEFAULT_FLOW
_thread_fallback_flow_lock = threading.Lock()

__all__ = ["current_flow", "event_flow"]


def current_flow() -> str:
    """Return the active event-log flow label."""
    flow = _current_flow.get()
    if flow != DEFAULT_FLOW:
        return flow
    with _thread_fallback_flow_lock:
        return _thread_fallback_flow


@contextlib.contextmanager
def event_flow(writer: EventWriter | None, flow: str) -> Iterator[None]:
    """Set the active flow label and emit flow boundary events."""
    global _thread_fallback_flow
    token = _current_flow.set(flow)
    with _thread_fallback_flow_lock:
        previous_thread_fallback_flow = _thread_fallback_flow
        _thread_fallback_flow = flow
    if writer is not None:
        writer.put_event("flow.start", payload={"flow": flow}, flow=flow)
    try:
        yield
    finally:
        if writer is not None:
            writer.put_event("flow.end", payload={"flow": flow}, flow=flow)
        with _thread_fallback_flow_lock:
            _thread_fallback_flow = previous_thread_fallback_flow
        _current_flow.reset(token)
