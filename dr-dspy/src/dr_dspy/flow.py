"""Flow labels for experiment event logs."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from dr_dspy.event_log import DEFAULT_FLOW, EventWriter

_current_flow: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dspy_flow", default=DEFAULT_FLOW
)

__all__ = ["current_flow", "event_flow"]


def current_flow() -> str:
    """Return the active event-log flow label."""
    return _current_flow.get()


@contextlib.contextmanager
def event_flow(writer: EventWriter | None, flow: str) -> Iterator[None]:
    """Set the active flow label and emit flow boundary events."""
    token = _current_flow.set(flow)
    if writer is not None:
        writer.put_event("flow.start", payload={"flow": flow}, flow=flow)
    try:
        yield
    finally:
        if writer is not None:
            writer.put_event("flow.end", payload={"flow": flow}, flow=flow)
        _current_flow.reset(token)
