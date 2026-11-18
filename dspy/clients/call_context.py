"""Task-local progress for retry/fallback policy, never part of a model Request."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class StreamProgress:
    emitted: bool = False


_adapter_calls = ContextVar("dspy_adapter_stream_progress", default=())


@contextmanager
def adapter_fallback_scope():
    progress = StreamProgress()
    token = _adapter_calls.set((*_adapter_calls.get(), progress))
    try:
        yield progress
    finally:
        _adapter_calls.reset(token)


def stream_emitted():
    # A nested extraction call's visible output also prevents its parent from
    # replaying a program segment. Independent concurrent calls have own scopes.
    for progress in _adapter_calls.get():
        progress.emitted = True
