import functools
from contextvars import ContextVar
from typing import Callable

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)


def _bind_active_call_id(fn: Callable) -> Callable:
    """Bind the current callback call ID to each synchronous invocation of ``fn``."""
    parent_call_id = ACTIVE_CALL_ID.get()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = ACTIVE_CALL_ID.set(parent_call_id)
        try:
            return fn(*args, **kwargs)
        finally:
            ACTIVE_CALL_ID.reset(token)

    return wrapper
