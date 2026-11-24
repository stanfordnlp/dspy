import functools
from contextvars import ContextVar
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

ACTIVE_CALL_ID: ContextVar[str | None] = ContextVar("active_call_id", default=None)
_ACTIVE_COMPILES = ContextVar("active_compiles", default=())


def _bind_active_call_id(fn: Callable[P, T]) -> Callable[P, T]:
    """Bind the current callback call ID to each synchronous invocation of ``fn``."""
    parent_call_id = ACTIVE_CALL_ID.get()

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        token = ACTIVE_CALL_ID.set(parent_call_id)
        try:
            return fn(*args, **kwargs)
        finally:
            ACTIVE_CALL_ID.reset(token)

    return wrapper
