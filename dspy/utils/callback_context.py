import functools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)
_ACTIVE_CALL = ContextVar("active_call", default=None)


class _ActiveCall:
    __slots__ = ("call_id", "is_active", "is_optimizer", "parent", "parent_call_id")

    def __init__(
        self,
        call_id: str,
        parent_call_id: str | None,
        parent: "_ActiveCall | None",
        *,
        is_optimizer: bool,
    ):
        self.call_id = call_id
        self.parent_call_id = parent_call_id
        self.parent = parent
        self.is_optimizer = is_optimizer
        self.is_active = True


def _resolve_active_call(
    call_id: str | None, active_call: _ActiveCall | None
) -> tuple[str | None, _ActiveCall | None]:
    # IDs set directly through ACTIVE_CALL_ID have no lifecycle metadata to resolve.
    if active_call is None or active_call.call_id != call_id:
        return call_id, None

    while active_call is not None and not active_call.is_active:
        call_id = active_call.parent_call_id
        active_call = active_call.parent

    return call_id, active_call


def _normalize_active_call_context() -> str | None:
    """Discard completed call scopes copied into detached tasks or threads."""
    call_id = ACTIVE_CALL_ID.get()
    active_call = _ACTIVE_CALL.get()
    live_call_id, live_active_call = _resolve_active_call(call_id, active_call)

    if live_call_id != call_id:
        ACTIVE_CALL_ID.set(live_call_id)
    if live_active_call is not active_call:
        _ACTIVE_CALL.set(live_active_call)

    return live_call_id


def _is_active_optimizer_call() -> bool:
    call_id = ACTIVE_CALL_ID.get()
    active_call = _ACTIVE_CALL.get()
    return (
        active_call is not None
        and active_call.call_id == call_id
        and active_call.is_active
        and active_call.is_optimizer
    )


@contextmanager
def _active_call_context(call_id: str, *, is_optimizer: bool = False):
    parent_call_id = _normalize_active_call_context()
    parent = _ACTIVE_CALL.get()
    active_call = _ActiveCall(call_id, parent_call_id, parent, is_optimizer=is_optimizer)
    call_id_token = ACTIVE_CALL_ID.set(call_id)
    active_call_token = _ACTIVE_CALL.set(active_call)

    try:
        yield
    finally:
        active_call.is_active = False
        ACTIVE_CALL_ID.reset(call_id_token)
        _ACTIVE_CALL.reset(active_call_token)
        _normalize_active_call_context()


def _bind_active_call_id(fn: Callable) -> Callable:
    """Bind the current callback call ID to each synchronous invocation of ``fn``."""
    parent_call_id = _normalize_active_call_context()
    parent = _ACTIVE_CALL.get()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        live_call_id, live_active_call = _resolve_active_call(parent_call_id, parent)
        call_id_token = ACTIVE_CALL_ID.set(live_call_id)
        active_call_token = _ACTIVE_CALL.set(live_active_call)
        try:
            return fn(*args, **kwargs)
        finally:
            ACTIVE_CALL_ID.reset(call_id_token)
            _ACTIVE_CALL.reset(active_call_token)

    return wrapper
