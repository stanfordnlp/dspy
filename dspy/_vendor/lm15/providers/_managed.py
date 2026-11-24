"""Managed provider operation preparation; execution adapters never share auth state.

Decorated drivers acquire one snapshot before encoding any requests. Nested
drivers run on the prepared adapter, retaining that snapshot across every wire
call. Pure mapping, planning, and job-handle constructors are not operations.
"""
from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from ..errors import AuthOperationError, UnsupportedFeatureError

_F = TypeVar("_F", bound=Callable[..., Any])


def _prepare(lm):
    prepare = getattr(lm, "_managed_prepare", None)
    return prepare() if prepare is not None else lm


def operation(method: _F) -> _F:
    if inspect.iscoroutinefunction(method):
        @wraps(method)
        async def call(self, *args, **kwargs):
            lm = await asyncio.to_thread(_prepare, self) if getattr(self, "_managed_prepare", None) else self
            return await method(lm, *args, **kwargs)
    elif inspect.isgeneratorfunction(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            yield from method(_prepare(self), *args, **kwargs)
    else:
        @wraps(method)
        def call(self, *args, **kwargs):
            return method(_prepare(self), *args, **kwargs)
    return cast(_F, call)


def async_stream(method: _F) -> _F:
    @wraps(method)
    async def call(self, *args, **kwargs):
        lm = await asyncio.to_thread(_prepare, self) if getattr(self, "_managed_prepare", None) else self
        events = method(lm, *args, **kwargs)
        try:
            async for event in events:
                yield event
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
    return cast(_F, call)


def require_prepared(lm):
    if getattr(lm, "_managed_prepare", None) is not None:
        raise AuthOperationError(
            "managed requests must be built through an operation driver",
            reason="indeterminate", stage="dispatch", commit_state="not_committed",
        )


def admit(lm, request):
    require_prepared(lm)
    expected = getattr(lm, "_managed_admit", None)
    if expected is not None and request._admit is not expected:
        raise AuthOperationError(
            "managed request has no matching authentication snapshot",
            reason="connection_changed", stage="dispatch", commit_state="not_committed",
        )
    if request._admit is not None:
        request._admit()


def require_unmanaged_live(lm):
    if getattr(lm, "_managed_prepare", None) is not None or getattr(lm, "_managed_admit", None) is not None:
        raise UnsupportedFeatureError("managed authentication is not supported on websocket transports")
