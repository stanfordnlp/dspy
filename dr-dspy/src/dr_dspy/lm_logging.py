"""DSPy LM wrappers that emit request/response telemetry."""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

import dspy
from dr_dspy.serialization import sanitize_lm_kwargs, to_jsonable
from dspy.utils.dummies import dotdict  # type: ignore[attr-defined]

PutEventFn = Callable[..., None]
CallableLMFn = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, str]]

__all__ = [
    "CallableLM",
    "LoggingCallableLM",
    "LoggingLM",
    "wrap_provider_response",
]


def wrap_provider_response(field_values: dict[str, str]) -> Any:
    """Wrap output-field values as a ChatAdapter provider response."""
    from dspy.adapters.chat_adapter import (  # type: ignore[attr-defined]
        FieldInfoWithName,
        format_field_value,
    )
    from dspy.signatures.field import OutputField

    parts = []
    for name, value in field_values.items():
        field = FieldInfoWithName(name=name, info=OutputField())
        try:
            rendered = format_field_value(field_info=field.info, value=value)
        except Exception:
            rendered = str(value)
        parts.append(f"[[ ## {name} ## ]]\n{rendered}")
    parts.append("[[ ## completed ## ]]")
    content = "\n\n".join(parts)
    msg = dotdict(content=content, tool_calls=None)
    choice = dotdict(message=msg, finish_reason="stop")
    return dotdict(
        choices=[choice],
        usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model="mock",
    )


class _LoggingMixin:
    """Shared lm.request/lm.response/lm.error logging for DSPy LM wrappers."""

    _log: PutEventFn

    def _log_request(
        self, req_id: str, messages: Any, kwargs: dict[str, Any]
    ) -> None:
        try:
            self._log(
                "lm.request",
                payload={
                    "req_id": req_id,
                    "messages": to_jsonable(messages),
                    "kwargs": sanitize_lm_kwargs(kwargs),
                },
            )
        except Exception as e:
            print(
                f"[{type(self).__name__} log_request] {e!r}", file=sys.stderr
            )

    def _log_response(self, req_id: str, resp: Any, dt: float) -> None:
        try:
            self._log(
                "lm.response",
                payload={
                    "req_id": req_id,
                    "dt": dt,
                    "response": to_jsonable(resp),
                },
            )
        except Exception as e:
            print(
                f"[{type(self).__name__} log_response] {e!r}", file=sys.stderr
            )

    def _log_error(self, req_id: str, exc: BaseException, dt: float) -> None:
        try:
            self._log(
                "lm.error",
                payload={"req_id": req_id, "dt": dt, "error": repr(exc)},
                error=repr(exc),
            )
        except Exception as e:
            print(f"[{type(self).__name__} log_error] {e!r}", file=sys.stderr)

    def _run_logged_forward(
        self,
        forward_fn: Callable[[], Any],
        *,
        messages: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        self._log_request(req_id, messages, kwargs)
        try:
            resp = forward_fn()
        except BaseException as e:
            self._log_error(req_id, e, time.time() - t0)
            raise
        self._log_response(req_id, resp, time.time() - t0)
        return resp


class LoggingLM(_LoggingMixin, dspy.LM):
    """dspy.LM subclass that logs requests and raw responses."""

    def __init__(self, model: str, *, log: PutEventFn, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._log = log

    def forward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        return self._run_logged_forward(
            lambda: super(LoggingLM, self).forward(
                prompt=prompt, messages=messages, **kwargs
            ),
            messages=messages,
            kwargs=kwargs,
        )

    async def aforward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        self._log_request(req_id, messages, kwargs)
        try:
            resp = await super().aforward(
                prompt=prompt, messages=messages, **kwargs
            )
        except BaseException as e:
            self._log_error(req_id, e, time.time() - t0)
            raise
        self._log_response(req_id, resp, time.time() - t0)
        return resp


class CallableLM(dspy.BaseLM):
    """Programmatic LM whose responses come from a user-supplied callable."""

    forward_contract = "legacy"

    def __init__(
        self,
        fn: CallableLMFn,
        *,
        model: str = "callable/mock",
    ) -> None:
        super().__init__(model=model)
        self._fn = fn
        self.calls: list[dict[str, Any]] = []

    def _serve(self, messages: Any, kwargs: dict[str, Any]) -> Any:
        msg_list: list[dict[str, Any]] = list(messages) if messages else []
        try:
            output = self._fn(msg_list, kwargs)
        except Exception as e:
            output = {"_error": repr(e)}
        self.calls.append(
            {"messages": msg_list, "kwargs": dict(kwargs), "output": output}
        )
        return wrap_provider_response(output)

    def forward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        return self._serve(messages, kwargs)

    async def aforward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        return self._serve(messages, kwargs)


class LoggingCallableLM(_LoggingMixin, CallableLM):
    """CallableLM with lm.request/response/error logging."""

    def __init__(
        self,
        fn: CallableLMFn,
        *,
        log: PutEventFn,
        model: str = "callable/mock",
    ) -> None:
        super().__init__(fn, model=model)
        self._log = log

    def forward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        return self._run_logged_forward(
            lambda: super(LoggingCallableLM, self).forward(
                prompt=prompt, messages=messages, **kwargs
            ),
            messages=messages,
            kwargs=kwargs,
        )

    async def aforward(
        self, prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)
