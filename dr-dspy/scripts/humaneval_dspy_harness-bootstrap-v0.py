"""HumanEval+ x DSPy evaluation / optimization harness.

Single-file harness that runs Evaluate → BootstrapFewShot → Evaluate on
HumanEval+ using DSPy, logs every observable boundary to a SQLite database,
sandboxes code execution in subprocesses, and ships with deterministic
self-tests using a mock LM.

Run modes:
    python humaneval_dspy_harness.py            # real run (needs OPENAI_API_KEY)
    python humaneval_dspy_harness.py --mock     # full flow with CallableLM
    python humaneval_dspy_harness.py --test     # deterministic self-tests
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import inspect
import json
import logging
import multiprocessing as mp
import os
import platform
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from multiprocessing.connection import Connection
from typing import Any, Protocol, cast

import pydantic

import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import (
    dotdict,  # type: ignore[attr-defined]  # internal DSPy helper
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_DB_PATH = "./runs.db"
DEFAULT_COMPILED_PATH = "./compiled_humaneval.json"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TRAIN_SIZE = 32
DEFAULT_DEV_SIZE = 32
DEFAULT_NUM_THREADS = 8
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
DEFAULT_MEM_LIMIT_BYTES = 1 << 30  # 1 GiB
DEFAULT_CPU_LIMIT_SECONDS = 17
DEFAULT_SEED = 0
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 4
DEFAULT_MAX_LABELED_DEMOS = 0
PAYLOAD_MAX_BYTES = 256 * 1024
REPR_TRUNCATE = 4096
SANITIZE_KEYS = frozenset({"api_key", "api_base", "base_url", "model_list", "authorization"})

# --------------------------------------------------------------------------- #
# Context vars: current flow and call-id parent chain
# --------------------------------------------------------------------------- #

_current_flow: contextvars.ContextVar[str] = contextvars.ContextVar("dspy_flow", default="unknown")
_current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("dspy_call_id", default=None)

# Module-level writer binding so the metric (which is not a method) can log.
_writer: SQLiteWriter | None = None


def _bind_writer(writer: SQLiteWriter | None) -> None:
    """Set the module-level writer used by humaneval_metric."""
    global _writer
    _writer = writer


@contextlib.contextmanager
def _flow_context(flow: str) -> Iterator[None]:
    """Set the active flow label for the duration of a with-block."""
    token = _current_flow.set(flow)
    if _writer is not None:
        _writer.put_event("flow.start", payload={"flow": flow}, flow=flow)
    try:
        yield
    finally:
        if _writer is not None:
            _writer.put_event("flow.end", payload={"flow": flow}, flow=flow)
        _current_flow.reset(token)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _sanitize(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from an LM kwargs dict before logging."""
    if not kwargs:
        return {}
    return {k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v) for k, v in kwargs.items()}


def _signature_summary(sig_cls: type[dspy.Signature]) -> dict[str, Any]:
    """Summarize a Signature class for logging."""
    try:
        fields_summary = [
            (
                name,
                str(field.annotation),
                (field.json_schema_extra or {}).get("__dspy_field_type")
                if isinstance(field.json_schema_extra, dict)
                else None,
            )
            for name, field in sig_cls.fields.items()
        ]
    except Exception:
        fields_summary = []
    return {
        "signature": getattr(sig_cls, "signature", repr(sig_cls)),
        "instructions": getattr(sig_cls, "instructions", ""),
        "fields": fields_summary,
    }


def _to_jsonable_inner(x: Any, depth: int = 0) -> Any:
    """Recursive worker for to_jsonable. Depth-bounded to avoid pathological cycles."""
    if depth > 12:
        return repr(x)[:REPR_TRUNCATE]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_to_jsonable_inner(v, depth + 1) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable_inner(v, depth + 1) for k, v in x.items()}
    if isinstance(x, bytes):
        return f"<bytes len={len(x)}>"
    # DSPy Example / Prediction
    if isinstance(x, dspy.Example):
        try:
            return _to_jsonable_inner(x.toDict(), depth + 1)
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    # Signature class
    if isinstance(x, type):
        try:
            if issubclass(x, dspy.Signature):
                return _signature_summary(x)
        except TypeError:
            pass
        return f"<class {x.__module__}.{x.__name__}>"
    # LM instance
    if isinstance(x, dspy.BaseLM):
        return {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": _sanitize(getattr(x, "kwargs", {})),
        }
    # Pydantic model instance
    if isinstance(x, pydantic.BaseModel):
        try:
            return x.model_dump(mode="json")
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    # Coroutines / async iterators / generators — never await
    if inspect.iscoroutine(x) or inspect.isasyncgen(x) or inspect.isgenerator(x):
        return f"<{type(x).__name__}>"
    # dotdict and similar mapping-like
    if hasattr(x, "__dict__") and not callable(x):
        try:
            return {k: _to_jsonable_inner(v, depth + 1) for k, v in vars(x).items()}
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    return repr(x)[:REPR_TRUNCATE]


def to_jsonable(x: Any, *, max_bytes: int = PAYLOAD_MAX_BYTES) -> Any:
    """Serialize any object to a JSON-friendly structure. Never raises.

    If the resulting JSON exceeds ``max_bytes``, returns a truncated preview
    wrapped in ``{"_truncated": True, "preview": "..."}``.
    """
    try:
        value = _to_jsonable_inner(x)
        encoded = json.dumps(value, ensure_ascii=False, default=repr)
        if len(encoded.encode("utf-8")) > max_bytes:
            return {"_truncated": True, "preview": encoded[:max_bytes]}
        return value
    except Exception as e:
        return {"_serialize_error": repr(e), "repr": repr(x)[:REPR_TRUNCATE]}


# --------------------------------------------------------------------------- #
# SQLite writer
# --------------------------------------------------------------------------- #


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    run_id          TEXT    NOT NULL,
    flow            TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    call_id         TEXT,
    parent_call_id  TEXT,
    example_id      TEXT,
    score           REAL,
    error           TEXT,
    payload         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run   ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_call  ON events(call_id);
CREATE INDEX IF NOT EXISTS idx_events_type  ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ex    ON events(example_id);
"""

INSERT_SQL = (
    "INSERT INTO events "
    "(ts, run_id, flow, event_type, call_id, parent_call_id, example_id, "
    "score, error, payload) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class SQLiteWriter:
    """Background-thread SQLite writer fed by an unbounded queue."""

    _SENTINEL: object = object()

    def __init__(self, path: str, *, run_id: str) -> None:
        self.path = os.path.abspath(path)
        self.run_id = run_id
        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="sqlite-writer", daemon=True)
        self._thread.start()
        self._closed = False

    def put_event(
        self,
        event_type: str,
        payload: Any,
        *,
        flow: str | None = None,
        call_id: str | None = None,
        parent_call_id: str | None = None,
        example_id: str | None = None,
        score: float | None = None,
        error: str | None = None,
    ) -> None:
        """Enqueue an event. Non-blocking, never raises."""
        try:
            payload_json = json.dumps(to_jsonable(payload), ensure_ascii=False, default=repr)
        except Exception as e:
            payload_json = json.dumps({"_serialize_error": repr(e)})
        record: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "flow": flow or _current_flow.get(),
            "event_type": event_type,
            "call_id": call_id,
            "parent_call_id": parent_call_id,
            "example_id": example_id,
            "score": score,
            "error": error,
            "payload": payload_json,
        }
        try:
            self._q.put_nowait(record)
        except Exception as e:
            print(f"[SQLiteWriter.put_event] {e!r}", file=sys.stderr)

    def _run(
        self,
    ) -> None:
        try:
            conn = sqlite3.connect(self.path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.executescript(SCHEMA_SQL)
        except Exception as e:
            print(f"[SQLiteWriter._run] init failed: {e!r}", file=sys.stderr)
            return
        try:
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
                batch: list[dict[str, Any]] = [cast(dict[str, Any], item)]
                for _ in range(255):
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is self._SENTINEL:
                        self._q.put(self._SENTINEL)
                        break
                    batch.append(cast(dict[str, Any], nxt))
                try:
                    with conn:
                        conn.executemany(
                            INSERT_SQL,
                            [
                                (
                                    r["ts"],
                                    r["run_id"],
                                    r["flow"],
                                    r["event_type"],
                                    r["call_id"],
                                    r["parent_call_id"],
                                    r["example_id"],
                                    r["score"],
                                    r["error"],
                                    r["payload"],
                                )
                                for r in batch
                            ],
                        )
                except Exception as e:
                    print(f"[SQLiteWriter._run] insert failed: {e!r}", file=sys.stderr)
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def close(self, timeout: float = 10.0) -> None:
        """Flush queue and join the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._q.put(self._SENTINEL)
        self._thread.join(timeout=timeout)


# --------------------------------------------------------------------------- #
# DSPy callback → SQLite
# --------------------------------------------------------------------------- #


class SQLiteCallback(BaseCallback):
    """BaseCallback that writes every DSPy hook to a SQLiteWriter."""

    def __init__(self, writer: SQLiteWriter) -> None:
        super().__init__()
        self._writer = writer
        self._tokens: dict[str, contextvars.Token[str | None]] = {}

    # ----- helpers ----- #

    def _safe_log_start(self, event_type: str, call_id: str, payload: dict[str, Any]) -> None:
        try:
            parent = _current_call_id.get()
            token = _current_call_id.set(call_id)
            self._tokens[call_id] = token
            self._writer.put_event(event_type, payload=payload, call_id=call_id, parent_call_id=parent)
        except Exception as e:
            print(f"[SQLiteCallback {event_type}] {e!r}", file=sys.stderr)

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
            self._writer.put_event(event_type, payload=payload, call_id=call_id, error=err)
            if token is not None:
                _current_call_id.reset(token)
        except Exception as e:
            print(f"[SQLiteCallback {event_type}] {e!r}", file=sys.stderr)

    # ----- module ----- #

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "module.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_module_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("module.end", call_id, outputs, exception)

    # ----- lm ----- #

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "lm.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_lm_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("lm.end", call_id, outputs, exception)

    # ----- adapter format ----- #

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "adapter.format.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_adapter_format_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("adapter.format.end", call_id, outputs, exception)

    # ----- adapter parse ----- #

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "adapter.parse.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_adapter_parse_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("adapter.parse.end", call_id, outputs, exception)

    # ----- tool ----- #

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "tool.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_tool_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("tool.end", call_id, outputs, exception)

    # ----- evaluate ----- #

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safe_log_start(
            "evaluate.start",
            call_id,
            {"instance": to_jsonable(instance), "inputs": to_jsonable(inputs)},
        )

    def on_evaluate_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safe_log_end("evaluate.end", call_id, outputs, exception)


# --------------------------------------------------------------------------- #
# LM wrappers
# --------------------------------------------------------------------------- #


PutEventFn = Callable[..., None]


def _wrap_provider_response(field_values: dict[str, str]) -> Any:
    """Wrap a dict of output-field values as a ChatAdapter-compatible response.

    Mirrors what dspy.utils.dummies.DummyLM produces: a dotdict with
    ``choices=[dotdict(message=dotdict(content=str, tool_calls=None),
    finish_reason="stop")]`` and a zeroed ``usage``.
    """
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


class LoggingLM(dspy.LM):
    """dspy.LM subclass that logs the true post-preprocess payload + raw response."""

    def __init__(self, model: str, *, log: PutEventFn, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self._log = log

    def _log_request(self, req_id: str, messages: Any, kwargs: dict[str, Any]) -> None:
        try:
            self._log(
                "lm.request",
                payload={
                    "req_id": req_id,
                    "messages": to_jsonable(messages),
                    "kwargs": _sanitize(kwargs),
                },
            )
        except Exception as e:
            print(f"[LoggingLM log_request] {e!r}", file=sys.stderr)

    def _log_response(self, req_id: str, resp: Any, dt: float) -> None:
        try:
            self._log(
                "lm.response",
                payload={"req_id": req_id, "dt": dt, "response": to_jsonable(resp)},
            )
        except Exception as e:
            print(f"[LoggingLM log_response] {e!r}", file=sys.stderr)

    def _log_error(self, req_id: str, exc: BaseException, dt: float) -> None:
        try:
            self._log(
                "lm.error",
                payload={"req_id": req_id, "dt": dt, "error": repr(exc)},
                error=repr(exc),
            )
        except Exception as e:
            print(f"[LoggingLM log_error] {e!r}", file=sys.stderr)

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        self._log_request(req_id, messages, kwargs)
        try:
            resp = super().forward(prompt=prompt, messages=messages, **kwargs)
        except BaseException as e:
            self._log_error(req_id, e, time.time() - t0)
            raise
        self._log_response(req_id, resp, time.time() - t0)
        return resp

    async def aforward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        self._log_request(req_id, messages, kwargs)
        try:
            resp = await super().aforward(prompt=prompt, messages=messages, **kwargs)
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
        fn: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, str]],
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
        self.calls.append({"messages": msg_list, "kwargs": dict(kwargs), "output": output})
        return _wrap_provider_response(output)

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        return self._serve(messages, kwargs)

    async def aforward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        return self._serve(messages, kwargs)


class LoggingCallableLM(CallableLM):
    """CallableLM with the same lm.request/response/error logging as LoggingLM."""

    def __init__(
        self,
        fn: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, str]],
        *,
        log: PutEventFn,
        model: str = "callable/mock",
    ) -> None:
        super().__init__(fn, model=model)
        self._log = log

    def forward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        req_id = uuid.uuid4().hex
        t0 = time.time()
        try:
            self._log(
                "lm.request",
                payload={
                    "req_id": req_id,
                    "messages": to_jsonable(messages),
                    "kwargs": _sanitize(kwargs),
                },
            )
        except Exception as e:
            print(f"[LoggingCallableLM req] {e!r}", file=sys.stderr)
        try:
            resp = super().forward(prompt=prompt, messages=messages, **kwargs)
        except BaseException as e:
            try:
                self._log(
                    "lm.error",
                    payload={
                        "req_id": req_id,
                        "dt": time.time() - t0,
                        "error": repr(e),
                    },
                    error=repr(e),
                )
            except Exception as e2:
                print(f"[LoggingCallableLM err] {e2!r}", file=sys.stderr)
            raise
        try:
            self._log(
                "lm.response",
                payload={
                    "req_id": req_id,
                    "dt": time.time() - t0,
                    "response": to_jsonable(resp),
                },
            )
        except Exception as e:
            print(f"[LoggingCallableLM resp] {e!r}", file=sys.stderr)
        return resp

    async def aforward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        return self.forward(prompt=prompt, messages=messages, **kwargs)


# --------------------------------------------------------------------------- #
# Sandboxed code scoring
# --------------------------------------------------------------------------- #


def _worker(
    code: str,
    test: str,
    entry_point: str,
    mem_limit_bytes: int,
    cpu_limit_seconds: int,
    conn: Connection,
) -> None:
    """Child process body: apply rlimits, exec code+test, send (status, msg)."""
    try:
        import resource
        import signal

        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))

        def _alarm_handler(_sig: int, _frame: Any) -> None:
            raise TimeoutError("CPU alarm fired")

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(cpu_limit_seconds))

        ns: dict[str, Any] = {}
        exec(code, ns)
        exec(test, ns)
        check_fn = ns.get("check")
        candidate = ns.get(entry_point)
        if check_fn is None or candidate is None:
            conn.send(("err", f"missing check or entry_point={entry_point!r}"))
            return
        check_fn(candidate)
        conn.send(("ok", None))
    except BaseException as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def _run_in_subprocess(
    *,
    code: str,
    test: str,
    entry_point: str,
    timeout: float,
    mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
    cpu_limit_seconds: int = DEFAULT_CPU_LIMIT_SECONDS,
) -> tuple[float, str | None]:
    """Run ``code`` + ``test`` in a sandboxed child. Returns (score, error_or_None)."""
    method = "fork" if platform.system() != "Windows" else "spawn"
    ctx: Any = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker,
        args=(code, test, entry_point, mem_limit_bytes, cpu_limit_seconds, child_conn),
    )
    proc.start()
    child_conn.close()  # parent only reads
    score = 0.0
    err: str | None = None
    try:
        if parent_conn.poll(timeout):
            try:
                status, msg = parent_conn.recv()
            except EOFError:
                status, msg = "err", "EOF from worker"
            if status == "ok":
                score = 1.0
            else:
                err = str(msg)
        else:
            err = f"timeout after {timeout}s"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
        with contextlib.suppress(Exception):
            parent_conn.close()
    return score, err


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #


def _extract_source(pred: Any) -> str:
    """Pull the Python source out of a dspy.Predict result whose output is dspy.Code."""
    code_field = getattr(pred, "code", None)
    if code_field is None:
        return ""
    # Common cases: dspy.Code.code is the string; or it's already a string.
    inner = getattr(code_field, "code", None)
    if isinstance(inner, str):
        return inner
    if isinstance(code_field, str):
        return code_field
    try:
        as_str = str(code_field)
    except Exception:
        return ""
    # dspy.Code's __str__ may be "code='...'" — strip the wrapper if present.
    if as_str.startswith("code="):
        try:
            return as_str.split("=", 1)[1].strip().strip("'\"")
        except Exception:
            return as_str
    return as_str


# Subprocess timeout is module-level so the metric (which doesn't take config)
# can read it. main_* sets it before running.
_subprocess_timeout: float = DEFAULT_SUBPROCESS_TIMEOUT


def _set_subprocess_timeout(seconds: float) -> None:
    global _subprocess_timeout
    _subprocess_timeout = seconds


def humaneval_metric(example: dspy.Example, pred: Any, trace: list[Any] | None = None) -> float | bool:
    """Score one (example, prediction) pair by sandboxed test execution.

    Returns a float in {0.0, 1.0} when called by Evaluate (``trace is None``),
    and a bool when called by BootstrapFewShot during bootstrapping.
    """
    code = _extract_source(pred)
    score, err = _run_in_subprocess(
        code=code,
        test=example.test,
        entry_point=example.entry_point,
        timeout=_subprocess_timeout,
    )
    if _writer is not None:
        _writer.put_event(
            "metric.score",
            payload={"task_id": example.task_id, "code": code, "error": err},
            example_id=getattr(example, "task_id", None),
            score=score,
            error=err,
        )
    if trace is None:
        return score
    return score >= 1.0


# --------------------------------------------------------------------------- #
# Signature & dataset
# --------------------------------------------------------------------------- #


class Solve(dspy.Signature):
    """Write a self-contained Python function that satisfies the prompt.

    Include any imports inside the function or at the top. Do not include
    tests or example calls. Define exactly the function named in the prompt.
    """

    prompt: str = dspy.InputField()
    code: dspy.Code = dspy.OutputField()


HumanEvalRow = Mapping[str, Any]


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


def build_dataset(
    *,
    seed: int = DEFAULT_SEED,
    train_size: int = DEFAULT_TRAIN_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load HumanEval+ and produce (trainset, devset). Requires the ``datasets`` lib."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = cast(HumanEvalDataset, load_dataset("evalplus/humanevalplus", split="test"))
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    rows = [ds[i] for i in indices]
    examples = [
        dspy.Example(
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
            task_id=row["task_id"],
        ).with_inputs("prompt")
        for row in rows
    ]
    return examples[:train_size], examples[train_size : train_size + dev_size]


# --------------------------------------------------------------------------- #
# Real / mock runs
# --------------------------------------------------------------------------- #


def _run_metadata(args: argparse.Namespace, *, model_id: str) -> dict[str, Any]:
    import dspy as _dspy

    return {
        "argv": sys.argv,
        "args": vars(args),
        "model": model_id,
        "dspy_version": getattr(_dspy, "__version__", "?"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def _check_demos(compiled: dspy.Module, writer: SQLiteWriter) -> int:
    """Return total demo count; log a `bootstrap.no_demos` event if zero."""
    total = 0
    by_predictor: list[tuple[str, int]] = []
    for name, p in compiled.named_predictors():
        n = len(getattr(p, "demos", []) or [])
        by_predictor.append((name, n))
        total += n
    if total == 0:
        writer.put_event(
            "bootstrap.no_demos",
            payload={
                "by_predictor": by_predictor,
                "hint": (
                    "BootstrapFewShot produced zero demos at the full-pass "
                    "threshold. Either pass a stronger teacher= to compile() "
                    "or lower the threshold by changing humaneval_metric's "
                    "`trace is not None` branch to `return score >= 0.8`."
                ),
            },
        )
    return total


def _build_eval(devset: list[dspy.Example], num_threads: int) -> dspy.Evaluate:
    return dspy.Evaluate(
        devset=devset,
        metric=humaneval_metric,
        num_threads=num_threads,
        display_progress=True,
        display_table=10,
    )


class AsyncProgramRunner:
    """Sync-callable wrapper that drives ``program.acall`` via a per-thread loop.

    ``dspy.Evaluate`` parallelises via threads and invokes ``program(...)``
    synchronously, so wrapping with ``dspy.asyncify`` would return a coroutine
    that never gets awaited. This wrapper instead exposes a sync ``__call__``
    that runs the program's async path (``acall`` → ``aforward`` → LM
    ``aforward``) on a per-thread asyncio loop, giving us true async I/O while
    staying compatible with Evaluate's sync executor.
    """

    def __init__(self, program: dspy.Module) -> None:
        self.program = program
        self._tls = threading.local()
        self._loops: list[asyncio.AbstractEventLoop] = []
        self._loops_lock = threading.Lock()

    def _loop(self) -> asyncio.AbstractEventLoop:
        loop = getattr(self._tls, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            self._tls.loop = loop
            with self._loops_lock:
                self._loops.append(loop)
        return loop

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        loop = self._loop()
        return loop.run_until_complete(self.program.acall(*args, **kwargs))

    def close(self) -> None:
        """Close every per-thread loop created by this runner."""
        with self._loops_lock:
            loops = list(self._loops)
            self._loops.clear()
        for loop in loops:
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception as e:
                print(f"[AsyncProgramRunner close] {e!r}", file=sys.stderr)


def main_real(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set; either export it or use --mock.",
            file=sys.stderr,
        )
        return 2
    return _run_flows(args, mock=False)


def main_mock(args: argparse.Namespace) -> int:
    return _run_flows(args, mock=True)


def _make_mock_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Tiny inline dataset for --mock runs (no HF download)."""
    rows = [
        {
            "task_id": "mock/add",
            "prompt": "def add(a, b):\n    'return a + b'\n",
            "entry_point": "add",
            "test": ("def check(candidate):\n    assert candidate(1,2)==3\n    assert candidate(-1,1)==0\n"),
        },
        {
            "task_id": "mock/sub",
            "prompt": "def sub(a, b):\n    'return a - b'\n",
            "entry_point": "sub",
            "test": ("def check(candidate):\n    assert candidate(3,2)==1\n    assert candidate(0,5)==-5\n"),
        },
        {
            "task_id": "mock/mul",
            "prompt": "def mul(a, b):\n    'return a * b'\n",
            "entry_point": "mul",
            "test": "def check(candidate):\n    assert candidate(2,3)==6\n",
        },
        {
            "task_id": "mock/identity",
            "prompt": "def identity(x):\n    'return x'\n",
            "entry_point": "identity",
            "test": "def check(candidate):\n    assert candidate(7)==7\n",
        },
    ]
    train = [dspy.Example(**r).with_inputs("prompt") for r in rows]
    dev = [dspy.Example(**r).with_inputs("prompt") for r in rows]
    return train, dev


def _mock_solver(messages: list[dict[str, Any]], _kwargs: dict[str, Any]) -> dict[str, str]:
    """Heuristic mock LM: read entry_point from the user prompt, emit matching code."""
    text = "\n".join(str(m.get("content", "")) for m in messages)
    body_map = {
        "add": "def add(a, b):\n    return a + b\n",
        "sub": "def sub(a, b):\n    return a - b\n",
        "mul": "def mul(a, b):\n    return a * b\n",
        "identity": "def identity(x):\n    return x\n",
    }
    for name, body in body_map.items():
        # Match prefer the function-definition form to avoid false hits.
        if re.search(rf"\bdef\s+{name}\s*\(", text):
            return {"code": body}
    return {"code": "def f():\n    return None\n"}


def _run_flows(args: argparse.Namespace, *, mock: bool) -> int:
    run_id = uuid.uuid4().hex
    writer = SQLiteWriter(args.db_path, run_id=run_id)
    _bind_writer(writer)
    _set_subprocess_timeout(args.timeout)

    rc = 0
    try:
        cb = SQLiteCallback(writer)
        if mock:
            lm: dspy.BaseLM = LoggingCallableLM(_mock_solver, log=writer.put_event)
        else:
            lm = LoggingLM(args.model, log=writer.put_event, cache=False)
        dspy.configure(lm=lm, callbacks=[cb], track_usage=True, max_trace_size=10_000)

        writer.put_event("run.start", payload=_run_metadata(args, model_id=lm.model))

        if mock:
            trainset, devset = _make_mock_dataset()
        else:
            trainset, devset = build_dataset(seed=args.seed, train_size=args.train_size, dev_size=args.dev_size)

        evaluator = _build_eval(devset, args.num_threads)

        with _flow_context("eval_baseline"):
            baseline_program = dspy.Predict(Solve)
            baseline_runner = AsyncProgramRunner(baseline_program)
            try:
                baseline = evaluator(baseline_runner)
            except Exception:
                traceback.print_exc()
                baseline = evaluator(baseline_program)
            finally:
                baseline_runner.close()

        with _flow_context("optimize"):
            opt = BootstrapFewShot(
                metric=humaneval_metric,
                max_bootstrapped_demos=args.max_bootstrapped_demos,
                max_labeled_demos=args.max_labeled_demos,
            )
            student = dspy.Predict(Solve)
            compiled = opt.compile(student=student, trainset=trainset)
            total_demos = _check_demos(compiled, writer)
            if total_demos == 0:
                print(
                    "BootstrapFewShot produced zero demos at the full-pass threshold. "
                    "See bootstrap.no_demos in the SQLite log for next steps.",
                    file=sys.stderr,
                )
                rc = 3

        with _flow_context("eval_optimized"):
            optimized_runner = AsyncProgramRunner(compiled)
            try:
                optimized = evaluator(optimized_runner)
            except Exception:
                traceback.print_exc()
                optimized = evaluator(compiled)
            finally:
                optimized_runner.close()

        try:
            compiled.save(args.compiled_path)
        except Exception as e:
            print(f"[save compiled] {e!r}", file=sys.stderr)

        writer.put_event(
            "run.end",
            payload={
                "baseline_score": float(baseline.score),
                "optimized_score": float(optimized.score),
                "total_demos": total_demos,
            },
        )
        print(f"baseline:  {baseline.score:.3f}")
        print(f"optimized: {optimized.score:.3f}")
    except Exception as e:
        traceback.print_exc()
        with contextlib.suppress(Exception):
            writer.put_event("run.error", payload={"error": repr(e)}, error=repr(e))
        rc = 1
    finally:
        _bind_writer(None)
        writer.close()
    return rc


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #


TestResult = tuple[str, bool, str, float]


def _t(name: str, fn: Callable[[], str | None]) -> TestResult:
    t0 = time.time()
    try:
        detail = fn() or ""
        return (name, True, detail, time.time() - t0)
    except AssertionError as e:
        return (name, False, f"assert: {e}", time.time() - t0)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}", time.time() - t0)


def test_metric_scoring() -> str:
    cases = [
        (
            "pass",
            "def add(a, b):\n    return a + b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            1.0,
            None,
        ),
        (
            "fail",
            "def add(a, b):\n    return a - b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "AssertionError",
        ),
        (
            "syntax",
            "def add(a, b: return a+b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "SyntaxError",
        ),
        (
            "crash",
            "def add(a, b):\n    raise ValueError('x')\n",
            "def check(candidate):\n    candidate(1,2)\n",
            "add",
            0.0,
            "ValueError",
        ),
    ]
    for name, code, test, ep, expected_score, expected_err_sub in cases:
        score, err = _run_in_subprocess(code=code, test=test, entry_point=ep, timeout=5.0)
        assert score == expected_score, f"case {name}: score {score} != {expected_score}"
        if expected_err_sub is not None:
            assert err is not None and expected_err_sub in err, f"case {name}: err {err!r} missing {expected_err_sub!r}"
    # Timeout case
    t0 = time.time()
    score, err = _run_in_subprocess(
        code="def loop():\n    while True: pass\n",
        test="def check(candidate):\n    candidate()\n",
        entry_point="loop",
        timeout=2.0,
        cpu_limit_seconds=3,
    )
    wall = time.time() - t0
    assert score == 0.0, f"timeout: score {score} != 0.0"
    assert err is not None and "timeout" in err, f"timeout: err {err!r}"
    assert wall < 6.0, f"timeout took {wall:.2f}s (too long)"
    return f"5/5 cases (timeout wall={wall:.2f}s)"


def test_to_jsonable_robustness() -> str:
    # Example / Prediction
    ex = dspy.Example(question="why?", answer="because").with_inputs("question")
    assert isinstance(to_jsonable(ex), dict)
    pred = dspy.Prediction(code="x")
    assert isinstance(to_jsonable(pred), dict)
    # Signature class
    sig_json = to_jsonable(Solve)
    assert isinstance(sig_json, dict) and "signature" in sig_json
    # LM instance (no network)
    lm = dspy.LM("openai/gpt-4o-mini")
    lm_json = to_jsonable(lm)
    assert isinstance(lm_json, dict) and lm_json.get("model") == "openai/gpt-4o-mini"

    # Coroutine — must NOT be awaited
    async def _coro() -> int:
        return 1

    c = _coro()
    try:
        out = to_jsonable(c)
        assert isinstance(out, str) and "coroutine" in out
    finally:
        c.close()

    # Pydantic model
    class M(pydantic.BaseModel):
        x: int
        y: str

    assert to_jsonable(M(x=1, y="a")) == {"x": 1, "y": "a"}
    # Weird containers
    weird = {
        "s": {1, 2, 3},
        "fs": frozenset({4, 5}),
        "b": b"hello",
        "nested": [{"k": (1, 2)}],
    }
    out = to_jsonable(weird)
    assert isinstance(out, dict) and "s" in out and "b" in out
    # Big string truncation
    big = "x" * (PAYLOAD_MAX_BYTES + 1000)
    out_big = to_jsonable(big)
    assert isinstance(out_big, dict) and out_big.get("_truncated") is True
    return "8/8 cases"


def test_bootstrap_selects_passing_demos() -> str:
    # In-memory log sink to count LM calls
    calls: list[dict[str, Any]] = []

    def sink(event_type: str, **kw: Any) -> None:
        calls.append(kw)

    # Trainset: passing examples produce code that passes; failing examples
    # produce wrong code.
    train = [
        dspy.Example(
            task_id=f"toy/{name}",
            prompt=f"def {name}(a, b):\n    'do it'\n",
            test=test,
            entry_point=name,
        ).with_inputs("prompt")
        for name, test in [
            ("add", "def check(candidate):\n    assert candidate(1,2)==3\n"),
            ("sub", "def check(candidate):\n    assert candidate(5,3)==2\n"),
            ("mul", "def check(candidate):\n    assert candidate(2,3)==6\n"),
        ]
    ]

    def solver(messages: list[dict[str, Any]], _kw: dict[str, Any]) -> dict[str, str]:
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if re.search(r"\bdef\s+add\s*\(", text):
            return {"code": "def add(a, b):\n    return a + b\n"}
        if re.search(r"\bdef\s+sub\s*\(", text):
            return {"code": "def sub(a, b):\n    return a - b\n"}
        if re.search(r"\bdef\s+mul\s*\(", text):
            return {"code": "def mul(a, b):\n    return a * b\n"}
        return {"code": "def f():\n    return None\n"}

    # Use a tiny SQLite writer just so metric.score events are accepted
    writer = SQLiteWriter(":memory:" if False else "./.test_runs.db", run_id=uuid.uuid4().hex)
    _bind_writer(writer)
    _set_subprocess_timeout(5.0)
    try:
        lm = LoggingCallableLM(solver, log=writer.put_event)
        dspy.configure(lm=lm, callbacks=[])
        student = dspy.Predict(Solve)
        opt = BootstrapFewShot(metric=humaneval_metric, max_bootstrapped_demos=4, max_labeled_demos=0)
        compiled = opt.compile(student=student, trainset=train)
        preds = compiled.named_predictors()
        assert len(preds) == 1, f"expected 1 predictor, got {len(preds)}"
        _, p = preds[0]
        demos = list(p.demos or [])
        # All three examples pass with this mock, so we expect 3 demos selected.
        assert len(demos) >= 1, f"expected at least 1 demo, got {len(demos)}"
        assert len(demos) <= 4, f"expected at most 4 demos, got {len(demos)}"
        # Demo prompts should all be from the trainset
        train_prompts = {ex.prompt for ex in train}
        for d in demos:
            assert d.prompt in train_prompts, f"demo not from trainset: {d.prompt!r}"
        # LM should have been called at least once per training example
        assert len(lm.calls) >= len(train), f"lm.calls={len(lm.calls)} < trainset={len(train)}"
        return f"{len(demos)} demos selected (lm calls: {len(lm.calls)})"
    finally:
        _bind_writer(None)
        writer.close()
        with contextlib.suppress(Exception):
            os.remove("./.test_runs.db")


def test_logging_lm_captures_payload() -> str:
    db_path = "./.test_logging.db"
    with contextlib.suppress(Exception):
        os.remove(db_path)
    writer = SQLiteWriter(db_path, run_id=uuid.uuid4().hex)
    _bind_writer(writer)
    try:

        def solver(_msgs: list[dict[str, Any]], _kw: dict[str, Any]) -> dict[str, str]:
            return {"code": "def f():\n    return 1\n"}

        lm = LoggingCallableLM(solver, log=writer.put_event)
        dspy.configure(lm=lm, callbacks=[])
        p = dspy.Predict(Solve)
        _ = p(prompt="def f():\n    'return 1'\n")
        # Drain the queue
        writer.close()
        # Reopen to query
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT event_type, payload FROM events WHERE event_type IN ('lm.request','lm.response')"
            ).fetchall()
        finally:
            conn.close()
        reqs = [json.loads(p) for et, p in rows if et == "lm.request"]
        resps = [json.loads(p) for et, p in rows if et == "lm.response"]
        assert len(reqs) >= 1 and len(resps) >= 1, f"requests={len(reqs)} responses={len(resps)}"
        assert reqs[0]["req_id"] == resps[0]["req_id"], "req_id mismatch"
        # messages should be non-empty
        assert reqs[0].get("messages"), "messages empty in lm.request"
        assert "dt" in resps[0] and isinstance(resps[0]["dt"], (int, float))
        req_id_short = reqs[0]["req_id"][:8]
        dt = resps[0]["dt"]
        return f"req/resp pair matched (req_id={req_id_short}..., dt={dt:.4f}s)"
    finally:
        _bind_writer(None)
        with contextlib.suppress(Exception):
            os.remove(db_path)


def test_full_harness_smoke() -> str:
    db_path = "./.test_smoke.db"
    compiled_path = "./.test_smoke_compiled.json"
    for p in (db_path, compiled_path):
        with contextlib.suppress(Exception):
            os.remove(p)
    args = argparse.Namespace(
        db_path=db_path,
        compiled_path=compiled_path,
        model="callable/mock",
        seed=0,
        train_size=4,
        dev_size=4,
        num_threads=2,
        timeout=5.0,
        max_bootstrapped_demos=4,
        max_labeled_demos=0,
    )
    rc = main_mock(args)
    assert rc == 0, f"main_mock returned {rc}"
    assert os.path.exists(db_path), "db not created"
    assert os.path.exists(compiled_path), "compiled artifact not created"
    conn = sqlite3.connect(db_path)
    try:
        types_present = {r[0] for r in conn.execute("SELECT DISTINCT event_type FROM events").fetchall()}
        flows = {
            r[0] for r in conn.execute("SELECT DISTINCT flow FROM events WHERE event_type='flow.start'").fetchall()
        }
        rows = conn.execute("SELECT COUNT(*), MIN(payload IS NULL) FROM events").fetchone()
    finally:
        conn.close()
    required = {
        "run.start",
        "flow.start",
        "module.start",
        "module.end",
        "lm.request",
        "lm.response",
        "adapter.format.end",
        "adapter.parse.end",
        "metric.score",
        "evaluate.end",
        "run.end",
    }
    missing = required - types_present
    assert not missing, f"missing event_types: {missing}"
    assert {"eval_baseline", "optimize", "eval_optimized"} <= flows, f"flows={flows}"
    assert rows[0] > 0, "no rows"
    assert rows[1] in (0, None), "found NULL payload"
    # Load compiled artifact back
    fresh = dspy.Predict(Solve)
    fresh.load(compiled_path)
    for p in (db_path, compiled_path):
        with contextlib.suppress(Exception):
            os.remove(p)
    return f"events={rows[0]}, types={len(types_present)}, flows={sorted(flows)}"


ALL_TESTS: list[tuple[str, Callable[[], str | None]]] = [
    ("test_metric_scoring", test_metric_scoring),
    ("test_to_jsonable_robustness", test_to_jsonable_robustness),
    ("test_logging_lm_captures_payload", test_logging_lm_captures_payload),
    ("test_bootstrap_selects_passing_demos", test_bootstrap_selects_passing_demos),
    ("test_full_harness_smoke", test_full_harness_smoke),
]


def run_tests() -> int:
    results: list[TestResult] = [_t(name, fn) for name, fn in ALL_TESTS]
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = len(results) - n_pass
    width = max(len(name) for name, *_ in results)
    total_dt = 0.0
    for name, ok, detail, dt in results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {name.ljust(width)}  ({detail}, {dt:.2f}s)")
        total_dt += dt
    print()
    print(f"{n_pass} passed, {n_fail} failed in {total_dt:.2f}s")
    return 0 if n_fail == 0 else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HumanEval+ x DSPy harness")
    p.add_argument("--mock", action="store_true", help="Run with CallableLM mock instead of OpenAI")
    p.add_argument("--test", action="store_true", help="Run deterministic self-tests")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH)
    p.add_argument("--compiled-path", default=DEFAULT_COMPILED_PATH)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    p.add_argument("--dev-size", type=int, default=DEFAULT_DEV_SIZE)
    p.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    p.add_argument("--timeout", type=float, default=DEFAULT_SUBPROCESS_TIMEOUT)
    p.add_argument("--max-bootstrapped-demos", type=int, default=DEFAULT_MAX_BOOTSTRAPPED_DEMOS)
    p.add_argument("--max-labeled-demos", type=int, default=DEFAULT_MAX_LABELED_DEMOS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.getLogger("dspy").setLevel(logging.WARNING)
    if args.test:
        return run_tests()
    if args.mock:
        return main_mock(args)
    return main_real(args)


if __name__ == "__main__":
    # multiprocessing start method: prefer fork on POSIX for speed; spawn elsewhere.
    if platform.system() != "Windows":
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("fork", force=True)
    else:
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
