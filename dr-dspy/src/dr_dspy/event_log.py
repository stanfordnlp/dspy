"""Event-log writers and JSON serialization helpers for harness telemetry."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, cast

import pydantic

import dspy

PAYLOAD_MAX_BYTES = 256 * 1024
REPR_TRUNCATE = 4096
SANITIZE_KEYS = frozenset(
    {"api_key", "api_base", "base_url", "model_list", "authorization"}
)
DEFAULT_FLOW = "unknown"
BATCH_SIZE = 256

DefaultFlowFn = Callable[[], str]


class EventWriter(Protocol):
    """Storage boundary for run event logs."""

    run_id: str

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
        """Enqueue an event. Implementations should not raise."""

    def close(self, timeout: float = 10.0) -> None:
        """Flush queued events and release writer resources."""


def default_flow() -> str:
    """Return the fallback flow label used outside an active harness flow."""
    return DEFAULT_FLOW


def _sanitize(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Strip credential-like keys from an LM kwargs dict before logging."""
    if not kwargs:
        return {}
    return {
        k: ("<redacted>" if k.lower() in SANITIZE_KEYS else v)
        for k, v in kwargs.items()
    }


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
    if isinstance(x, dspy.Example):
        try:
            return _to_jsonable_inner(x.toDict(), depth + 1)
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    if isinstance(x, type):
        try:
            if issubclass(x, dspy.Signature):
                return _signature_summary(x)
        except TypeError:
            pass
        return f"<class {x.__module__}.{x.__name__}>"
    if isinstance(x, dspy.BaseLM):
        return {
            "_kind": "BaseLM",
            "class": f"{type(x).__module__}.{type(x).__name__}",
            "model": getattr(x, "model", None),
            "kwargs": _sanitize(getattr(x, "kwargs", {})),
        }
    if isinstance(x, pydantic.BaseModel):
        try:
            return x.model_dump(mode="json")
        except Exception:
            return repr(x)[:REPR_TRUNCATE]
    if (
        inspect.iscoroutine(x)
        or inspect.isasyncgen(x)
        or inspect.isgenerator(x)
    ):
        return f"<{type(x).__name__}>"
    if hasattr(x, "__dict__") and not callable(x):
        try:
            return {
                k: _to_jsonable_inner(v, depth + 1) for k, v in vars(x).items()
            }
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


SQLITE_SCHEMA_SQL = """
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

SQLITE_INSERT_SQL = (
    "INSERT INTO events "
    "(ts, run_id, flow, event_type, call_id, parent_call_id, example_id, "
    "score, error, payload) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class SQLiteWriter:
    """Background-thread SQLite writer fed by an unbounded queue."""

    _SENTINEL: object = object()

    def __init__(
        self,
        path: str,
        *,
        run_id: str,
        default_flow_fn: DefaultFlowFn = default_flow,
    ) -> None:
        self.path = os.path.abspath(path)
        self.run_id = run_id
        self._default_flow_fn = default_flow_fn
        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="sqlite-writer", daemon=True
        )
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
            payload_json = json.dumps(
                to_jsonable(payload), ensure_ascii=False, default=repr
            )
        except Exception as e:
            payload_json = json.dumps({"_serialize_error": repr(e)})
        record: dict[str, Any] = {
            "ts": time.time(),
            "run_id": self.run_id,
            "flow": flow or self._default_flow_fn(),
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

    def _run(self) -> None:
        try:
            conn = sqlite3.connect(self.path)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.executescript(SQLITE_SCHEMA_SQL)
        except Exception as e:
            print(f"[SQLiteWriter._run] init failed: {e!r}", file=sys.stderr)
            return
        try:
            while True:
                item = self._q.get()
                if item is self._SENTINEL:
                    break
                batch: list[dict[str, Any]] = [cast(dict[str, Any], item)]
                for _ in range(BATCH_SIZE - 1):
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
                            SQLITE_INSERT_SQL,
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
                    print(
                        f"[SQLiteWriter._run] insert failed: {e!r}",
                        file=sys.stderr,
                    )
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
