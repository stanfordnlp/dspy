"""Event-log writers for harness telemetry."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, cast

from dr_dspy.serialization import to_jsonable

DEFAULT_FLOW = "unknown"
BATCH_SIZE = 256
DATABASE_URL_ENV = "DATABASE_URL"

DefaultFlowFn = Callable[[], str]


class EventStore(str, Enum):
    """Supported event-log backends."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


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


def build_event_writer(
    *,
    event_store: EventStore | str,
    run_id: str,
    db_path: str | None = None,
    database_url: str | None = None,
    default_flow_fn: DefaultFlowFn = default_flow,
) -> EventWriter:
    """Build an event writer from explicit backend configuration."""
    store = EventStore(event_store)
    if store is EventStore.POSTGRES:
        resolved_url = database_url or os.environ.get(DATABASE_URL_ENV)
        if not resolved_url:
            raise ValueError(
                f"--database-url or {DATABASE_URL_ENV} is required with "
                "--event-store postgres"
            )
        return PostgresWriter(
            resolved_url, run_id=run_id, default_flow_fn=default_flow_fn
        )
    if not db_path:
        raise ValueError("db_path is required with --event-store sqlite")
    return SQLiteWriter(
        db_path, run_id=run_id, default_flow_fn=default_flow_fn
    )


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

POSTGRES_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS events (
        id              BIGSERIAL PRIMARY KEY,
        ts              DOUBLE PRECISION NOT NULL,
        run_id          TEXT             NOT NULL,
        flow            TEXT             NOT NULL,
        event_type      TEXT             NOT NULL,
        call_id         TEXT,
        parent_call_id  TEXT,
        example_id      TEXT,
        score           DOUBLE PRECISION,
        error           TEXT,
        payload         JSONB            NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_run  ON events(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_call ON events(call_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_events_ex   ON events(example_id)",
)

POSTGRES_INSERT_SQL = (
    "INSERT INTO events "
    "(ts, run_id, flow, event_type, call_id, parent_call_id, example_id, "
    "score, error, payload) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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


class PostgresWriter:
    """Background-thread Postgres writer fed by an unbounded queue."""

    _SENTINEL: object = object()

    def __init__(
        self,
        database_url: str,
        *,
        run_id: str,
        default_flow_fn: DefaultFlowFn = default_flow,
    ) -> None:
        self.database_url = database_url
        self.run_id = run_id
        self._default_flow_fn = default_flow_fn
        self._q: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="postgres-writer", daemon=True
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
            payload_value = to_jsonable(payload)
        except Exception as e:
            payload_value = {"_serialize_error": repr(e)}
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
            "payload": payload_value,
        }
        try:
            self._q.put_nowait(record)
        except Exception as e:
            print(f"[PostgresWriter.put_event] {e!r}", file=sys.stderr)

    def _run(self) -> None:
        try:
            import psycopg
            from psycopg.types.json import Jsonb

            conn = psycopg.connect(self.database_url)
            with conn.cursor() as cur:
                for stmt in POSTGRES_SCHEMA_SQL:
                    cur.execute(cast(Any, stmt))
            conn.commit()
        except Exception as e:
            print(f"[PostgresWriter._run] init failed: {e!r}", file=sys.stderr)
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
                    with conn.transaction():
                        with conn.cursor() as cur:
                            cur.executemany(
                                POSTGRES_INSERT_SQL,
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
                                        Jsonb(r["payload"]),
                                    )
                                    for r in batch
                                ],
                            )
                except Exception as e:
                    print(
                        f"[PostgresWriter._run] insert failed: {e!r}",
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
