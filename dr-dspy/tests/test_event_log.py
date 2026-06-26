from __future__ import annotations

import json
import os
import sqlite3
import uuid

import pytest

from dr_dspy.event_log import (
    DATABASE_URL_ENV,
    EventStore,
    PostgresWriter,
    build_event_writer,
)


def test_build_event_writer_sqlite_uses_default_flow(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    writer = build_event_writer(
        event_store=EventStore.SQLITE,
        run_id=uuid.uuid4().hex,
        db_path=str(db_path),
        default_flow_fn=lambda: "unit_test",
    )
    writer.put_event("test.sqlite_writer", payload={"ok": True}, score=1.0)
    writer.close()

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT flow, event_type, score, payload FROM events"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    flow, event_type, score, payload = row
    assert flow == "unit_test"
    assert event_type == "test.sqlite_writer"
    assert score == 1.0
    assert json.loads(payload) == {"ok": True}


def test_build_event_writer_postgres_requires_database_url(monkeypatch) -> None:
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    with pytest.raises(ValueError, match=DATABASE_URL_ENV):
        build_event_writer(
            event_store=EventStore.POSTGRES,
            run_id=uuid.uuid4().hex,
        )


def test_postgres_writer_optional() -> None:
    database_url = os.environ.get(DATABASE_URL_ENV)
    if not database_url:
        pytest.skip(f"{DATABASE_URL_ENV} not set")

    run_id = uuid.uuid4().hex
    writer = PostgresWriter(
        database_url,
        run_id=run_id,
        default_flow_fn=lambda: "postgres_test",
    )
    writer.put_event(
        "test.postgres_writer",
        payload={"ok": True, "items": [1, 2, 3]},
        score=1.0,
    )
    writer.close()

    import psycopg

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT flow, event_type, score, payload
                FROM events
                WHERE run_id = %s AND event_type = 'test.postgres_writer'
                """,
                (run_id,),
            )
            row = cur.fetchone()
            cur.execute("DELETE FROM events WHERE run_id = %s", (run_id,))
        conn.commit()

    assert row is not None
    flow, event_type, score, payload = row
    assert flow == "postgres_test"
    assert event_type == "test.postgres_writer"
    assert score == 1.0
    assert isinstance(payload, dict)
    assert payload.get("ok") is True
