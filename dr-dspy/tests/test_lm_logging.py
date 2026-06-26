from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

import dspy

from dr_dspy.event_log import SQLiteWriter
from dr_dspy.lm_logging import LoggingCallableLM


class Solve(dspy.Signature):
    prompt: str = dspy.InputField()
    code: dspy.Code = dspy.OutputField()


def test_logging_callable_lm_captures_request_response_payloads(
    tmp_path,
) -> None:
    db_path = tmp_path / "logging.db"
    writer = SQLiteWriter(str(db_path), run_id=uuid.uuid4().hex)

    def solver(
        _messages: list[dict[str, Any]], _kwargs: dict[str, Any]
    ) -> dict[str, str]:
        return {"code": "def f():\n    return 1\n"}

    try:
        lm = LoggingCallableLM(solver, log=writer.put_event)
        dspy.configure(lm=lm, callbacks=[])
        predictor = dspy.Predict(Solve)
        predictor(prompt="def f():\n    'return 1'\n")
    finally:
        writer.close()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT event_type, payload
            FROM events
            WHERE event_type IN ('lm.request','lm.response')
            """
        ).fetchall()
    finally:
        conn.close()

    requests = [
        json.loads(payload) for event_type, payload in rows if event_type == "lm.request"
    ]
    responses = [
        json.loads(payload) for event_type, payload in rows if event_type == "lm.response"
    ]
    assert requests
    assert responses
    assert requests[0]["req_id"] == responses[0]["req_id"]
    assert requests[0].get("messages")
    assert isinstance(responses[0]["dt"], (int, float))
