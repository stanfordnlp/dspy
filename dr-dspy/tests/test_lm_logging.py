from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from typing import Any

import dspy

from dr_dspy.event_log import SQLiteWriter
from dr_dspy.lm_logging import LoggingCallableLM, LoggingLM


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


def test_logging_lm_aforward_uses_sync_forward(monkeypatch) -> None:
    lm = LoggingLM("openai/unit-test", log=lambda *_args, **_kwargs: None)
    calls: list[dict[str, Any]] = []

    def fake_forward(
        prompt: Any = None, messages: Any = None, **kwargs: Any
    ) -> str:
        calls.append(
            {"prompt": prompt, "messages": messages, "kwargs": dict(kwargs)}
        )
        return "sync-result"

    async def unexpected_litellm_async_path(
        _self: dspy.LM,
        prompt: Any = None,
        messages: Any = None,
        **kwargs: Any,
    ) -> str:
        raise AssertionError("LoggingLM.aforward reached dspy.LM.aforward")

    monkeypatch.setattr(lm, "forward", fake_forward)
    monkeypatch.setattr(dspy.LM, "aforward", unexpected_litellm_async_path)

    result = asyncio.run(
        lm.aforward(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
        )
    )

    assert result == "sync-result"
    assert calls == [
        {
            "prompt": None,
            "messages": [{"role": "user", "content": "hello"}],
            "kwargs": {"temperature": 0.0},
        }
    ]
