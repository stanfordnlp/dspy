from __future__ import annotations

import sqlite3

import dspy
from dspy.teleprompt import BootstrapFewShot

from dr_dspy.event_log import EventStore, SQLiteWriter


def test_bootstrap_selects_passing_demos(
    humaneval_harness,
    humaneval_mock_harness,
    tmp_path,
) -> None:
    train, _ = humaneval_mock_harness.make_mock_dataset()
    writer = SQLiteWriter(str(tmp_path / "runs.db"), run_id="bootstrap-test")
    metric = humaneval_harness.HumanEvalMetric(writer=writer, timeout=5.0)
    try:
        lm = humaneval_mock_harness.build_mock_lm(
            humaneval_mock_harness.mock_solver,
            log=writer.put_event,
        )
        dspy.configure(lm=lm, callbacks=[])
        student = dspy.Predict(humaneval_harness.Solve)
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=0,
        )
        compiled = optimizer.compile(student=student, trainset=train)
    finally:
        writer.close()

    predictors = compiled.named_predictors()
    assert len(predictors) == 1
    _, predictor = predictors[0]
    demos = list(predictor.demos or [])
    assert 1 <= len(demos) <= 4
    train_prompts = {example.prompt for example in train}
    assert {demo.prompt for demo in demos} <= train_prompts
    assert len(lm.calls) >= len(train)


def test_full_harness_smoke(
    humaneval_harness,
    humaneval_mock_harness,
    tmp_path,
) -> None:
    db_path = tmp_path / "smoke.db"
    compiled_path = tmp_path / "compiled.json"
    config = humaneval_harness.HarnessConfig(
        event_store=EventStore.SQLITE,
        db_path=str(db_path),
        database_url=None,
        compiled_path=str(compiled_path),
        model="callable/mock",
        seed=0,
        train_size=4,
        dev_size=4,
        num_threads=2,
        timeout=5.0,
        max_bootstrapped_demos=4,
        max_labeled_demos=0,
    )

    rc = humaneval_mock_harness.run_mock_humaneval_bootstrap(config)

    assert rc == 0
    assert db_path.exists()
    assert compiled_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        types_present = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT event_type FROM events"
            ).fetchall()
        }
        flows = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT flow FROM events WHERE event_type='flow.start'"
            ).fetchall()
        }
        row_count, min_payload_is_null = conn.execute(
            "SELECT COUNT(*), MIN(payload IS NULL) FROM events"
        ).fetchone()
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
    assert not required - types_present
    assert {"eval_baseline", "optimize", "eval_optimized"} <= flows
    assert row_count > 0
    assert min_payload_is_null in (0, None)

    fresh = dspy.Predict(humaneval_harness.Solve)
    fresh.load(str(compiled_path))
