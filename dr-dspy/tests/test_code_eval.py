from __future__ import annotations

import time

import dspy

from dr_dspy.code_eval import extract_dspy_code, run_python_check


def test_run_python_check_scores_pass_fail_and_errors() -> None:
    cases = [
        (
            "def add(a, b):\n    return a + b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            1.0,
            None,
        ),
        (
            "def add(a, b):\n    return a - b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "AssertionError",
        ),
        (
            "def add(a, b: return a+b\n",
            "def check(candidate):\n    assert candidate(1,2)==3\n",
            "add",
            0.0,
            "SyntaxError",
        ),
        (
            "def add(a, b):\n    raise ValueError('x')\n",
            "def check(candidate):\n    candidate(1,2)\n",
            "add",
            0.0,
            "ValueError",
        ),
    ]

    for code, test, entry_point, expected_score, expected_error in cases:
        result = run_python_check(
            code=code,
            test=test,
            entry_point=entry_point,
            timeout=5.0,
        )
        assert result.score == expected_score
        if expected_error is not None:
            assert result.error is not None
            assert expected_error in result.error


def test_run_python_check_timeout() -> None:
    start = time.time()
    result = run_python_check(
        code="def loop():\n    while True: pass\n",
        test="def check(candidate):\n    candidate()\n",
        entry_point="loop",
        timeout=2.0,
        cpu_limit_seconds=3,
    )
    wall = time.time() - start

    assert result.score == 0.0
    assert result.error is not None
    assert "timeout" in result.error
    assert wall < 6.0


def test_extract_dspy_code_handles_common_prediction_shapes() -> None:
    prediction = dspy.Prediction(code="x = 1")
    assert extract_dspy_code(prediction) == "x = 1"

    class CodeObject:
        code = "def f():\n    return 1\n"

    class PredictionObject:
        code = CodeObject()

    assert extract_dspy_code(PredictionObject()) == "def f():\n    return 1\n"
