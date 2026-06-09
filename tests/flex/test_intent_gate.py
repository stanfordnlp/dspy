"""Tests for the intent-clarity gate that runs before Flex codegen.

The gate makes one LM call (the IntentClaritySignature judge) *before* the
codegen call, so a DummyLM queue is ``[verdict_dict, codegen_dict]`` for the
proceed paths, or just ``[verdict_dict]`` when the gate is expected to raise.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

import dspy
from dspy.flex import FlexIntentError, flex
from dspy.utils.dummies import DummyLM

CANNED_PREDICTORS = textwrap.dedent("""
    PREDICTORS = {
        "echo": dspy.Predict("q -> a"),
    }
""").strip()

CANNED_FORWARD = textwrap.dedent("""
    def forward(self, q):
        out = self.echo(q=q)
        return dspy.Prediction(a=out.a)
""").strip()

CODEGEN = {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}


def _verdict(verdict: str, concern: str = "", question: str = "") -> dict:
    return {"verdict": verdict, "concern": concern, "clarifying_question": question}


@pytest.fixture
def dspy_warnings(caplog):
    """caplog wired to dspy's logger, which sets ``propagate=False`` by default
    (so records never reach pytest's root handler without this)."""
    dspy_logger = logging.getLogger("dspy")
    previous = dspy_logger.propagate
    dspy_logger.propagate = True
    caplog.set_level(logging.WARNING, logger="dspy.flex.flex")
    try:
        yield caplog
    finally:
        dspy_logger.propagate = previous


def _echo_factory(tmp_path: Path, intent_check: str = "error"):
    @flex(persist_to=str(tmp_path / "echo.py"), intent_check=intent_check)
    class Echo(dspy.Signature):
        """Echo the question."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    return Echo


def test_clear_verdict_proceeds(tmp_path: Path) -> None:
    dspy.configure(lm=DummyLM([_verdict("clear"), CODEGEN]))
    program = _echo_factory(tmp_path)()
    assert program.forward_src is not None


def test_insufficient_verdict_raises_with_question(tmp_path: Path) -> None:
    dspy.configure(lm=DummyLM([_verdict("insufficient", "objective missing", "What should it output?")]))
    with pytest.raises(FlexIntentError) as ei:
        _echo_factory(tmp_path)()
    msg = str(ei.value)
    assert "What should it output?" in msg  # the concrete question
    assert 'intent_check="warn"' in msg  # the escape hatch is advertised


def test_insufficient_downgraded_when_intent_check_is_warn(tmp_path: Path, dspy_warnings) -> None:
    dspy.configure(lm=DummyLM([_verdict("insufficient", "vague", "What is X?"), CODEGEN]))
    program = _echo_factory(tmp_path, intent_check="warn")()
    assert program.forward_src is not None  # proceeded instead of raising
    assert any("insufficient" in r.getMessage() for r in dspy_warnings.records)


def test_underspecified_verdict_warns_and_proceeds(tmp_path: Path, dspy_warnings) -> None:
    dspy.configure(lm=DummyLM([_verdict("underspecified", "unit unclear", "Cents or dollars?"), CODEGEN]))
    program = _echo_factory(tmp_path)()  # default intent_check="error"
    assert program.forward_src is not None
    warnings = " ".join(r.getMessage() for r in dspy_warnings.records)
    assert "underspecified" in warnings
    assert "Cents or dollars?" in warnings


def test_unrecognized_verdict_degrades_to_clear(tmp_path: Path) -> None:
    """A garbled judge verdict must never block codegen — treat it as 'clear'."""
    dspy.configure(lm=DummyLM([_verdict("banana"), CODEGEN]))
    program = _echo_factory(tmp_path)()
    assert program.forward_src is not None


def test_off_skips_the_judge(tmp_path: Path) -> None:
    """With intent_check='off', a single codegen response is enough (no judge call)."""
    dspy.configure(lm=DummyLM([CODEGEN]))  # no verdict queued
    program = _echo_factory(tmp_path, intent_check="off")()
    assert program.forward_src is not None


def test_clean_reload_does_not_invoke_the_judge(tmp_path: Path) -> None:
    """A signature-unchanged reload must not run the gate (no generation happens)."""
    dspy.configure(lm=DummyLM([_verdict("clear"), CODEGEN]))
    _echo_factory(tmp_path)()  # first construction writes the file

    class _NoLM(DummyLM):
        def __init__(self) -> None:
            super().__init__([])

        def forward(self, *args, **kwargs):
            raise AssertionError("codegen LM must not be called on a clean reload")

    dspy.configure(lm=_NoLM())
    program = _echo_factory(tmp_path)()  # reload from disk — no gate, no codegen
    assert program.forward_src is not None
