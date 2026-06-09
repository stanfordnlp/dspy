"""Tests that a generated Flex module may be fully deterministic.

Covers an empty ``PREDICTORS = {}`` plus a pure-Python ``forward`` with a nested
helper: it must bind, run with no LM call, expose no predictors, and survive the
persistence round-trip.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import dspy
from dspy.flex import flex
from dspy.utils.dummies import DummyLM

EMPTY_PREDICTORS = "PREDICTORS = {}"

# A pure-Python forward with a nested helper and a blank line (exercises the
# indent/dedent round-trip in persistence.py).
DETERMINISTIC_FORWARD = textwrap.dedent("""
    def forward(self, value):
        def double(x):
            return x * 2

        return dspy.Prediction(result=double(value))
""").strip()


def _doubler_factory(persist_to: Path):
    @flex(persist_to=str(persist_to), intent_check="off")
    class Doubler(dspy.Signature):
        """Return double the input value."""

        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    return Doubler


def test_empty_predictors_binds_and_runs_without_an_lm(tmp_path: Path) -> None:
    # One codegen response produces the deterministic implementation; after that
    # NO LM is configured, proving forward() makes no LM calls.
    dspy.configure(lm=DummyLM([{"predictors_src": EMPTY_PREDICTORS, "forward_src": DETERMINISTIC_FORWARD}]))
    program = _doubler_factory(tmp_path / "doubler.py")()

    assert program.named_predictors() == []  # no predictors attached
    result = program(value=21)
    assert result.result == 42


def test_deterministic_module_roundtrips_through_disk(tmp_path: Path) -> None:
    persist_path = tmp_path / "doubler.py"
    dspy.configure(lm=DummyLM([{"predictors_src": EMPTY_PREDICTORS, "forward_src": DETERMINISTIC_FORWARD}]))
    _doubler_factory(persist_path)()

    text = persist_path.read_text()
    assert "PREDICTORS = {}" in text
    assert "def double(x):" in text  # nested helper survived rendering

    # Reconstruct with no LM at all — must load the deterministic body from disk
    # and run identically.
    dspy.configure(lm=DummyLM([]))
    program2 = _doubler_factory(persist_path)()
    assert program2(value=5).result == 10
    assert program2.named_predictors() == []
