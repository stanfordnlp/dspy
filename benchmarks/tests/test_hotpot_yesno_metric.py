"""Regression test for HotPotQA verbose yes/no scoring."""

import sys
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics import hotpotqa_metric


def test_verbose_no_matches_gold_no() -> None:
    example = dspy.Example(question="Is X true?", answer="no").with_inputs("question")
    pred = dspy.Prediction(answer="No, both individuals are not film producers.")
    assert hotpotqa_metric(example, pred) == 1.0


def test_verbose_yes_matches_gold_yes() -> None:
    example = dspy.Example(question="Is X true?", answer="yes").with_inputs("question")
    pred = dspy.Prediction(answer="Yes, that is correct based on the context.")
    assert hotpotqa_metric(example, pred) == 1.0
