"""Tests for IFEval dataset adapter and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import dspy
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ifeval_metrics import ifeval_constraint_metric, ifeval_metric
from data_adapters.ifeval import IFEvalAdapter


@pytest.fixture(scope="module")
def sample_example() -> dspy.Example:
    return dspy.Example(
        key=1000,
        prompt="Write exactly 2 paragraphs. Separate paragraphs with ***.",
        instruction_id_list=["length_constraints:number_paragraphs"],
        kwargs=[{"num_paragraphs": 2}],
    ).with_inputs("prompt")


def test_ifeval_metric_passes_valid_response(sample_example: dspy.Example) -> None:
    response = "First paragraph with enough content.\n\n***\n\nSecond paragraph here."
    pred = dspy.Prediction(response=response)
    assert ifeval_metric(sample_example, pred) == 1.0
    assert ifeval_constraint_metric(sample_example, pred) == 1.0


def test_ifeval_metric_fails_invalid_response(sample_example: dspy.Example) -> None:
    pred = dspy.Prediction(response="Only one paragraph.")
    assert ifeval_metric(sample_example, pred) == 0.0
    assert ifeval_constraint_metric(sample_example, pred) == 0.0


def test_ifeval_metric_filters_hf_style_kwargs() -> None:
    """HF IFEval rows include a full kwargs template; only relevant keys should be used."""
    example = dspy.Example(
        key=3453,
        prompt="Summarize Japan. Italicize at least 5 keywords with *asterisks*.",
        instruction_id_list=["detectable_format:number_highlighted_sections"],
        kwargs=[
            {
                "num_highlights": 5,
                "relation": None,
                "num_words": None,
                "num_placeholders": None,
                "prompt_to_repeat": None,
            }
        ],
    ).with_inputs("prompt")
    response = "Japan has a *rich* *ancient* *culture*, *modern* cities, and *traditions*."
    pred = dspy.Prediction(response=response)
    assert ifeval_metric(example, pred) == 1.0


def test_ifeval_adapter_loads_small_split() -> None:
  adapter = IFEvalAdapter(
      {
          "train_size": 4,
          "dev_size": 2,
          "test_size": 2,
          "split_seed": 1,
      }
  )
  train_set, val_set, test_set = adapter.load_dataset()
  assert len(train_set) == 4
  assert len(val_set) == 2
  assert len(test_set) == 2
  assert train_set[0].prompt
  assert train_set[0].instruction_id_list
