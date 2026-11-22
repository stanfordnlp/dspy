"""Test that COPRO.compile handles eval_kwargs with constructor-only Evaluate params."""

import dspy
from dspy import Example
from dspy.teleprompt import COPRO
from dspy.utils.dummies import DummyLM


def test_copro_compile_with_max_errors():
    """max_errors is an Evaluate constructor param, not a __call__ param.

    Before the fix, passing it via eval_kwargs crashed with:
        TypeError: Evaluate.__call__() got an unexpected keyword argument 'max_errors'
    """
    lm = DummyLM([{"proposed_instruction": "answer the question", "proposed_prefix_for_output_field": "Output:"}])
    dspy.configure(lm=lm)

    trainset = [Example(input="sky color?", output="blue").with_inputs("input")]

    copro = COPRO(metric=lambda ex, pred: ex.output == pred.output, breadth=2, depth=1)
    compiled = copro.compile(
        dspy.ChainOfThought("input -> output"),
        trainset=trainset,
        eval_kwargs={"num_threads": 1, "max_errors": 5},
    )
    assert compiled is not None


def test_copro_compile_with_provide_traceback():
    """provide_traceback is also constructor-only."""
    lm = DummyLM([{"proposed_instruction": "answer", "proposed_prefix_for_output_field": "Out:"}])
    dspy.configure(lm=lm)

    trainset = [Example(input="1+1?", output="2").with_inputs("input")]

    copro = COPRO(metric=lambda ex, pred: ex.output == pred.output, breadth=2, depth=1)
    compiled = copro.compile(
        dspy.ChainOfThought("input -> output"),
        trainset=trainset,
        eval_kwargs={"num_threads": 1, "provide_traceback": True, "failure_score": 0.0},
    )
    assert compiled is not None


def test_copro_compile_call_only_kwargs_still_work():
    """Ensure call-compatible kwargs like display_progress still reach __call__."""
    lm = DummyLM([{"proposed_instruction": "do it", "proposed_prefix_for_output_field": "Result:"}])
    dspy.configure(lm=lm)

    trainset = [Example(input="hi", output="hello").with_inputs("input")]

    copro = COPRO(metric=lambda ex, pred: ex.output == pred.output, breadth=2, depth=1)
    compiled = copro.compile(
        dspy.ChainOfThought("input -> output"),
        trainset=trainset,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )
    assert compiled is not None
