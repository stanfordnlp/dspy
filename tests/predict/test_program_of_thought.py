from unittest.mock import patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


@pytest.mark.deno
def test_pot_code_generation():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


# This test ensures the old finetuned saved models still work
@pytest.mark.deno
def test_old_style_pot():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")


@pytest.mark.deno
def test_pot_support_multiple_fields():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nmaximum = 6\nminimum = 2\nSUBMIT({'maximum': maximum, 'minimum': minimum})\n```",
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(ExtremumFinder)
    res = pot(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"


@pytest.mark.deno
def test_pot_code_generation_with_one_error():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
            {
                "reasoning": "Reason_B",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


@pytest.mark.deno
def test_pot_code_generation_persistent_errors():
    max_iters = 3
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
        ]
        * max_iters
    )
    dspy.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
        pot(question="What is 1+1?")


def test_pot_code_parse_error():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"},
        ]
        * max_iters
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with (
        patch("dspy.predict.program_of_thought.ProgramOfThought._execute_code") as mock_execute_code,
        pytest.raises(
            RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."
        ),
    ):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()


@pytest.mark.deno
def test_pot_thread_safety_with_evaluate():
    """Regression test for #9082: ProgramOfThought should work with num_threads > 1."""
    num_examples = 8
    # DummyLM needs to return pairs of responses (code gen + answer extraction) for each example
    lm_responses = [
        {
            "reasoning": "Reason_A",
            "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
        },
        {"reasoning": "Reason_B", "answer": "2"},
    ] * num_examples
    lm = DummyLM(lm_responses)
    dspy.configure(lm=lm)

    pot = ProgramOfThought(BasicQA)
    devset = [dspy.Example(question="What is 1+1?", answer="2").with_inputs("question") for _ in range(num_examples)]

    evaluate = dspy.Evaluate(devset=devset, metric=lambda example, pred, trace=None: pred.answer == example.answer, num_threads=4)
    result = evaluate(pot)
    assert result >= 0  # Should complete without "I/O operation on closed file" errors
