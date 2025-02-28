from unittest.mock import patch
import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM

# This test suite requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def test_pot_code_generation():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


def test_pot_code_generation_with_one_error():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\n```"},
            {"reasoning": "Reason_B", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"


def test_pot_code_generation_persistent_errors():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\n```"},
        ] * max_iters
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought. Error message: Sandbox Error:"):
        pot(question="What is 1+1?")


def test_pot_code_parse_error():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"},
        ] * max_iters
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with patch("dspy.predict.program_of_thought.ProgramOfThought.execute_code") as mock_execute_code, pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought. Error message: Code format is not correct."):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()