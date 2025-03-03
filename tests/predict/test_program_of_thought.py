from unittest.mock import patch
import pytest
import shutil

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM

# This test suite requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
is_deno_available = shutil.which("deno") is not None

class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


@pytest.mark.skipif(not is_deno_available, reason="Deno is not installed or not in PATH")
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
    assert pot.interpreter.deno_process is None


@pytest.mark.skipif(not is_deno_available, reason="Deno is not installed or not in PATH")
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
    assert pot.interpreter.deno_process is None


@pytest.mark.skipif(not is_deno_available, reason="Deno is not installed or not in PATH")
def test_pot_code_generation_persistent_errors():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+0/0\n```"},
        ] * max_iters
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
        pot(question="What is 1+1?")
        assert pot.interpreter.deno_process is None


def test_pot_code_parse_error():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"},
        ] * max_iters
    )
    dspy.settings.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with patch("dspy.predict.program_of_thought.ProgramOfThought._execute_code") as mock_execute_code, pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()
