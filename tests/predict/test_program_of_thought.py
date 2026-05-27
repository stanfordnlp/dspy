from unittest.mock import patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


@pytest.mark.deno
def test_pot_real_interpreter_cases():
    with PythonInterpreter() as interpreter:
        _assert_pot_code_generation(interpreter)
        _assert_old_style_pot(interpreter)
        _assert_pot_support_multiple_fields(interpreter)
        _assert_pot_code_generation_with_one_error(interpreter)
        _assert_pot_code_generation_persistent_errors(interpreter)


def _assert_pot_code_generation(interpreter):
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
    pot = ProgramOfThought(BasicQA, interpreter=interpreter)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is not None


# This test ensures the old finetuned saved models still work
def _assert_old_style_pot(interpreter):
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA, interpreter=interpreter)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is not None


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")


def _assert_pot_support_multiple_fields(interpreter):
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
    pot = ProgramOfThought(ExtremumFinder, interpreter=interpreter)
    res = pot(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert pot.interpreter.deno_process is not None


def _assert_pot_code_generation_with_one_error(interpreter):
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
    pot = ProgramOfThought(BasicQA, interpreter=interpreter)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is not None


def _assert_pot_code_generation_persistent_errors(interpreter):
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

    pot = ProgramOfThought(BasicQA, max_iters=max_iters, interpreter=interpreter)
    with pytest.raises(RuntimeError, match=r"Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
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
            RuntimeError, match=r"Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."
        ),
    ):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()
