import threading
from unittest.mock import MagicMock, patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.primitives.code_interpreter import FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
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
    assert pot.interpreter.deno_process is None


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
    assert pot.interpreter.deno_process is None


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
    assert pot.interpreter.deno_process is None


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
    assert pot.interpreter.deno_process is None


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


def test_pot_creates_fresh_interpreter_per_forward():
    """Each forward() creates an independent PythonInterpreter (fixes thread-safety for num_threads > 1)."""
    lm = DummyLM(
        [
            {"reasoning": "R1", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "R2", "answer": "2"},
            {"reasoning": "R3", "generated_code": "```python\nresult = 2+2\n```"},
            {"reasoning": "R4", "answer": "4"},
        ]
    )
    dspy.configure(lm=lm)

    created_interpreters = []

    def make_mock_interp():
        m = MagicMock(spec=PythonInterpreter)
        m.execute.return_value = FinalOutput({"answer": "2"})
        created_interpreters.append(m)
        return m

    with patch("dspy.predict.program_of_thought.PythonInterpreter", side_effect=make_mock_interp):
        pot = ProgramOfThought(BasicQA)
        assert pot.interpreter is None  # no interpreter allocated until forward()

        pot(question="What is 1+1?")
        assert len(created_interpreters) == 1
        assert created_interpreters[0].shutdown.called

        pot(question="What is 2+2?")
        assert len(created_interpreters) == 2  # second call gets a new interpreter
        assert created_interpreters[1].shutdown.called


def test_pot_thread_safe_concurrent_forward():
    """Concurrent forward() calls each get an independent interpreter (issue #9082)."""
    errors: list[str] = []
    created_interpreters: list[MagicMock] = []

    def make_mock_interp():
        m = MagicMock(spec=PythonInterpreter)
        m.execute.return_value = FinalOutput({"answer": "1"})
        created_interpreters.append(m)
        return m

    with patch("dspy.predict.program_of_thought.PythonInterpreter", side_effect=make_mock_interp):
        pot = ProgramOfThought(BasicQA)

        def run():
            # Each thread uses its own DummyLM to avoid shared-state contention
            thread_lm = DummyLM(
                [
                    {"reasoning": "R", "generated_code": "```python\nresult = 1\n```"},
                    {"reasoning": "R", "answer": "1"},
                ]
            )
            try:
                with dspy.context(lm=thread_lm):
                    pot(question="What is 1+1?")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"Errors in concurrent forward(): {errors}"
    assert len(created_interpreters) == 3  # one interpreter per thread
    for m in created_interpreters:
        assert m.shutdown.called  # each interpreter was properly closed


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
