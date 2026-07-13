import threading
from unittest.mock import Mock, patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils import DummyLM
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class StaticPredictor:
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, **kwargs):
        return dspy.Prediction(**self.fields)


class RecordingPythonInterpreterFactory:
    def __init__(self, parties: int):
        self.instances = []
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(parties)

    def __call__(self):
        interpreter = PythonInterpreter()
        with self._lock:
            self.instances.append(interpreter)
        self._barrier.wait(timeout=30)
        return interpreter


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
def test_pot_evaluate_creates_one_interpreter_per_example():
    factory = RecordingPythonInterpreterFactory(parties=4)
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")
    devset = [
        dspy.Example(question=f"What is 1+1? ({index})", answer="2").with_inputs("question")
        for index in range(4)
    ]

    result = dspy.Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,
        display_progress=False,
    )(pot)

    assert result.score == 100.0
    assert len(factory.instances) == 4
    assert len({id(interpreter) for interpreter in factory.instances}) == 4
    assert all(interpreter.deno_process is None for interpreter in factory.instances)


def test_pot_allows_interpreter_as_signature_input():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "CPython"})])
    pot = ProgramOfThought("interpreter -> answer", interpreter_factory=factory)
    pot.code_generate = Mock(return_value=dspy.Prediction(generated_code="SUBMIT({'answer': interpreter})"))
    pot.generate_output = StaticPredictor(answer="CPython")

    result = pot(interpreter="CPython")

    assert result.answer == "CPython"
    pot.code_generate.assert_called_once_with(interpreter="CPython")


def test_pot_rejects_keyword_interpreter_override():
    factory = MockInterpreterFactory()
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)

    with pytest.raises(TypeError, match="first positional argument"):
        pot(question="What is 1+1?", interpreter=MockInterpreter())

    assert factory.instances == []


def test_pot_does_not_shutdown_caller_owned_interpreter():
    factory = MockInterpreterFactory()
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")
    interpreter = MockInterpreter(responses=[FinalOutput({"answer": "2"})])

    try:
        result = pot.forward(interpreter, question="What is 1+1?")

        assert result.answer == "2"
        assert factory.instances == []
        assert interpreter.execute("print('still open')") == ""
    finally:
        interpreter.shutdown()


def test_pot_shuts_down_factory_interpreter_when_execution_raises():
    factory = MockInterpreterFactory(responses=[ValueError("unexpected interpreter failure")])
    pot = ProgramOfThought(BasicQA, interpreter_factory=factory)
    pot.code_generate = StaticPredictor(generated_code="raise ValueError")

    with pytest.raises(ValueError, match="unexpected interpreter failure"):
        pot(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


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
