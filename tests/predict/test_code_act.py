import threading
from unittest.mock import Mock

import pytest

import dspy
from dspy import Signature
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import CodeAct
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.utils import DummyLM
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory

pytestmark = pytest.mark.deno


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class StaticPredictor:
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, **kwargs):
        return dspy.Prediction(**self.fields)


class RaisingPredictor:
    def __call__(self, **kwargs):
        raise ValueError("unexpected extractor failure")

def add(a: float, b: float) -> float:
    "add two numbers"
    return a + b

def test_codeact_code_generation():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "code_output_0": '"2\\n"',
        "generated_code_0": "result = add(1,1)\nprint(result)",
    }


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")

def extract_maximum_minimum(input_list: str) -> dict[str, float]:
    numbers = list(map(float, input_list.split(",")))
    return {"maximum": max(numbers), "minimum": min(numbers)}

def test_codeact_support_multiple_fields():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(ExtremumFinder, tools=[extract_maximum_minimum])
    res = program(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert res.trajectory == {
        "code_output_0": '"{\'maximum\': 6.0, \'minimum\': 2.0}\\n"',
        "generated_code_0": "result = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)",
    }


def test_codeact_code_parse_failure():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nparse(error\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "parse(error",
        "observation_0": "Failed to execute the generated code: Invalid Python syntax. message: ",
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }


def test_codeact_code_execution_failure():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nunknown+1\n```",
                "finished": False,
            },
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        "generated_code_0": "unknown+1",
        "observation_0": 'Failed to execute the generated code: NameError: ["name \'unknown\' is not defined"]',
        "generated_code_1": "result = add(1,1)\nprint(result)",
        "code_output_1": '"2\\n"',
    }


def test_codeact_evaluate_creates_one_interpreter_per_example():
    tool_registration_barrier = threading.Barrier(4)

    def execute(code, variables):
        if code.startswith("def add"):
            tool_registration_barrier.wait(timeout=30)
            return ""
        return "2\n"

    factory = MockInterpreterFactory(execute_fn=execute)
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")
    devset = [
        dspy.Example(question=f"What is 1+1? ({index})", answer="2").with_inputs("question")
        for index in range(4)
    ]

    result = dspy.Evaluate(
        devset=devset,
        metric=answer_exact_match,
        num_threads=4,
        display_progress=False,
    )(program)

    assert result.score == 100.0
    assert len(factory.instances) == 4
    assert len({id(interpreter) for interpreter in factory.instances}) == 4
    for interpreter in factory.instances:
        assert interpreter.call_count == 2
        with pytest.raises(CodeInterpreterError, match="shutdown"):
            interpreter.execute("print('closed')")


def test_codeact_allows_interpreter_as_signature_input():
    factory = MockInterpreterFactory(responses=["", "CPython\n"])
    program = CodeAct("interpreter -> answer", tools=[add], interpreter_factory=factory)
    program.codeact = Mock(
        return_value=dspy.Prediction(generated_code="print(interpreter)", finished=True)
    )
    program.extractor = StaticPredictor(answer="CPython")

    result = program(interpreter="CPython")

    assert result.answer == "CPython"
    assert program.codeact.call_count == 1
    assert program.codeact.call_args.kwargs["interpreter"] == "CPython"


def test_codeact_rejects_keyword_interpreter_override():
    factory = MockInterpreterFactory()
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)

    with pytest.raises(TypeError, match="first positional argument"):
        program(question="What is 1+1?", interpreter=MockInterpreter())

    assert factory.instances == []


def test_codeact_rejects_removed_constructor_interpreter_keyword():
    with pytest.raises(TypeError, match="unexpected keyword argument 'interpreter'"):
        CodeAct(BasicQA, tools=[add], interpreter=MockInterpreter())


def test_codeact_does_not_shutdown_caller_owned_interpreter():
    factory = MockInterpreterFactory()
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")
    interpreter = MockInterpreter(responses=["", "2\n"])

    try:
        result = program(interpreter, question="What is 1+1?")

        assert result.answer == "2"
        assert factory.instances == []
        assert interpreter.execute("print('still open')") == ""
    finally:
        interpreter.shutdown()


def test_codeact_shuts_down_factory_interpreter_when_extractor_raises():
    factory = MockInterpreterFactory(responses=["", "2\n"])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = RaisingPredictor()

    with pytest.raises(ValueError, match="unexpected extractor failure"):
        program(question="What is 1+1?")

    assert len(factory.instances) == 1
    with pytest.raises(CodeInterpreterError, match="shutdown"):
        factory.instances[0].execute("print('closed')")


class CustomTool:
    def __call__(self, a: float, b: float) -> float:
        return a + b

def test_codeact_tool_validation():
    with pytest.raises(ValueError, match=r"CodeAct only accepts functions and not callable objects."):
        CodeAct(BasicQA, tools=[CustomTool()])
