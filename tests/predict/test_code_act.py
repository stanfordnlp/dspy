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


def test_codeact_factory_creates_fresh_interpreter_per_sequential_call():
    factory = MockInterpreterFactory(responses=["", "2\n"])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")

    first = program(question="What is 1+1?")
    second = program(question="What is 1+1 again?")

    assert first.answer == second.answer == "2"
    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    for interpreter in factory.instances:
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


def test_codeact_propagates_terminal_interpreter_failure_and_shuts_down():
    factory = MockInterpreterFactory(responses=["", CodeInterpreterError("protocol corrupt")])
    program = CodeAct(BasicQA, tools=[add], interpreter_factory=factory)
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)

    with pytest.raises(CodeInterpreterError, match="protocol corrupt"):
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


def _simple_tool(x: int) -> int:
    return x


def test_codeact_truncate_trajectory_drops_oldest_iteration_boundary_aware():
    """CodeAct steps have a variable key count (1 on failure: observation_i; 2 on success:
    generated_code_i + code_output_i|observation_i), unlike ReAct's fixed 4 keys/step. Popping
    ReAct's fixed keys[:4] slice cuts across iteration boundaries and desynchronizes the
    trajectory. CodeAct must drop exactly the earliest iteration's keys."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    trajectory = {
        "generated_code_0": "print(1)",   # iter 0: success -> 2 keys
        "code_output_0": "1",
        "observation_1": "parse failure",  # iter 1: failure -> 1 key
        "generated_code_2": "print(2)",    # iter 2: execution error -> 2 keys
        "observation_2": "exec error",
    }
    out = program.truncate_trajectory(trajectory)

    # Oldest iteration (0) fully removed; every later iteration is preserved intact.
    assert "generated_code_0" not in out
    assert "code_output_0" not in out
    assert out["observation_1"] == "parse failure"
    assert out["generated_code_2"] == "print(2)"
    assert out["observation_2"] == "exec error"
    assert out is trajectory


def test_codeact_truncate_trajectory_parse_failure_only_steps():
    """Consecutive AdapterParseError iterations leave observation-only entries (one key each).
    The inherited ReAct truncation (fixed keys[:4] slice, <4 keys -> ValueError) could not trim
    such a trajectory; CodeAct's iteration-boundary truncation must drop the oldest step."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    trajectory = {"observation_0": "parse failure", "observation_1": "parse failure"}
    out = program.truncate_trajectory(trajectory)

    assert "observation_0" not in out
    assert out["observation_1"] == "parse failure"


def test_codeact_truncate_trajectory_single_iteration_raises():
    """A trajectory with only one iteration cannot be truncated (dropping it leaves no context),
    mirroring ReAct's single-tool-call guard."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    with pytest.raises(ValueError):
        program.truncate_trajectory({"generated_code_0": "print(1)", "code_output_0": "1"})


def test_codeact_truncate_trajectory_skips_non_indexed_keys():
    """The extractor parse retry adds a prompt-only ``parse_feedback`` key (no iteration index).
    Truncation must skip non-indexed keys when computing iteration indices (``int("feedback")``
    used to raise ValueError) and must never pop them: the feedback survives the trim."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    trajectory = {
        "observation_0": "parse failure",
        "observation_1": "parse failure",
        "parse_feedback": "fix your output",
    }
    out = program.truncate_trajectory(trajectory)

    assert "observation_0" not in out
    assert out["observation_1"] == "parse failure"
    assert out["parse_feedback"] == "fix your output"


def test_codeact_truncate_trajectory_only_feedback_and_single_iteration_raises():
    """A non-indexed key alone does not make a trajectory truncatable: with one real iteration
    plus ``parse_feedback``, the single-iteration guard must still raise."""
    program = CodeAct("question -> answer", tools=[_simple_tool])
    with pytest.raises(ValueError):
        program.truncate_trajectory({"observation_0": "parse failure", "parse_feedback": "fix"})
