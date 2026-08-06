from typing import Any

import pytest

import dspy
from dspy import Signature
from dspy.predict.per_instance_synthesis import PIPS
from dspy.primitives.python_interpreter import InterpreterError
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


@pytest.fixture
def stub_python_interpreter(monkeypatch):
    class StubPythonInterpreter:
        scheduled_results: list[Any] = []
        instances: list["StubPythonInterpreter"] = []

        def __init__(self, *_, **__):
            self.calls: list[dict[str, Any]] = []
            self.shutdown_called = False
            type(self).instances.append(self)

        def __call__(self, code: str, variables: dict[str, Any] | None = None):
            self.calls.append({"code": code, "variables": variables})
            if not type(self).scheduled_results:
                return None
            result = type(self).scheduled_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr("dspy.predict.per_instance_synthesis.PythonInterpreter", StubPythonInterpreter)
    StubPythonInterpreter.scheduled_results = []
    StubPythonInterpreter.instances = []
    return StubPythonInterpreter


def test_pips_code_generation(stub_python_interpreter):
    stub_python_interpreter.scheduled_results = [2]
    lm = DummyLM(
        [
            {"reasoning": "ModeSelection", "probability": "0.9"},
            {
                "symbols": '{"numbers": [1, 1]}',
                "code": "def solve(symbols):\n    return sum(symbols['numbers'])",
            },
            {"feedback": "", "passed": True},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = PIPS(BasicQA)

    res = program(question="What is 1+1?")

    assert res.answer == "2"
    assert res.pips_result.mode == "code"
    assert res.pips_result.symbols == {"numbers": [1, 1]}
    assert stub_python_interpreter.instances and stub_python_interpreter.instances[0].shutdown_called


def test_pips_symbol_parse_failure(stub_python_interpreter):
    stub_python_interpreter.scheduled_results = [3]
    lm = DummyLM(
        [
            {"reasoning": "ModeSelection", "probability": "0.95"},
            {"symbols": "not-json", "code": "def solve(symbols):\n    return symbols['answer']"},
            {
                "symbols": '{"answer": 3}',
                "code": "def solve(symbols):\n    return symbols['answer']",
            },
            {"feedback": "", "passed": True},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = PIPS(BasicQA, max_iters=2)

    res = program(question="How many apples?")

    assert res.answer == "3"
    assert res.pips_result.attempts == 2
    assert res.pips_result.symbols == {"answer": 3}


def test_pips_runtime_failure(stub_python_interpreter):
    stub_python_interpreter.scheduled_results = [InterpreterError("boom"), 5]
    lm = DummyLM(
        [
            {"reasoning": "ModeSelection", "probability": "0.8"},
            {
                "symbols": '{"answer": 0}',
                "code": "def solve(symbols):\n    raise ValueError('oops')",
            },
            {"feedback": "Fix the bug", "passed": False},
            {
                "symbols": '{"answer": 5}',
                "code": "def solve(symbols):\n    return symbols['answer']",
            },
            {"feedback": "", "passed": True},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = PIPS(BasicQA, max_iters=2)

    res = program(question="Return five")

    assert res.answer == "5"
    assert res.pips_result.attempts == 2
    assert len(stub_python_interpreter.instances) == 2


def test_pips_falls_back_to_cot(stub_python_interpreter):
    lm = DummyLM(
        [
            {"reasoning": "ModeSelection", "probability": "0.1"},
            {"reasoning": "Cot path", "answer": "two"},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = PIPS(BasicQA)

    res = program(question="What number comes after one?")

    assert res.answer == "two"
    assert res.pips_result.mode == "cot"
    assert stub_python_interpreter.instances == []
