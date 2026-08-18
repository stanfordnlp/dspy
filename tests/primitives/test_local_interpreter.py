import sys

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreter, CodeInterpreterError, FinalOutput


def test_public_protocol_and_execution_instructions():
    interpreter = dspy.LocalInterpreter()
    try:
        assert isinstance(interpreter, CodeInterpreter)
        assert dspy.LocalInterpreter.execution_instructions
    finally:
        interpreter.shutdown()


def test_persistent_state_variables_and_output():
    with dspy.LocalInterpreter() as interpreter:
        assert interpreter.execute("value = 40") is None
        assert interpreter.execute("value + increment", {"increment": 2}) == 42
        assert interpreter.execute("print('hello')") == "hello"


def test_rejects_invalid_variable_names():
    with dspy.LocalInterpreter() as interpreter:
        with pytest.raises(CodeInterpreterError, match="valid Python identifiers"):
            interpreter.execute("pass", {"for": 1})


def test_syntax_and_runtime_errors_are_recoverable():
    with dspy.LocalInterpreter() as interpreter:
        with pytest.raises(SyntaxError):
            interpreter.execute("if")
        with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
            interpreter.execute("1 / 0")
        assert interpreter.execute("6 * 7") == 42


def test_tools_refresh_between_executions():
    with dspy.LocalInterpreter(tools={"old_tool": lambda: 1}) as interpreter:
        assert interpreter.execute("old_tool()") == 1
        interpreter.tools.clear()
        interpreter.tools["new_tool"] = lambda: 2
        assert interpreter.execute("new_tool()") == 2
        with pytest.raises(CodeExecutionError, match="old_tool"):
            interpreter.execute("old_tool()")


def test_untyped_and_typed_submit():
    with dspy.LocalInterpreter() as interpreter:
        result = interpreter.execute("SUBMIT(42)")
        assert result == FinalOutput({"output": 42})

    fields = [{"name": "answer", "type": "str"}, {"name": "score", "type": "int"}]
    with dspy.LocalInterpreter(output_fields=fields) as interpreter:
        result = interpreter.execute("SUBMIT(answer='yes', score=42)")
        assert result == FinalOutput({"answer": "yes", "score": 42})
        with pytest.raises(CodeExecutionError, match="missing"):
            interpreter.execute("SUBMIT(answer='no')")


def test_submit_stops_execution():
    calls = []

    def after_submit():
        calls.append(True)

    with dspy.LocalInterpreter(tools={"after_submit": after_submit}) as interpreter:
        assert interpreter.execute("SUBMIT(42)\nafter_submit()") == FinalOutput({"output": 42})
        assert calls == []


def test_shutdown_is_terminal_and_idempotent():
    interpreter = dspy.LocalInterpreter()
    interpreter.start()
    interpreter.start()
    interpreter.shutdown()
    interpreter.shutdown()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.start()
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


def test_execution_restores_host_dspy_module():
    host_module = sys.modules["dspy"]
    with dspy.LocalInterpreter() as interpreter:
        interpreter.execute("import sys, types\nsys.modules['dspy'] = types.ModuleType('dspy')")
    assert sys.modules["dspy"] is host_module


def test_rlm_uses_local_interpreter_factory():
    calls = []

    def add(*, left: int, right: int) -> int:
        calls.append((left, right))
        return left + right

    class Actions(dspy.Predict):
        def __init__(self, signature):
            super().__init__(signature)
            self.responses = iter(
                [
                    dspy.Prediction(reasoning="compute", code="answer = add(left=20, right=22)"),
                    dspy.Prediction(reasoning="submit", code="SUBMIT(answer=str(answer))"),
                ]
            )

        def forward(self, **kwargs):
            return next(self.responses)

    rlm = dspy.RLM("question: str -> answer: str", max_iters=2, tools=[add], interpreter_factory=dspy.LocalInterpreter)
    rlm.generate_action = Actions(rlm.generate_action.signature)

    assert rlm(question="What is 20 + 22?").answer == "42"
    assert calls == [(20, 22)]
    assert dspy.LocalInterpreter.execution_instructions in rlm.generate_action.signature.instructions
