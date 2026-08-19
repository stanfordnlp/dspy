import contextvars
import os
import threading
import time

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def test_persistent_worker_is_separate_and_captures_stdout(capsys):
    with dspy.LocalInterpreter() as interpreter:
        assert interpreter.execute("import os\nos.getpid()") != os.getpid()
        interpreter.execute("remembered = 41")
        assert interpreter.execute("remembered + 1") == 42
        assert interpreter.execute("print('guest only')") == "guest only"
        assert capsys.readouterr().out == ""


def test_tools_and_typed_submit():
    fields = [{"name": "answer", "type": "str"}, {"name": "score", "type": "int"}]
    with dspy.LocalInterpreter(
        tools={"add": lambda *, left, right: left + right}, output_fields=fields
    ) as interpreter:
        assert interpreter.execute("add(left=19, right=23)") == 42
        result = interpreter.execute("SUBMIT(answer='yes', score=42)")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "yes", "score": 42}


@pytest.mark.parametrize("name", ["not-valid", "class", "SUBMIT", "__builtins__"])
def test_invalid_tool_names_fail_without_starting_or_consuming_session(name):
    with pytest.raises(ValueError, match="invalid names"):
        dspy.LocalInterpreter(tools={name: lambda: 1})

    interpreter = dspy.LocalInterpreter()
    interpreter.tools[name] = lambda: 1
    with pytest.raises(ValueError, match="invalid names"):
        interpreter.execute("1")
    assert interpreter._process is None

    interpreter.tools = {}
    assert interpreter.execute("6 * 7") == 42
    interpreter.shutdown()


@pytest.mark.parametrize(
    "output_fields",
    [
        [{"name": "class"}],
        [{"name": "not-valid"}],
        [{"name": "answer"}, {"name": "answer"}],
        [{"type": "str"}],
        [{"name": "answer", "default": object()}],
    ],
)
def test_invalid_output_fields_fail_without_consuming_session(output_fields):
    with pytest.raises(ValueError):
        dspy.LocalInterpreter(output_fields=output_fields)

    interpreter = dspy.LocalInterpreter()
    assert interpreter.execute("remembered = 41") is None
    process = interpreter._process
    interpreter.output_fields = output_fields
    with pytest.raises(ValueError):
        interpreter.execute("1")
    assert interpreter._process is process

    interpreter.output_fields = [{"name": "answer"}]
    assert interpreter.execute("remembered + 1") == 42
    assert interpreter.execute("SUBMIT(answer=42)").output == {"answer": 42}
    interpreter.shutdown()


def test_reserved_variable_names_fail_without_starting_or_consuming_session():
    interpreter = dspy.LocalInterpreter(tools={"add": lambda left, right: left + right})
    for variables in ({"class": 1}, {"SUBMIT": "shadowed"}, {"__builtins__": {}}, {"add": "shadowed"}):
        with pytest.raises(CodeInterpreterError, match="invalid names"):
            interpreter.execute("1", variables=variables)
        assert interpreter._process is None

    assert interpreter.execute("len([1])") == 1
    assert interpreter.execute("add(19, 23)") == 42
    assert interpreter.execute("SUBMIT(42)").output == {"output": 42}
    interpreter.shutdown()


def test_guest_cannot_corrupt_interpreter_owned_globals_across_executions():
    with dspy.LocalInterpreter(tools={"add": lambda left, right: left + right}) as interpreter:
        interpreter.execute("__builtins__['len'] = None\nSUBMIT = None\nadd = None")
        assert interpreter.execute("len([1])") == 1
        assert interpreter.execute("add(19, 23)") == 42
        assert interpreter.execute("SUBMIT(42)").output == {"output": 42}


def test_errors_are_recoverable_but_timeout_is_terminal():
    interpreter = dspy.LocalInterpreter(execution_timeout=0.1)
    with pytest.raises(SyntaxError):
        interpreter.execute("if")
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute("1 / 0")
    assert interpreter.execute("6 * 7") == 42
    with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
        interpreter.execute("import time\ntime.sleep(10)")
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")
    interpreter.shutdown()


def test_async_host_tool():
    async def add(*, left, right):
        return left + right

    with dspy.LocalInterpreter(tools={"add": add}) as interpreter:
        assert interpreter.execute("add(left=19, right=23)") == 42


def test_timed_async_host_tool_preserves_context():
    marker = contextvars.ContextVar("marker", default="missing")
    marker.set("visible")

    async def read_marker():
        return marker.get()

    with dspy.LocalInterpreter(tools={"read_marker": read_marker}, execution_timeout=1) as interpreter:
        assert interpreter.execute("read_marker()") == "visible"


def test_host_tool_within_execution_timeout():
    def slow():
        time.sleep(0.08)
        return 42

    with dspy.LocalInterpreter(tools={"slow": slow}, execution_timeout=0.2) as interpreter:
        assert interpreter.execute("slow()") == 42


def test_execution_timeout_does_not_wait_for_blocked_host_tool():
    tool_started = threading.Event()

    def blocked():
        tool_started.set()
        time.sleep(1)
        return 42

    interpreter = dspy.LocalInterpreter(tools={"blocked": blocked}, execution_timeout=0.1)
    started = time.monotonic()
    with pytest.raises(CodeInterpreterError, match="exceeded execution timeout"):
        interpreter.execute("blocked()")
    elapsed = time.monotonic() - started

    assert tool_started.is_set()
    assert elapsed < 0.5
    with pytest.raises(CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


def test_interpreter_callbacks():
    events = []

    class RecordingCallback(BaseCallback):
        def on_interpreter_execute_start(self, call_id, instance, inputs):
            events.append(("execute_start", inputs["code"]))

        def on_interpreter_execute_end(self, call_id, outputs, exception=None):
            events.append(("execute_end", outputs))

        def on_interpreter_tool_call_start(self, call_id, instance, inputs):
            events.append(("tool_start", inputs["tool_name"]))

        def on_interpreter_tool_call_end(self, call_id, outputs, exception=None):
            events.append(("tool_end", outputs))

    with dspy.LocalInterpreter(tools={"answer": lambda: 42}, callbacks=[RecordingCallback()]) as interpreter:
        assert interpreter.execute("answer()") == 42

    assert ("execute_start", "answer()") in events
    assert ("tool_start", "answer") in events
    assert ("tool_end", 42) in events
    assert ("execute_end", 42) in events


def test_rlm_uses_local_interpreter():
    calls = []

    def add(*, left: int, right: int) -> int:
        calls.append((left, right))
        return left + right

    class Actions(dspy.Predict):
        def __init__(self, signature):
            super().__init__(signature)
            self.actions = iter(
                [
                    dspy.Prediction(reasoning="calculate", code="total = add(left=19, right=23)"),
                    dspy.Prediction(reasoning="submit", code="SUBMIT(answer=str(total))"),
                ]
            )

        def forward(self, **kwargs):
            return next(self.actions)

    rlm = dspy.RLM(
        "question: str -> answer: str",
        max_iters=2,
        tools=[add],
        interpreter_factory=dspy.LocalInterpreter,
    )
    rlm.generate_action = Actions(rlm.generate_action.signature)

    assert rlm(question="What is 19 + 23?").answer == "42"
    assert calls == [(19, 23)]
    assert dspy.LocalInterpreter.execution_instructions in rlm.generate_action.signature.instructions


@pytest.mark.asyncio
async def test_async_rlm_uses_local_interpreter_with_async_tool():
    async def add(*, left: int, right: int) -> int:
        return left + right

    class Actions(dspy.Predict):
        def __init__(self, signature):
            super().__init__(signature)
            self.actions = iter(
                [
                    dspy.Prediction(reasoning="calculate", code="total = add(left=19, right=23)"),
                    dspy.Prediction(reasoning="submit", code="SUBMIT(answer=str(total))"),
                ]
            )

        async def aforward(self, **kwargs):
            return next(self.actions)

    rlm = dspy.RLM(
        "question: str -> answer: str",
        max_iters=2,
        tools=[add],
        interpreter_factory=dspy.LocalInterpreter,
    )
    rlm.generate_action = Actions(rlm.generate_action.signature)

    assert (await rlm.acall(question="What is 19 + 23?")).answer == "42"


def test_flex_uses_local_interpreter():
    class Signature(dspy.Signature):
        value: int = dspy.InputField()
        result: int = dspy.OutputField()

    module_src = """
class AddModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.Predict("value: int -> result: int")

    def forward(self, **inputs):
        predicted = self.solve(value=inputs["value"])
        return dspy.Prediction(result=predicted.result)
""".strip()

    program = dspy.Flex(Signature, interpreter_factory=dspy.LocalInterpreter)
    program._bind_code(module_src)
    with dspy.context(lm=DummyLM([{"result": 42}])):
        assert program(value=20).result == 42
