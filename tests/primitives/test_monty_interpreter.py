import pytest

import dspy

pytest.importorskip("pydantic_monty")


@pytest.fixture
def interpreter():
    interpreter = dspy.MontyInterpreter(output_fields=[{"name": "answer"}])
    yield interpreter
    interpreter.shutdown()


def test_state_output_and_recovery(interpreter):
    assert interpreter.execute("values = [3, 8]\ndef total(): return sum(values)") is None
    with pytest.raises(dspy.CodeExecutionError, match="ZeroDivisionError"):
        interpreter.execute("1 / 0")
    assert interpreter.execute("print('total:', total())") == "total: 11"
    with pytest.raises(SyntaxError):
        interpreter.execute("if :")
    assert interpreter.execute("total()") == 11


def test_submit_stops_and_allows_another_iteration(interpreter):
    calls = []
    interpreter.tools["record"] = lambda: calls.append(True)
    result = interpreter.execute("value = 13\nSUBMIT(answer=value)\nrecord()")
    assert result == dspy.FinalOutput({"answer": 13})
    assert calls == []
    assert interpreter.execute("SUBMIT(answer=value + 6)") == dspy.FinalOutput({"answer": 19})
    with pytest.raises(dspy.CodeExecutionError, match="fields"):
        interpreter.execute("SUBMIT(wrong=0)")
    assert interpreter.execute("value") == 13


def test_tools_can_change_and_recover_from_errors(interpreter):
    interpreter.tools["tool"] = lambda x, scale=3: x * scale
    assert interpreter.execute("tool(7, scale=4)") == 28
    interpreter.tools["tool"] = lambda x, scale=3: x + scale
    assert interpreter.execute("tool(7)") == 10
    interpreter.tools["tool"] = lambda: 1 / 0
    assert interpreter.execute("try:\n    tool()\nexcept Exception:\n    print('recovered')") == "recovered"


@pytest.mark.asyncio
async def test_async_host_tool_in_running_loop(interpreter):
    async def tool(value: int) -> int:
        return value * 3

    interpreter.tools["tool"] = tool
    assert interpreter.execute("tool(7)") == 21


def test_shutdown_is_terminal(interpreter):
    interpreter.start()
    interpreter.start()
    interpreter.shutdown()
    interpreter.shutdown()
    with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
        interpreter.execute("1")


@pytest.mark.parametrize("limits,code", [
    ({"max_feed_duration_secs": 0.01}, "while True: pass"),
    ({"max_memory": 100_000}, "large = 'x' * 1_000_000"),
])
def test_resource_limits_end_the_session(limits, code):
    interpreter = dspy.MontyInterpreter(limits=limits)
    try:
        with pytest.raises(dspy.CodeInterpreterError, match="resource limit") as error:
            interpreter.execute(code)
        assert not isinstance(error.value, dspy.CodeExecutionError)
        with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
            interpreter.execute("1")
    finally:
        interpreter.shutdown()


def test_tool_positional_varargs_and_kwargs(interpreter):
    def tool(first, *rest, scale=1, **extras):
        return (first + sum(rest) + sum(extras.values())) * scale

    interpreter.tools["tool"] = tool
    assert interpreter.execute("tool(2, 7, 11, scale=3, extra=5)") == 75


def test_host_interrupt_discards_suspended_session(interpreter):
    def interrupt():
        raise KeyboardInterrupt

    interpreter.tools["interrupt"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        interpreter.execute("interrupt()")
    with pytest.raises(dspy.CodeInterpreterError, match="shut down"):
        interpreter.execute("1")
