import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError


@pytest.mark.parametrize(
    "interpreter_type",
    [dspy.LocalInterpreter, pytest.param(dspy.PythonInterpreter, marks=pytest.mark.deno)],
)
@pytest.mark.parametrize("error_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit, ValueError])
def test_host_tool_failures_match_deno(interpreter_type, error_type):
    error = error_type("host tool stopped")

    async def fail():
        raise error

    kwargs = {"execution_timeout": 1} if interpreter_type is dspy.LocalInterpreter else {}
    interpreter = interpreter_type(tools={"fail": fail}, **kwargs)
    try:
        if error_type is ValueError:
            with pytest.raises(CodeExecutionError, match="host tool stopped"):
                interpreter.execute("fail()")
            assert str(interpreter.execute("6 * 7")).strip() == "42"
        else:
            # Python 3.10's asyncio.run replaces task cancellation with an empty CancelledError.
            with pytest.raises(error_type) as expected:
                asyncio.run(fail())
            with pytest.raises(error_type) as caught:
                interpreter.execute("fail()")
            assert type(caught.value) is type(expected.value)
            assert caught.value.args == expected.value.args
            if interpreter_type is dspy.LocalInterpreter:
                assert interpreter._process is None
                with pytest.raises(CodeInterpreterError, match="shut down"):
                    interpreter.execute("6 * 7")
    finally:
        interpreter.shutdown()


def test_cancelled_host_tool_wakes_execution_without_timeout():
    async def fail():
        raise asyncio.CancelledError("host tool stopped")

    interpreter = dspy.LocalInterpreter(tools={"fail": fail})
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with pytest.raises(asyncio.CancelledError):
                executor.submit(interpreter.execute, "fail()").result(timeout=5)
        finally:
            interpreter.shutdown()
