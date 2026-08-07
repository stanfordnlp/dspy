import dspy
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError


def test_code_interpreter_error_is_dspy_error():
    error = CodeInterpreterError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, RuntimeError)
    assert str(error) == "boom"
    assert error.args == ("boom",)

    try:
        raise error
    except RuntimeError as caught:
        assert caught is error


def test_code_execution_error_is_dspy_error():
    error = CodeExecutionError("boom")
    assert isinstance(error, dspy.DSPyError)
    assert isinstance(error, CodeInterpreterError)
