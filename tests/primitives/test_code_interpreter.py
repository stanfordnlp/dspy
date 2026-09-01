import pytest

import dspy
from dspy.primitives.code_interpreter import (
    SUB_DSPY_FACTORY_NAME,
    CodeExecutionError,
    CodeInterpreterError,
    InterpreterCapability,
    interpreter_capabilities,
)


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


def test_sub_dspy_contract_is_public():
    # A DSPy user writing a CodeInterpreter declares capabilities against this enum. The
    # factory-name constant and the reader helper are deliberately module-level only.
    assert dspy.InterpreterCapability is InterpreterCapability
    assert dspy.InterpreterCapability.SUB_DSPY
    assert dspy.InterpreterCapability.FACADE_DSPY
    combined = InterpreterCapability.SUB_DSPY | InterpreterCapability.FACADE_DSPY
    assert InterpreterCapability.SUB_DSPY in combined
    assert InterpreterCapability.FACADE_DSPY in combined
    assert SUB_DSPY_FACTORY_NAME == "dspy_interpreter_factory"
    assert not hasattr(dspy, "SUB_DSPY_FACTORY_NAME")
    assert not hasattr(dspy, "interpreter_capabilities")


def test_interpreter_capabilities_default_is_empty():
    class Bare:
        pass

    assert interpreter_capabilities(Bare) == InterpreterCapability(0)
    assert not interpreter_capabilities(Bare())
    assert not interpreter_capabilities(lambda: None)


def test_interpreter_capabilities_reads_class_instance_and_factory():
    class Capable:
        capabilities = InterpreterCapability.SUB_DSPY

    assert InterpreterCapability.SUB_DSPY in interpreter_capabilities(Capable)
    assert InterpreterCapability.SUB_DSPY in interpreter_capabilities(Capable())

    def factory():
        return Capable()

    factory.capabilities = InterpreterCapability.SUB_DSPY
    assert InterpreterCapability.SUB_DSPY in interpreter_capabilities(factory)


@pytest.mark.parametrize("bad", ["sub_dspy", 42, [InterpreterCapability.SUB_DSPY]])
def test_interpreter_capabilities_rejects_invalid_declarations(bad):
    # Stringly-typed or wrapped declarations fail loudly instead of silently not applying.
    class Misdeclared:
        capabilities = bad

    with pytest.raises(TypeError, match="InterpreterCapability"):
        interpreter_capabilities(Misdeclared)
