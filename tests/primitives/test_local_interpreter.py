"""Tests for LocalInterpreter — unsandboxed host-process Python execution."""

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.local_interpreter import LocalInterpreter


# =============================================================================
# Basic Execution
# =============================================================================


def test_execute_simple_print():
    with LocalInterpreter() as interp:
        result = interp.execute("print('Hello, World!')")
        assert result == "Hello, World!"


def test_execute_no_output():
    with LocalInterpreter() as interp:
        result = interp.execute("x = 42")
        assert result is None


def test_execute_multiline():
    with LocalInterpreter() as interp:
        code = "a = 3\nb = 4\nprint(a + b)"
        result = interp.execute(code)
        assert result == "7"


def test_import_stdlib():
    with LocalInterpreter() as interp:
        result = interp.execute("import math\nprint(math.sqrt(16))")
        assert result == "4.0"


def test_import_third_party():
    """LocalInterpreter can import any installed package (unlike Pyodide sandbox)."""
    with LocalInterpreter() as interp:
        result = interp.execute("import numpy as np\nprint(np.array([1,2,3]).sum())")
        assert result == "6"


# =============================================================================
# State Persistence
# =============================================================================


def test_state_persists_across_calls():
    with LocalInterpreter() as interp:
        interp.execute("x = 10")
        interp.execute("x += 5")
        result = interp.execute("print(x)")
        assert result == "15"


def test_state_cleared_on_shutdown():
    interp = LocalInterpreter()
    interp.start()
    interp.execute("x = 42")
    interp.shutdown()
    interp.start()
    with pytest.raises(CodeInterpreterError, match="NameError"):
        interp.execute("print(x)")
    interp.shutdown()


def test_auto_start():
    """execute() auto-starts if not already started."""
    interp = LocalInterpreter()
    result = interp.execute("print('auto')")
    assert result == "auto"
    interp.shutdown()


# =============================================================================
# Variable Injection
# =============================================================================


def test_variable_injection():
    with LocalInterpreter() as interp:
        result = interp.execute("print(number + 1)", variables={"number": 4})
        assert result == "5"


def test_variable_injection_objects():
    """Variables are injected AS-IS, not serialized — live Python objects."""
    with LocalInterpreter() as interp:
        data = {"key": [1, 2, 3]}
        result = interp.execute("print(type(data).__name__, len(data['key']))", variables={"data": data})
        assert result == "dict 3"


def test_variable_injection_persists():
    with LocalInterpreter() as interp:
        interp.execute("pass", variables={"x": 100})
        result = interp.execute("print(x)")
        assert result == "100"


# =============================================================================
# Error Handling
# =============================================================================


def test_syntax_error():
    with LocalInterpreter() as interp:
        with pytest.raises(SyntaxError):
            interp.execute("def foo(")


def test_runtime_error():
    with LocalInterpreter() as interp:
        with pytest.raises(CodeInterpreterError, match="ZeroDivisionError"):
            interp.execute("1 / 0")


def test_name_error():
    with LocalInterpreter() as interp:
        with pytest.raises(CodeInterpreterError, match="NameError"):
            interp.execute("print(undefined_var)")


def test_error_includes_traceback():
    with LocalInterpreter() as interp:
        with pytest.raises(CodeInterpreterError, match="Traceback"):
            interp.execute("raise RuntimeError('test error')")


def test_stdout_restored_after_error():
    """sys.stdout must be restored even if execution raises."""
    import sys
    original_stdout = sys.stdout
    interp = LocalInterpreter()
    interp.start()
    with pytest.raises(CodeInterpreterError):
        interp.execute("raise ValueError('boom')")
    assert sys.stdout is original_stdout
    interp.shutdown()


# =============================================================================
# SUBMIT / FinalOutput
# =============================================================================


def test_submit_single_arg():
    """Single-arg SUBMIT wraps in {"output": value}."""
    with LocalInterpreter() as interp:
        result = interp.execute('SUBMIT("the answer")')
        assert isinstance(result, FinalOutput)
        assert result.output == {"output": "the answer"}


def test_submit_kwargs():
    with LocalInterpreter() as interp:
        result = interp.execute('SUBMIT(answer="hello", score=42)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "hello", "score": 42}


def test_submit_typed_positional():
    """Positional args mapped to output_fields names."""
    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]
    with LocalInterpreter(output_fields=output_fields) as interp:
        result = interp.execute('SUBMIT("the answer", 0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_typed_kwargs():
    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]
    with LocalInterpreter(output_fields=output_fields) as interp:
        result = interp.execute('SUBMIT(answer="the answer", confidence=0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_wrong_arg_count():
    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]
    with LocalInterpreter(output_fields=output_fields) as interp:
        with pytest.raises(CodeInterpreterError, match="takes 2 positional"):
            interp.execute("SUBMIT('only one')")


def test_submit_no_args():
    with LocalInterpreter() as interp:
        with pytest.raises(CodeInterpreterError, match="SUBMIT requires at least one argument"):
            interp.execute("SUBMIT()")


def test_submit_mixed_args_and_kwargs():
    with LocalInterpreter() as interp:
        with pytest.raises(CodeInterpreterError, match="SUBMIT accepts either positional"):
            interp.execute('SUBMIT("pos", key="val")')


def test_submit_stops_execution():
    """Code after SUBMIT should not execute."""
    with LocalInterpreter() as interp:
        result = interp.execute('SUBMIT(answer="done")\nprint("should not print")')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "done"}


# =============================================================================
# Tools
# =============================================================================


def test_tool_basic():
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with LocalInterpreter(tools={"greet": greet}) as interp:
        result = interp.execute('print(greet("World"))')
        assert result == "Hello, World!"


def test_tool_default_args():
    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    with LocalInterpreter(tools={"search": search}) as interp:
        result = interp.execute('print(search("test"))')
        assert result == "query=test, limit=10"
        result = interp.execute('print(search("test", limit=5))')
        assert result == "query=test, limit=5"


def test_tools_via_setter():
    """Tools can be added/updated after construction via the tools setter."""
    interp = LocalInterpreter()

    def my_tool() -> str:
        return "tool_result"

    interp.tools = {"my_tool": my_tool}
    interp.start()
    result = interp.execute("print(my_tool())")
    assert result == "tool_result"
    interp.shutdown()


def test_tools_refreshed_each_execute():
    """Tools dict is re-injected on each execute(), so updates are visible."""
    interp = LocalInterpreter()
    interp.start()

    interp.tools["counter"] = lambda: "v1"
    result = interp.execute("print(counter())")
    assert result == "v1"

    interp.tools["counter"] = lambda: "v2"
    result = interp.execute("print(counter())")
    assert result == "v2"

    interp.shutdown()


# =============================================================================
# Context Manager
# =============================================================================


def test_context_manager():
    with LocalInterpreter() as interp:
        assert interp._started is True
        result = interp.execute("print('inside')")
        assert result == "inside"
    assert interp._started is False


def test_context_manager_cleanup_on_error():
    with pytest.raises(CodeInterpreterError):
        with LocalInterpreter() as interp:
            interp.execute("raise RuntimeError('fail')")
    assert interp._started is False


# =============================================================================
# Stdout Capture
# =============================================================================


def test_multiple_prints():
    with LocalInterpreter() as interp:
        result = interp.execute("print('a')\nprint('b')\nprint('c')")
        assert result == "a\nb\nc"


def test_trailing_newlines_stripped():
    with LocalInterpreter() as interp:
        result = interp.execute("print('hello')")
        assert result == "hello"  # No trailing newline


def test_empty_print():
    with LocalInterpreter() as interp:
        result = interp.execute("print()")
        assert result == ""  # Empty print produces empty string after strip
