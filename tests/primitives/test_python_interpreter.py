import shutil
import pytest
from dspy.primitives.python_interpreter import PythonInterpreter, InterpreterError

# This test suite requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
if shutil.which("deno") is None:
    pytest.skip(reason="Deno is not installed or not in PATH")

def test_execute_simple_code():
    interpreter = PythonInterpreter()
    code = "print('Hello, World!')"
    result = interpreter.execute(code)
    assert result == 'Hello, World!\n', "Simple print statement should return 'Hello World!\n'"

def test_import():
    interpreter = PythonInterpreter()
    code = "import math\nresult = math.sqrt(4)\nresult"
    result = interpreter.execute(code)
    assert result == 2, "Should be able to import and use math.sqrt"

def test_user_variable_definitions():
    interpreter = PythonInterpreter()
    code = "result = number + 1\nresult"
    result = interpreter.execute(code, variables={'number': 4})
    assert result == 5, "User variable assignment should work"

def test_failure_syntax_error():
    interpreter = PythonInterpreter()
    code = "+++"
    with pytest.raises(SyntaxError, match="Invalid Python syntax"):
        interpreter.execute(code)

def test_failure_zero_division():
    interpreter = PythonInterpreter()
    code = "1+0/0"
    with pytest.raises(InterpreterError, match="ZeroDivisionError"):
        interpreter.execute(code)
