import random
import shutil
import os
import pytest
from dspy.primitives.python_interpreter import InterpreterError, PythonInterpreter

# This test suite requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
if shutil.which("deno") is None:
    pytest.skip(reason="Deno is not installed or not in PATH", allow_module_level=True)


def test_execute_simple_code():
    with PythonInterpreter() as interpreter:
        code = "print('Hello, World!')"
        result = interpreter.execute(code)
        assert result == "Hello, World!\n", "Simple print statement should return 'Hello World!\n'"


def test_import():
    with PythonInterpreter() as interpreter:
        code = "import math\nresult = math.sqrt(4)\nresult"
        result = interpreter.execute(code)
        assert result == 2, "Should be able to import and use math.sqrt"


def test_user_variable_definitions():
    with PythonInterpreter() as interpreter:
        code = "result = number + 1\nresult"
        result = interpreter.execute(code, variables={"number": 4})
        assert result == 5, "User variable assignment should work"


def test_failure_syntax_error():
    with PythonInterpreter() as interpreter:
        code = "+++"
        with pytest.raises(SyntaxError, match="Invalid Python syntax"):
            interpreter.execute(code)


def test_failure_zero_division():
    with PythonInterpreter() as interpreter:
        code = "1+0/0"
        with pytest.raises(InterpreterError, match="ZeroDivisionError"):
            interpreter.execute(code)


def test_exception_args():
    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"raise ValueError({token})"
        with pytest.raises(InterpreterError, match=rf"ValueError: \[{token}\]"):
            interpreter.execute(code)


def test_final_answer_trick():
    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"final_answer('The result is', {token})"
        result = interpreter(code)

        # They should maintain the same order
        assert result == ["The result is", token], "The returned results are differ, `final_answer` trick doesn't work"
    
def test_enable_env_vars_flag():
    os.environ["FOO_TEST_ENV"] = "test_value"

    with PythonInterpreter(enable_env_vars=False) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "", "Environment variables should be inaccessible without allow-env"

    with PythonInterpreter(enable_env_vars=True) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "test_value", "Environment variables should be accessible with allow-env"


def test_read_file_access_control(tmp_path):
    testfile_path = tmp_path / "test_temp_file.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"
    with open(testfile_path, "w") as f:
        f.write("test content")

    with PythonInterpreter(enable_read_paths=[str(testfile_path)]) as interpreter:
        code = (
            f"with open({repr(virtual_path)}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert result == "test content", "Test file should be accessible with enable_read_paths and specified file"

    with PythonInterpreter(enable_read_paths=False) as interpreter:
        code = (
            f"try:\n"
            f"    with open({repr(virtual_path)}, 'r') as f:\n"
            f"        data = f.read()\n"
            f"except Exception as e:\n"
            f"    data = str(e)\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert ("PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower()), "Test file should not be accessible without enable_read_paths"

def test_enable_write_flag(tmp_path):
    testfile_path = tmp_path / "test_temp_output.txt"
    virtual_path = f"/sandbox/{testfile_path.name}"

    with PythonInterpreter(enable_write_paths=False) as interpreter:
        code = (
            f"try:\n"
            f"    with open({repr(virtual_path)}, 'w') as f:\n"
            f"        f.write('blocked')\n"
            f"    result = 'wrote'\n"
            f"except Exception as e:\n"
            f"    result = str(e)\n"
            f"result"
        )
        result = interpreter.execute(code)
        assert ("PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower()), "Test file should not be writable without enable_write_paths"

    with PythonInterpreter(enable_write_paths=[str(testfile_path)]) as interpreter:
        code = (
            f"with open({repr(virtual_path)}, 'w') as f:\n"
            f"    f.write('allowed')\n"
            f"'ok'"
        )
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"
    assert testfile_path.exists()
    with open(testfile_path, "r") as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"



def test_enable_net_flag():
    test_url = "https://example.com"

    with PythonInterpreter(enable_network_access=False) as interpreter:
        code = (
            "import js\n"
            f"resp = await js.fetch({repr(test_url)})\n"
            "resp.status"
        )
        with pytest.raises(InterpreterError, match="PythonError"):
            interpreter.execute(code)

    with PythonInterpreter(enable_network_access=True) as interpreter:
        code = (
            "import js\n"
            f"resp = await js.fetch({repr(test_url)})\n"
            "resp.status"
        )
        result = interpreter.execute(code)
        assert int(result) == 200, "Network access is permitted with enable_network_access"
