import os
import random
import shutil

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

    with PythonInterpreter(enable_env_vars=None) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "", "Environment variables should be inaccessible without allow-env"

    with PythonInterpreter(enable_env_vars=["FOO_TEST_ENV"]) as interpreter:
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
            f"with open({virtual_path!r}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert result == "test content", "Test file should be accessible with enable_read_paths and specified file"

    with PythonInterpreter(enable_read_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'r') as f:\n"
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

    with PythonInterpreter(enable_write_paths=None) as interpreter:
        code = (
            f"try:\n"
            f"    with open({virtual_path!r}, 'w') as f:\n"
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
            f"with open({virtual_path!r}, 'w') as f:\n"
            f"    f.write('allowed')\n"
            f"'ok'"
        )
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"
    assert testfile_path.exists()
    with open(testfile_path) as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"

    with open(testfile_path, "w") as f:
        f.write("original_content")
    with PythonInterpreter(enable_write_paths=[str(testfile_path)], sync_files=False) as interpreter:
        code = (
            f"with open({virtual_path!r}, 'w') as f:\n"
            f"    f.write('should_not_sync')\n"
            f"'done_no_sync'"
        )
        result = interpreter.execute(code)
        assert result == "done_no_sync"
    with open(testfile_path) as f:
        assert f.read() == "original_content", "File should not be changed when sync_files is False"



def test_enable_net_flag():
    test_url = "https://example.com"

    with PythonInterpreter(enable_network_access=None) as interpreter:
        code = (
            "import js\n"
            f"resp = await js.fetch({test_url!r})\n"
            "resp.status"
        )
        with pytest.raises(InterpreterError, match="PythonError"):
            interpreter.execute(code)

    with PythonInterpreter(enable_network_access=["example.com"]) as interpreter:
        code = (
            "import js\n"
            f"resp = await js.fetch({test_url!r})\n"
            "resp.status"
        )
        result = interpreter.execute(code)
        assert int(result) == 200, "Network access is permitted with enable_network_access"


def test_interpreter_security_filesystem_access(tmp_path):
    """
    Verify that the interpreter cannot read arbitrary files from the host system
    unless explicitly allowed.
    """
    # 1. Create a "secret" file on the host
    secret_file = tmp_path / "secret.txt"
    secret_content = "This is a secret content"
    secret_file.write_text(secret_content)
    secret_path_str = str(secret_file.absolute())

    # 2. Attempt to read the file WITHOUT permission
    malicious_code = f"""
import js
try:
    content = js.Deno.readTextFileSync('{secret_path_str}')
    print(content)
except Exception as e:
    print(f"Error: {{e}}")
"""

    with PythonInterpreter() as interpreter:
        output = interpreter(malicious_code)
        assert "Requires read access" in output
        assert secret_content not in output

    # 3. Attempt to read the file WITH permission
    with PythonInterpreter(enable_read_paths=[secret_path_str]) as interpreter:
        output = interpreter(malicious_code)
        assert secret_content in output

