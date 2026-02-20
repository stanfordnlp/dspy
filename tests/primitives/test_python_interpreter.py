import os
import random

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


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


def test_rejects_python_keywords_as_variable_names():
    """Test that Python keywords are rejected as variable names."""
    with PythonInterpreter() as interpreter:
        # These are valid Python identifiers but reserved keywords
        # Using them as variable names would cause syntax errors
        keywords_to_test = ["for", "class", "import", "def", "return", "if", "while"]

        for keyword in keywords_to_test:
            with pytest.raises(CodeInterpreterError, match="Invalid variable name"):
                interpreter.execute("print(x)", variables={keyword: 42})


def test_failure_syntax_error():
    with PythonInterpreter() as interpreter:
        code = "+++"
        with pytest.raises(SyntaxError, match="Invalid Python syntax"):
            interpreter.execute(code)


def test_failure_zero_division():
    with PythonInterpreter() as interpreter:
        code = "1+0/0"
        with pytest.raises(CodeInterpreterError, match="ZeroDivisionError"):
            interpreter.execute(code)


def test_exception_args():
    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"raise ValueError({token})"
        with pytest.raises(CodeInterpreterError, match=rf"ValueError: \[{token}\]"):
            interpreter.execute(code)


def test_submit_with_list():
    """Test SUBMIT() with a list argument returns FinalOutput with dict format."""

    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"SUBMIT(['The result is', {token}])"
        result = interpreter(code)

        assert isinstance(result, FinalOutput)
        # SUBMIT now always returns a dict with "output" key for single-output default
        assert result.output == {"output": ["The result is", token]}


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
        code = f"with open({virtual_path!r}, 'r') as f:\n    data = f.read()\ndata"
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
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be accessible without enable_read_paths"
        )


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
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be writable without enable_write_paths"
        )

    with PythonInterpreter(enable_write_paths=[str(testfile_path)]) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('allowed')\n'ok'"
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"
    assert testfile_path.exists()
    with open(testfile_path) as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"

    with open(testfile_path, "w") as f:
        f.write("original_content")
    with PythonInterpreter(enable_write_paths=[str(testfile_path)], sync_files=False) as interpreter:
        code = f"with open({virtual_path!r}, 'w') as f:\n    f.write('should_not_sync')\n'done_no_sync'"
        result = interpreter.execute(code)
        assert result == "done_no_sync"
    with open(testfile_path) as f:
        assert f.read() == "original_content", "File should not be changed when sync_files is False"


def test_enable_net_flag():
    test_url = "https://example.com"

    with PythonInterpreter(enable_network_access=None) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        with pytest.raises(CodeInterpreterError, match="PythonError"):
            interpreter.execute(code)

    with PythonInterpreter(enable_network_access=["example.com"]) as interpreter:
        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
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


def test_tools_dict_is_copied():
    """Test that tools dict is defensively copied, not stored by reference."""
    tools = {"my_tool": lambda: "result"}
    sandbox = PythonInterpreter(tools=tools)

    # Modify the original dict after construction
    tools["new_tool"] = lambda: "new"

    # The sandbox should not see the new tool
    assert "new_tool" not in sandbox.tools


def test_serialize_tuple():
    """Test that tuples can be serialized as variables."""
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("x", variables={"x": (1, 2, 3)})
        assert result == [1, 2, 3]  # Tuples become lists in JSON


def test_serialize_set():
    """Test that sets can be serialized as variables."""
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("sorted(x)", variables={"x": {3, 1, 2}})
        assert result == [1, 2, 3]


def test_serialize_set_mixed_types():
    """Test that sets with mixed types can be serialized (fallback to list)."""
    with PythonInterpreter() as interpreter:
        # Mixed types can't be sorted, so they serialize as a list in arbitrary order
        # We verify the list contains the expected elements
        result = interpreter.execute("x", variables={"x": {1, "a"}})
        assert isinstance(result, list)
        assert set(result) == {1, "a"}


def test_deno_command_dict_raises_type_error():
    """Test that passing a dict as deno_command raises TypeError."""
    with pytest.raises(TypeError, match="deno_command must be a list"):
        PythonInterpreter(deno_command={"invalid": "dict"})


# =============================================================================
# Typed Tool Signature Tests
# =============================================================================


def test_tool_with_typed_signature():
    """Test that tools get proper typed signatures from inspect."""

    def my_tool(query: str, limit: int = 10) -> str:
        return f"searched '{query}' with limit {limit}"

    with PythonInterpreter(tools={"my_tool": my_tool}) as sandbox:
        # Tool should be callable with typed signature
        result = sandbox.execute('my_tool(query="test", limit=5)')
        assert result == "searched 'test' with limit 5"


def test_tool_positional_args():
    """Test that tools work with positional arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    with PythonInterpreter(tools={"search": search}) as sandbox:
        result = sandbox.execute('search("hello")')
        assert result == "query=hello, limit=10"


def test_tool_keyword_args():
    """Test that tools work with keyword arguments."""

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    with PythonInterpreter(tools={"search": search}) as sandbox:
        result = sandbox.execute('search(query="hello", limit=5)')
        assert result == "query=hello, limit=5"


def test_tool_default_args():
    """Test that tool default arguments work correctly."""

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    with PythonInterpreter(tools={"greet": greet}) as sandbox:
        # Without default
        result = sandbox.execute('greet("World")')
        assert result == "Hello, World!"

        # Overriding default
        result = sandbox.execute('greet("World", "Hi")')
        assert result == "Hi, World!"


# =============================================================================
# Multi-Output SUBMIT Tests
# =============================================================================


def test_submit_with_typed_signature():
    """Test SUBMIT with typed output signature."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('SUBMIT(answer="the answer", confidence=0.95)')

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_positional_args():
    """Test SUBMIT with positional arguments."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('SUBMIT("the answer", 0.95)')

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_multi_output():
    """Test SUBMIT with multiple output fields using positional args."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        # Positional args: values mapped to output fields in order
        code = """
a = "my answer"
s = 42
SUBMIT(a, s)
"""
        result = sandbox.execute(code)

        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "my answer", "score": 42}


def test_submit_wrong_arg_count():
    """Test SUBMIT with wrong number of args gives clear error."""

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        with pytest.raises(CodeInterpreterError) as exc_info:
            sandbox.execute("x = 1; SUBMIT(x)")  # Only 1 arg, expects 2
        assert "missing 1 required positional argument" in str(exc_info.value)


def test_extract_parameters():
    """Test that _extract_parameters correctly extracts function signatures."""

    def example_fn(required: str, optional: int = 5, untyped=None) -> str:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(example_fn)

    assert len(params) == 3
    assert params[0] == {"name": "required", "type": "str"}
    assert params[1] == {"name": "optional", "type": "int", "default": 5}
    assert params[2] == {"name": "untyped", "default": None}


def test_extract_parameters_complex_types():
    """Test that _extract_parameters handles complex types gracefully."""

    def complex_fn(items: list | None = None, data: dict[str, int] | None = None) -> list:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(complex_fn)

    assert len(params) == 2
    # Complex types like Union are not included in type annotation
    assert params[0] == {"name": "items", "default": None}
    assert params[1] == {"name": "data", "default": None}


# =============================================================================
# Large Variable Injection Tests
# =============================================================================


def test_large_variable_injection():
    """Test that large strings are injected via filesystem to avoid Pyodide's FFI size limit."""
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    # Create a string just over the threshold
    large_data = "x" * (LARGE_VAR_THRESHOLD + 1024)

    with PythonInterpreter() as interpreter:
        result = interpreter.execute("len(data)", variables={"data": large_data})
        assert result == len(large_data), "Large variable should be correctly injected and accessible"


def test_large_variable_content_integrity():
    """Test that large variable content is preserved exactly through filesystem injection."""
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    # Create a string with recognizable pattern just over threshold
    pattern = "ABCDEFGHIJ" * 100
    large_data = pattern * ((LARGE_VAR_THRESHOLD // len(pattern)) + 1)

    with PythonInterpreter() as interpreter:
        # Check first and last parts to verify content integrity
        code = """
first_100 = data[:100]
last_100 = data[-100:]
(first_100, last_100)
"""
        result = interpreter.execute(code, variables={"data": large_data})
        assert result[0] == large_data[:100], "First 100 chars should match"
        assert result[1] == large_data[-100:], "Last 100 chars should match"


def test_mixed_small_and_large_variables():
    """Test that small and large variables can be used together."""
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    small_var = "hello"
    large_var = "x" * (LARGE_VAR_THRESHOLD + 1024)

    with PythonInterpreter() as interpreter:
        code = "f'{small} has {len(small)} chars, large has {len(large)} chars'"
        result = interpreter.execute(code, variables={"small": small_var, "large": large_var})
        expected = f"{small_var} has {len(small_var)} chars, large has {len(large_var)} chars"
        assert result == expected, "Both small and large variables should work together"


def test_multiple_large_variables():
    """Test that multiple large variables can be injected."""
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    large_a = "a" * (LARGE_VAR_THRESHOLD + 100)
    large_b = "b" * (LARGE_VAR_THRESHOLD + 200)

    with PythonInterpreter() as interpreter:
        code = "(len(var_a), len(var_b), var_a[0], var_b[0])"
        result = interpreter.execute(code, variables={"var_a": large_a, "var_b": large_b})
        assert result == [len(large_a), len(large_b), "a", "b"], "Multiple large variables should work"


def test_large_list_variable():
    """Test that large list variables are injected via filesystem and JSON parsed."""
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    # Each element "x" serializes to ~3 chars, so divide threshold by 3
    num_elements = LARGE_VAR_THRESHOLD // 3
    large_list = ["x"] * num_elements

    with PythonInterpreter() as interpreter:
        code = "(len(data), data[0], data[-1], type(data).__name__)"
        result = interpreter.execute(code, variables={"data": large_list})
        assert result == [num_elements, "x", "x", "list"]


def test_nested_sets_and_tuples():
    """Test that nested structures with sets and tuples are converted to JSON-compatible types."""
    complex_data = {"tags": {1, 2, 3}, "coords": (10, 20), "nested": [{"inner_set": {"a", "b"}}]}

    with PythonInterpreter() as interpreter:
        result = interpreter.execute("data", variables={"data": complex_data})
        # Sets become sorted lists, tuples become lists
        assert result["tags"] == [1, 2, 3]
        assert result["coords"] == [10, 20]
        assert result["nested"][0]["inner_set"] == ["a", "b"]


def test_small_variable_not_using_filesystem():
    """Test that small variables are embedded in code, not using filesystem."""
    small_var = "small string"

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}  # Initialize
    interpreter._inject_variables("print(x)", {"x": small_var})

    assert interpreter._pending_large_vars == {}, "Small variables should not be in _pending_large_vars"


def test_large_variable_threshold_boundary():
    """Test behavior at exactly the threshold boundary.

    The threshold applies to the serialized size, not the original value.
    For strings, serialization adds 2 bytes (quotes).
    """
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    # Serialized size at threshold - should use embedded (not filesystem)
    # Account for 2 bytes of quotes added by repr()
    at_threshold = "x" * (LARGE_VAR_THRESHOLD - 2)

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": at_threshold})
    assert interpreter._pending_large_vars == {}, "Serialized size at threshold should be embedded"

    # Serialized size over threshold - should use filesystem
    over_threshold = "x" * (LARGE_VAR_THRESHOLD - 1)
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": over_threshold})
    assert "x" in interpreter._pending_large_vars, "Serialized size over threshold should use filesystem"


def test_enable_read_paths_multiple_files(tmp_path):
    """Test that enable_read_paths works with multiple files in the same directory.

    Regression test for bug where mounting multiple files to /sandbox/ failed
    because Pyodide's ErrnoError has errno but no message property, causing
    the 'directory exists' check to fail on the second file.
    """
    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"
    file1.write_text("Content 1")
    file2.write_text("Content 2")
    file3.write_text("Content 3")

    with PythonInterpreter(enable_read_paths=[str(file1), str(file2), str(file3)]) as interpreter:
        code = (
            "import os\n"
            "files = sorted(os.listdir('/sandbox'))\n"
            "contents = {}\n"
            "for f in files:\n"
            "    with open(f'/sandbox/{f}') as fh:\n"
            "        contents[f] = fh.read()\n"
            "(files, contents)"
        )
        result = interpreter.execute(code)
        files, contents = result

        assert files == ["test1.txt", "test2.txt", "test3.txt"], "All three files should be mounted"
        assert contents["test1.txt"] == "Content 1"
        assert contents["test2.txt"] == "Content 2"
        assert contents["test3.txt"] == "Content 3"


# =============================================================================
# Subprocess Restart Resilience Tests
# =============================================================================


def test_tool_works_after_subprocess_kill():
    """Tool calls succeed after the Deno subprocess is forcefully killed."""

    def greet(name: str = "world") -> str:
        return f"Hello, {name}!"

    with PythonInterpreter(tools={"greet": greet}) as interp:
        result = interp.execute('print(greet(name="Alice"))')
        assert "Hello, Alice!" in result

        # Kill the subprocess to simulate crash
        interp.deno_process.kill()
        interp.deno_process.wait()

        # Tools should work after automatic restart and re-registration
        result = interp.execute('print(greet(name="Bob"))')
        assert "Hello, Bob!" in result


def test_restart_clears_registration_state():
    """_ensure_deno_process resets _tools_registered when creating a new process."""

    def my_tool(x: str = "") -> str:
        return f"result: {x}"

    interp = PythonInterpreter(tools={"my_tool": my_tool})
    interp._ensure_deno_process()
    interp._register_tools()
    assert interp._tools_registered is True

    # Kill the process so next _ensure_deno_process creates a new one
    interp.deno_process.kill()
    interp.deno_process.wait()

    interp._ensure_deno_process()
    assert interp._tools_registered is False, "_tools_registered should be cleared after subprocess restart"
    interp.shutdown()


def test_execute_retries_once_on_subprocess_death():
    """execute() retries exactly once when the subprocess dies."""
    from unittest.mock import patch

    from dspy.primitives.python_interpreter import _SubprocessDied

    interp = PythonInterpreter()
    attempts = []

    def mock_inner(code):
        attempts.append(1)
        if len(attempts) == 1:
            raise _SubprocessDied("simulated death")
        return "recovered"

    with patch.object(interp, '_execute_inner', side_effect=mock_inner):
        with patch.object(interp, '_kill_process'):
            result = interp.execute("print(1)")

    assert result == "recovered"
    assert len(attempts) == 2


def test_execute_raises_after_second_failure():
    """execute() raises CodeInterpreterError when both attempts fail."""
    from unittest.mock import patch

    from dspy.primitives.python_interpreter import _SubprocessDied

    interp = PythonInterpreter()

    def always_fail(code):
        raise _SubprocessDied("process died again")

    with patch.object(interp, '_execute_inner', side_effect=always_fail):
        with patch.object(interp, '_kill_process'):
            with pytest.raises(CodeInterpreterError, match="failed after automatic restart"):
                interp.execute("print(1)")


def test_error_message_includes_exit_code():
    """Subprocess death errors include exit code, never blank."""
    from unittest.mock import patch

    from dspy.primitives.python_interpreter import _SubprocessDied

    interp = PythonInterpreter()
    interp._ensure_deno_process()
    interp._tools_registered = True

    # Kill the process
    interp.deno_process.kill()
    interp.deno_process.wait()

    # Prevent auto-restart to observe the raw _SubprocessDied error
    with patch.object(interp, '_ensure_deno_process'):
        with pytest.raises(_SubprocessDied) as exc_info:
            interp._execute_inner("print(1)")

    error_msg = str(exc_info.value)
    assert "exit code" in error_msg
    assert len(error_msg) > 10
    interp._kill_process()
