import asyncio
import os
import random

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


def test_default_execution_and_denied_access_behaviors(monkeypatch, tmp_path):
    # These default-sandbox cases share one interpreter because separate tests only
    # duplicated Pyodide startup; the assertions still cover each behavior.
    large_var_threshold = 1024
    monkeypatch.setattr("dspy.primitives.python_interpreter.LARGE_VAR_THRESHOLD", large_var_threshold)

    os.environ["FOO_TEST_ENV"] = "test_value"
    test_url = "https://example.com"

    read_file = tmp_path / "test_temp_file.txt"
    read_file.write_text("test content")
    read_virtual_path = f"/sandbox/{read_file.name}"

    write_file = tmp_path / "test_temp_output.txt"
    write_virtual_path = f"/sandbox/{write_file.name}"

    secret_file = tmp_path / "secret.txt"
    secret_content = "This is a secret content"
    secret_file.write_text(secret_content)
    secret_path_str = str(secret_file.absolute())

    malicious_code = f"""
import js
try:
    content = js.Deno.readTextFileSync('{secret_path_str}')
    print(content)
except Exception as e:
    print(f"Error: {{e}}")
"""

    with PythonInterpreter() as interpreter:
        code = "print('Hello, World!')"
        result = interpreter.execute(code)
        assert result == "Hello, World!\n", "Simple print statement should return 'Hello World!\n'"

        code = "import math\nresult = math.sqrt(4)\nresult"
        result = interpreter.execute(code)
        assert result == 2, "Should be able to import and use math.sqrt"

        code = "result = number + 1\nresult"
        result = interpreter.execute(code, variables={"number": 4})
        assert result == 5, "User variable assignment should work"

        keywords_to_test = ["for", "class", "import", "def", "return", "if", "while"]
        for keyword in keywords_to_test:
            with pytest.raises(CodeInterpreterError, match="Invalid variable name"):
                interpreter.execute("print(x)", variables={keyword: 42})

        code = "+++"
        with pytest.raises(SyntaxError, match="Invalid Python syntax"):
            interpreter.execute(code)

        code = "1+0/0"
        with pytest.raises(CodeInterpreterError, match="ZeroDivisionError"):
            interpreter.execute(code)

        token = random.randint(1, 10**9)
        code = f"raise ValueError({token})"
        with pytest.raises(CodeInterpreterError, match=rf"ValueError: \[{token}\]"):
            interpreter.execute(code)

        token = random.randint(1, 10**9)
        code = f"SUBMIT(['The result is', {token}])"
        result = interpreter(code)

        assert isinstance(result, FinalOutput)
        # SUBMIT now always returns a dict with "output" key for single-output default
        assert result.output == {"output": ["The result is", token]}

        result = interpreter.execute("x", variables={"x": (1, 2, 3)})
        assert result == [1, 2, 3]  # Tuples become lists in JSON

        result = interpreter.execute("sorted(x)", variables={"x": {3, 1, 2}})
        assert result == [1, 2, 3]

        result = interpreter.execute("x", variables={"x": {1, "a"}})
        assert isinstance(result, list)
        assert set(result) == {1, "a"}

        complex_data = {"tags": {1, 2, 3}, "coords": (10, 20), "nested": [{"inner_set": {"a", "b"}}]}
        result = interpreter.execute("data", variables={"data": complex_data})
        assert result["tags"] == [1, 2, 3]
        assert result["coords"] == [10, 20]
        assert result["nested"][0]["inner_set"] == ["a", "b"]

        large_data = "x" * (large_var_threshold + 1024)
        result = interpreter.execute("len(data)", variables={"data": large_data})
        assert result == len(large_data), "Large variable should be correctly injected and accessible"

        pattern = "ABCDEFGHIJ" * 100
        patterned_data = pattern * ((large_var_threshold // len(pattern)) + 2)
        code = """
first_100 = data[:100]
last_100 = data[-100:]
(first_100, last_100)
"""
        result = interpreter.execute(code, variables={"data": patterned_data})
        assert result[0] == patterned_data[:100], "First 100 chars should match"
        assert result[1] == patterned_data[-100:], "Last 100 chars should match"

        small_var = "hello"
        large_var = "x" * (large_var_threshold + 1024)
        code = "f'{small} has {len(small)} chars, large has {len(large)} chars'"
        result = interpreter.execute(code, variables={"small": small_var, "large": large_var})
        expected = f"{small_var} has {len(small_var)} chars, large has {len(large_var)} chars"
        assert result == expected, "Both small and large variables should work together"

        large_a = "a" * (large_var_threshold + 100)
        large_b = "b" * (large_var_threshold + 200)
        code = "(len(var_a), len(var_b), var_a[0], var_b[0])"
        result = interpreter.execute(code, variables={"var_a": large_a, "var_b": large_b})
        assert result == [len(large_a), len(large_b), "a", "b"], "Multiple large variables should work"

        num_elements = large_var_threshold // 3
        large_list = ["x"] * num_elements
        code = "(len(data), data[0], data[-1], type(data).__name__)"
        result = interpreter.execute(code, variables={"data": large_list})
        assert result == [num_elements, "x", "x", "list"]

        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "", "Environment variables should be inaccessible without allow-env"

        code = (
            f"try:\n"
            f"    with open({read_virtual_path!r}, 'r') as f:\n"
            f"        data = f.read()\n"
            f"except Exception as e:\n"
            f"    data = str(e)\n"
            f"data"
        )
        result = interpreter.execute(code)
        assert "PermissionDenied" in result or "denied" in result.lower() or "no such file" in result.lower(), (
            "Test file should not be accessible without enable_read_paths"
        )

        code = (
            f"try:\n"
            f"    with open({write_virtual_path!r}, 'w') as f:\n"
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

        output = interpreter(malicious_code)
        assert "Requires read access" in output
        assert secret_content not in output

        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        with pytest.raises(CodeInterpreterError, match="PythonError"):
            interpreter.execute(code)


def test_enabled_access_control_behaviors(tmp_path):
    os.environ["FOO_TEST_ENV"] = "test_value"
    test_url = "https://example.com"

    read_file = tmp_path / "test_temp_file.txt"
    read_file.write_text("test content")
    read_virtual_path = f"/sandbox/{read_file.name}"

    write_file = tmp_path / "test_temp_output.txt"
    write_virtual_path = f"/sandbox/{write_file.name}"

    real_file = tmp_path / "real_name.txt"
    real_file.write_text("through symlink")
    link_file = tmp_path / "link_name.txt"
    try:
        link_file.symlink_to(real_file)
    except (OSError, NotImplementedError):
        link_file = None

    file1 = tmp_path / "test1.txt"
    file2 = tmp_path / "test2.txt"
    file3 = tmp_path / "test3.txt"
    file1.write_text("Content 1")
    file2.write_text("Content 2")
    file3.write_text("Content 3")

    secret_file = tmp_path / "secret.txt"
    secret_content = "This is a secret content"
    secret_file.write_text(secret_content)
    secret_path_str = str(secret_file.absolute())

    malicious_code = f"""
import js
try:
    content = js.Deno.readTextFileSync('{secret_path_str}')
    print(content)
except Exception as e:
    print(f"Error: {{e}}")
"""

    answer_confidence_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]
    answer_score_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    def my_tool(query: str, limit: int = 10) -> str:
        return f"searched '{query}' with limit {limit}"

    def search(query: str, limit: int = 10) -> str:
        return f"query={query}, limit={limit}"

    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    def add(a: int, b: int, c: int) -> str:
        return f"{a + b + c}"

    def failing_tool(x: int) -> str:
        raise ValueError(f"bad value: {x}")

    async def slow_search(query: str) -> str:
        await asyncio.sleep(0)
        return f"answer:{query}"

    async def failing_async(x: int) -> str:
        await asyncio.sleep(0)
        raise ValueError(f"boom:{x}")

    def echo(message: str = "") -> str:
        return f"Echo: {message}"

    host_file = tmp_path / "mount_restart.txt"
    host_file.write_text("restarted-ok")
    host_virtual_path = f"/sandbox/{host_file.name}"

    read_paths = [str(read_file), secret_path_str, str(file1), str(file2), str(file3)]
    if link_file is not None:
        read_paths.append(str(link_file))
    read_paths.append(str(host_file))

    with PythonInterpreter(
        output_fields=answer_confidence_fields,
        enable_env_vars=["FOO_TEST_ENV"],
        enable_read_paths=read_paths,
        enable_write_paths=[str(write_file)],
        enable_network_access=["example.com"],
        tools={
            "my_tool": my_tool,
            "search": search,
            "greet": greet,
            "add": add,
            "failing_tool": failing_tool,
            "slow_search": slow_search,
            "failing_async": failing_async,
            "echo": echo,
        },
    ) as interpreter:
        code = "import os\nresult = os.getenv('FOO_TEST_ENV')\nresult"
        result = interpreter.execute(code)
        assert result == "test_value", "Environment variables should be accessible with allow-env"

        code = f"with open({read_virtual_path!r}, 'r') as f:\n    data = f.read()\ndata"
        result = interpreter.execute(code)
        assert result == "test content", "Test file should be accessible with enable_read_paths and specified file"

        code = f"with open({write_virtual_path!r}, 'w') as f:\n    f.write('allowed')\n'ok'"
        result = interpreter.execute(code)
        assert result == "ok", "Test file should be writable with enable_write_paths"

        code = f"import js\nresp = await js.fetch({test_url!r})\nresp.status"
        result = interpreter.execute(code)
        assert int(result) == 200, "Network access is permitted with enable_network_access"

        output = interpreter(malicious_code)
        assert secret_content in output

        if link_file is not None:
            allow_read_arg = next(a for a in interpreter.deno_command if a.startswith("--allow-read="))
            allow_read = allow_read_arg[len("--allow-read="):].split(",")
            assert os.path.realpath(str(real_file)) in allow_read
            assert str(link_file) not in allow_read

            result = interpreter.execute("with open('/sandbox/link_name.txt') as f:\n    data = f.read()\ndata")
            assert result == "through symlink"

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
        expected_files = ["test1.txt", "test2.txt", "test3.txt"]
        if link_file is not None:
            expected_files = ["link_name.txt", *expected_files]
            assert contents["link_name.txt"] == "through symlink"
        assert set(expected_files).issubset(files), "All expected files should be mounted"
        assert contents["test1.txt"] == "Content 1"
        assert contents["test2.txt"] == "Content 2"
        assert contents["test3.txt"] == "Content 3"

        result = interpreter.execute('SUBMIT(answer="the answer", confidence=0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}

        result = interpreter.execute('SUBMIT("the answer", 0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}

        result = interpreter.execute('my_tool(query="test", limit=5)')
        assert result == "searched 'test' with limit 5"

        result = interpreter.execute('search("hello")')
        assert result == "query=hello, limit=10"

        result = interpreter.execute('search(query="hello", limit=5)')
        assert result == "query=hello, limit=5"

        result = interpreter.execute('greet("World")')
        assert result == "Hello, World!"

        result = interpreter.execute('greet("World", "Hi")')
        assert result == "Hi, World!"

        result = interpreter.execute("add(1, 2, 3)")
        assert result == "6"

        result = interpreter.execute("add(10, 20, c=30)")
        assert result == "60"

        result = interpreter.execute(
            "try:\n"
            "    failing_tool(42)\n"
            "    output = 'no error'\n"
            "except RuntimeError as e:\n"
            "    output = str(e)\n"
            "output"
        )
        assert "ValueError" in result
        assert "bad value: 42" in result

        result = interpreter.execute("slow_search(query='hello')")
        assert result == "answer:hello"

        result = interpreter.execute(
            "try:\n"
            "    failing_async(7)\n"
            "    output = 'no error'\n"
            "except RuntimeError as e:\n"
            "    output = str(e)\n"
            "output"
        )
        assert "ValueError" in result
        assert "boom:7" in result

        first_echo = interpreter.execute('print(echo(message="one"))')
        assert "Echo: one" in first_echo
        first_mount = interpreter.execute(
            f"with open({host_virtual_path!r}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        assert first_mount == "restarted-ok"

        first_pid = interpreter.deno_process.pid
        interpreter.deno_process.kill()
        interpreter.deno_process.wait()

        second_echo = interpreter.execute('print(echo(message="two"))')
        assert "Echo: two" in second_echo
        second_mount = interpreter.execute(
            f"with open({host_virtual_path!r}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        assert second_mount == "restarted-ok"
        assert interpreter.deno_process.pid != first_pid

    assert write_file.exists()
    with open(write_file) as f:
        assert f.read() == "allowed", "Test file outputs should match content written during execution"

    write_file.write_text("original_content")
    with PythonInterpreter(
        output_fields=answer_score_fields,
        enable_write_paths=[str(write_file)],
        sync_files=False,
    ) as interpreter:
        code = f"with open({write_virtual_path!r}, 'w') as f:\n    f.write('should_not_sync')\n'done_no_sync'"
        result = interpreter.execute(code)
        assert result == "done_no_sync"

        code = """
a = "my answer"
s = 42
SUBMIT(a, s)
"""
        result = interpreter.execute(code)
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "my answer", "score": 42}

        with pytest.raises(CodeInterpreterError) as exc_info:
            interpreter.execute("x = 1; SUBMIT(x)")  # Only 1 arg, expects 2
        assert "missing 1 required positional argument" in str(exc_info.value)

    with open(write_file) as f:
        assert f.read() == "original_content", "File should not be changed when sync_files is False"


def test_tools_dict_is_copied():
    """Test that tools dict is defensively copied, not stored by reference."""
    tools = {"my_tool": lambda: "result"}
    sandbox = PythonInterpreter(tools=tools)

    # Modify the original dict after construction
    tools["new_tool"] = lambda: "new"

    # The sandbox should not see the new tool
    assert "new_tool" not in sandbox.tools


def test_deno_command_dict_raises_type_error():
    """Test that passing a dict as deno_command raises TypeError."""
    with pytest.raises(TypeError, match="deno_command must be a list"):
        PythonInterpreter(deno_command={"invalid": "dict"})


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


def test_small_variable_not_using_filesystem():
    """Test that small variables are embedded in code, not using filesystem."""
    small_var = "small string"

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}  # Initialize
    interpreter._inject_variables("print(x)", {"x": small_var})

    assert interpreter._pending_large_vars == {}, "Small variables should not be in _pending_large_vars"


def test_large_variable_threshold_boundary(monkeypatch):
    """Test behavior at exactly the threshold boundary.

    The threshold applies to the serialized size, not the original value.
    For strings, serialization adds 2 bytes (quotes).
    """
    large_var_threshold = 1024
    monkeypatch.setattr("dspy.primitives.python_interpreter.LARGE_VAR_THRESHOLD", large_var_threshold)

    # Serialized size at threshold - should use embedded (not filesystem)
    # Account for 2 bytes of quotes added by repr()
    at_threshold = "x" * (large_var_threshold - 2)

    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": at_threshold})
    assert interpreter._pending_large_vars == {}, "Serialized size at threshold should be embedded"

    # Serialized size over threshold - should use filesystem
    over_threshold = "x" * (large_var_threshold - 1)
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": over_threshold})
    assert "x" in interpreter._pending_large_vars, "Serialized size over threshold should use filesystem"
