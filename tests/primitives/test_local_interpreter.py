import os
import random
import shutil

import pytest

from dspy.primitives.local_interpreter import PythonInterpreter
from dspy.primitives.interpreter import InterpreterError

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


def test_rejects_python_keywords_as_variable_names():
    """Test that Python keywords are rejected as variable names."""
    with PythonInterpreter() as interpreter:
        # These are valid Python identifiers but reserved keywords
        # Using them as variable names would cause syntax errors
        keywords_to_test = ["for", "class", "import", "def", "return", "if", "while"]

        for keyword in keywords_to_test:
            with pytest.raises(InterpreterError, match="Invalid variable name"):
                interpreter.execute("print(x)", variables={keyword: 42})


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


def test_final_with_list():
    """Test FINAL() with a list argument returns FinalAnswerResult with dict format."""
    from dspy.primitives.interpreter import FinalAnswerResult

    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"FINAL(['The result is', {token}])"
        result = interpreter(code)

        assert isinstance(result, FinalAnswerResult)
        # FINAL now always returns a dict with "answer" key for single-output default
        assert result.answer == {"answer": ["The result is", token]}

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
# Multi-Output FINAL Tests
# =============================================================================

def test_final_with_typed_signature():
    """Test FINAL with typed output signature."""
    from dspy.primitives.interpreter import FinalAnswerResult

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('FINAL(answer="the answer", confidence=0.95)')

        assert isinstance(result, FinalAnswerResult)
        assert result.answer == {"answer": "the answer", "confidence": 0.95}


def test_final_positional_args():
    """Test FINAL with positional arguments."""
    from dspy.primitives.interpreter import FinalAnswerResult

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "confidence", "type": "float"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('FINAL("the answer", 0.95)')

        assert isinstance(result, FinalAnswerResult)
        assert result.answer == {"answer": "the answer", "confidence": 0.95}


def test_final_var_multi_output():
    """Test FINAL_VAR with multiple output fields using positional args."""
    from dspy.primitives.interpreter import FinalAnswerResult

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        # Positional args: variable names mapped to output fields in order
        code = """
a = "my answer"
s = 42
FINAL_VAR("a", "s")
"""
        result = sandbox.execute(code)

        assert isinstance(result, FinalAnswerResult)
        assert result.answer == {"answer": "my answer", "score": 42}


def test_final_var_wrong_arg_count():
    """Test FINAL_VAR with wrong number of args gives clear error."""
    from dspy.primitives.interpreter import InterpreterError

    output_fields = [
        {"name": "answer", "type": "str"},
        {"name": "score", "type": "int"},
    ]

    with PythonInterpreter(output_fields=output_fields) as sandbox:
        with pytest.raises(InterpreterError) as exc_info:
            sandbox.execute('x = 1; FINAL_VAR("x")')  # Only 1 arg, expects 2
        assert "expects 2 variable names" in str(exc_info.value)


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

