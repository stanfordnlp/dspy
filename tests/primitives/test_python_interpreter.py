import os
import random

import pydantic
import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


class ForwardRefProfile(pydantic.BaseModel):
    name: str
    age: int


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


def test_tools_re_register_after_process_restart():
    """Tools should remain callable after Deno subprocess restart."""
    def echo(message: str = "") -> str:
        return f"Echo: {message}"

    with PythonInterpreter(tools={"echo": echo}) as interpreter:
        first = interpreter.execute('print(echo(message="one"))')
        assert "Echo: one" in first

        first_pid = interpreter.deno_process.pid
        interpreter.deno_process.kill()
        interpreter.deno_process.wait()

        second = interpreter.execute('print(echo(message="two"))')
        assert "Echo: two" in second
        assert interpreter.deno_process.pid != first_pid


def test_mounts_replay_after_process_restart(tmp_path):
    """Mounted files should still be accessible after subprocess restart."""
    host_file = tmp_path / "mount_restart.txt"
    host_file.write_text("restarted-ok")
    virtual_path = f"/sandbox/{host_file.name}"

    with PythonInterpreter(enable_read_paths=[str(host_file)]) as interpreter:
        first = interpreter.execute(
            f"with open({virtual_path!r}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        assert first == "restarted-ok"

        first_pid = interpreter.deno_process.pid
        interpreter.deno_process.kill()
        interpreter.deno_process.wait()

        second = interpreter.execute(
            f"with open({virtual_path!r}, 'r') as f:\n"
            f"    data = f.read()\n"
            f"data"
        )
        assert second == "restarted-ok"
        assert interpreter.deno_process.pid != first_pid


def test_tool_all_positional_args():
    """Test that tools work when all arguments are passed positionally."""

    def add(a: int, b: int, c: int) -> str:
        return f"{a + b + c}"

    with PythonInterpreter(tools={"add": add}) as sandbox:
        result = sandbox.execute("add(1, 2, 3)")
        assert result == "6"

        # Mixed: some positional, some keyword
        result = sandbox.execute("add(10, 20, c=30)")
        assert result == "60"


def test_tool_error_surfaces_as_runtime_error():
    """Test that exceptions raised by a tool surface as RuntimeError in the sandbox."""

    def failing_tool(x: int) -> str:
        raise ValueError(f"bad value: {x}")

    with PythonInterpreter(tools={"failing_tool": failing_tool}) as sandbox:
        result = sandbox.execute(
            "try:\n"
            "    failing_tool(42)\n"
            "    output = 'no error'\n"
            "except RuntimeError as e:\n"
            "    output = str(e)\n"
            "output"
        )
        assert "ValueError" in result
        assert "bad value: 42" in result


def test_tool_pydantic_arg_parsing():
    """Test that tool args are parsed into Pydantic models when annotated."""

    class Profile(pydantic.BaseModel):
        name: str
        age: int

    def greet(profile: Profile) -> str:
        return f"hello {profile.name} ({profile.age})"

    with PythonInterpreter(tools={"greet": greet}) as sandbox:
        result = sandbox.execute('greet(profile={"name": "Ada", "age": "36"})')
        assert result == "hello Ada (36)"


def test_tool_pydantic_arg_parsing_with_forward_ref_annotation():
    """Test pydantic parsing works for resolvable forward-ref string annotations."""

    def greet(profile: "ForwardRefProfile") -> str:
        return f"hello {profile.name} ({profile.age})"

    with PythonInterpreter(tools={"greet": greet}) as sandbox:
        result = sandbox.execute('greet(profile={"name": "Ada", "age": "36"})')
        assert result == "hello Ada (36)"


def test_tool_pydantic_arg_parsing_with_local_forward_ref_annotation_raises():
    """Local forward-ref string annotations should fail with a clear error."""

    class LocalProfile(pydantic.BaseModel):
        name: str
        age: int

    def greet(profile: "LocalProfile") -> str:
        return f"hello {profile.name} ({profile.age})"

    sandbox = PythonInterpreter()
    with pytest.raises(TypeError, match=r"Forward-ref strings are not supported"):
        sandbox._build_tool_info(greet)


def test_tool_pydantic_invalid_input_surfaces_validation_details():
    """Test invalid pydantic input keeps validation details in the surfaced error."""

    class Profile(pydantic.BaseModel):
        name: str
        age: int

    def greet(profile: Profile) -> str:
        return f"hello {profile.name} ({profile.age})"

    with PythonInterpreter(tools={"greet": greet}) as sandbox:
        with pytest.raises(CodeInterpreterError) as exc_info:
            sandbox.execute('greet(profile={"name": "Ada", "age": "not-an-int"})')
        message = str(exc_info.value)
        assert "validationerror" in message.lower()
        assert "age" in message.lower()

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


def test_build_tool_info():
    """Test that _build_tool_info correctly extracts function signatures."""

    def example_fn(required: str, optional: int = 5, untyped=None) -> str:
        pass

    sandbox = PythonInterpreter()
    params, adapters = sandbox._build_tool_info(example_fn)

    assert len(params) == 3
    assert params[0] == {"name": "required", "type": "str"}
    assert params[1] == {"name": "optional", "type": "int", "default": 5}
    assert params[2] == {"name": "untyped", "default": None}
    assert adapters == {}


@pytest.mark.parametrize(
    "fn_factory",
    [
        lambda: (lambda *values: None),
        lambda: (lambda **values: None),
    ],
)
def test_build_tool_info_rejects_variadic_tool_params(fn_factory):
    """Tool signatures with *args/**kwargs should be rejected."""
    sandbox = PythonInterpreter()
    fn = fn_factory()
    with pytest.raises(TypeError, match=r"variadic tool parameters \(\*args/\*\*kwargs\) are not supported"):
        sandbox._build_tool_info(fn)


def test_build_tool_info_complex_types():
    """Test that _build_tool_info handles complex types gracefully."""

    def complex_fn(items: list | None = None, data: dict[str, int] | None = None) -> list:
        pass

    sandbox = PythonInterpreter()
    params, adapters = sandbox._build_tool_info(complex_fn)

    assert len(params) == 2
    assert params[0]["name"] == "items"
    assert params[0]["default"] is None
    assert "json_schema" in params[0]

    assert params[1]["name"] == "data"
    assert params[1]["default"] is None
    assert "json_schema" in params[1]

    assert "items" in adapters
    assert "data" in adapters


def test_build_tool_info_includes_json_schema_and_adapter_for_pydantic_types():
    class Profile(pydantic.BaseModel):
        name: str
        age: int

    def greet(profile: Profile) -> str:
        return f"hello {profile.name}"

    sandbox = PythonInterpreter()
    params, adapters = sandbox._build_tool_info(greet)

    assert len(params) == 1
    assert params[0]["name"] == "profile"
    assert params[0]["json_schema"] == {
        "properties": {
            "name": {"title": "Name", "type": "string"},
            "age": {"title": "Age", "type": "integer"},
        },
        "required": ["name", "age"],
        "title": "Profile",
        "type": "object",
    }

    assert "profile" in adapters
    profile = adapters["profile"].validate_python({"name": "Ada", "age": "36"})
    assert isinstance(profile, Profile)
    assert profile.name == "Ada"
    assert profile.age == 36


def test_build_tool_info_includes_json_schema_for_forward_ref_annotation():
    def greet(profile: "ForwardRefProfile") -> str:
        return f"hello {profile.name}"

    sandbox = PythonInterpreter()
    params, adapters = sandbox._build_tool_info(greet)

    assert len(params) == 1
    assert params[0]["name"] == "profile"
    assert "json_schema" in params[0]
    assert params[0]["json_schema"]["type"] == "object"
    assert "profile" in adapters


def test_build_tool_info_adapter_raises_on_invalid_pydantic_input():
    class Profile(pydantic.BaseModel):
        name: str
        age: int

    def greet(profile: Profile) -> str:
        return f"hello {profile.name}"

    sandbox = PythonInterpreter()
    _, adapters = sandbox._build_tool_info(greet)

    with pytest.raises(pydantic.ValidationError):
        adapters["profile"].validate_python({"name": "Ada", "age": "not-an-int"})


def test_execute_with_pydantic_model_variable():
    """Test that pydantic model instances can be injected as input variables."""

    class Person(pydantic.BaseModel):
        name: str
        age: int

    person = Person(name="Ada", age=36)
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("person['age'] + 1", variables={"person": person})
        assert result == 37


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


# =============================================================================
# Pydantic Stress Tests
# =============================================================================


def test_nested_and_list_pydantic_tool_args():
    """Deep nesting (3 levels) and list[Model] field coerced from dicts."""

    class Employee(pydantic.BaseModel):
        name: str
        role: str

    class Department(pydantic.BaseModel):
        name: str
        lead: Employee
        members: list[Employee]

    def summarize(dept: Department) -> str:
        names = ", ".join(m.name for m in dept.members)
        return f"{dept.name} led by {dept.lead.name}: [{names}]"

    with PythonInterpreter(tools={"summarize": summarize}) as sandbox:
        code = """summarize(dept={
            "name": "Eng",
            "lead": {"name": "Ada", "role": "CTO"},
            "members": [{"name": "Bob", "role": "SWE"}, {"name": "Eve", "role": "SRE"}]
        })"""
        result = sandbox.execute(code)
        assert result == "Eng led by Ada: [Bob, Eve]"


def test_pydantic_optional_and_default_fields():
    """Optional fields (None/omitted) and defaults preserved through coercion."""

    class Config(pydantic.BaseModel):
        name: str
        mode: str = "standard"
        tag: str | None = None

    def show(config: Config) -> str:
        return f"{config.name}:{config.mode}:{config.tag}"

    with PythonInterpreter(tools={"show": show}) as sandbox:
        # All provided
        assert sandbox.execute('show(config={"name": "A", "mode": "fast", "tag": "v1"})') == "A:fast:v1"
        # Defaults kick in
        assert sandbox.execute('show(config={"name": "B"})') == "B:standard:None"
        # Explicit None
        assert sandbox.execute('show(config={"name": "C", "tag": None})') == "C:standard:None"


def test_multi_pydantic_tools_with_mixed_args():
    """Multiple tools with different pydantic types + simple args; tests adapter isolation."""

    class Cat(pydantic.BaseModel):
        name: str
        indoor: bool

    class Dog(pydantic.BaseModel):
        name: str
        breed: str

    def describe_cat(cat: Cat, prefix: str = "") -> str:
        loc = "indoor" if cat.indoor else "outdoor"
        return f"{prefix}{cat.name} is {loc}"

    def describe_dog(dog: Dog) -> str:
        return f"{dog.name} is a {dog.breed}"

    with PythonInterpreter(tools={"describe_cat": describe_cat, "describe_dog": describe_dog}) as sandbox:
        code = """
c = describe_cat(cat={"name": "Whiskers", "indoor": True}, prefix=">> ")
d = describe_dog(dog={"name": "Rex", "breed": "Labrador"})
f"{c} | {d}"
"""
        result = sandbox.execute(code)
        assert result == ">> Whiskers is indoor | Rex is a Labrador"


def test_pydantic_constraint_validation_errors():
    """Constrained fields and nested invalid types surface validation errors."""

    class Inner(pydantic.BaseModel):
        score: int = pydantic.Field(ge=0, le=100)

    class Outer(pydantic.BaseModel):
        inner: Inner

    def process(data: Outer) -> str:
        return str(data.inner.score)

    with PythonInterpreter(tools={"process": process}) as sandbox:
        # Valid
        assert sandbox.execute('process(data={"inner": {"score": 85}})') == "85"
        # Invalid type nested
        with pytest.raises(CodeInterpreterError, match="(?i)validationerror"):
            sandbox.execute('process(data={"inner": {"score": "not-a-number"}})')
        # Constraint violation
        with pytest.raises(CodeInterpreterError, match="(?i)validationerror"):
            sandbox.execute('process(data={"inner": {"score": 150}})')


def test_pydantic_models_as_input_variables():
    """Nested model, list of models, and dict of models all injected as variables."""

    class Address(pydantic.BaseModel):
        city: str

    class Person(pydantic.BaseModel):
        name: str
        address: Address

    class Score(pydantic.BaseModel):
        value: int

    person = Person(name="Ada", address=Address(city="London"))
    items = [Score(value=80), Score(value=90)]
    registry = {"alice": Score(value=95), "bob": Score(value=72)}

    with PythonInterpreter() as interpreter:
        code = """
city = person['address']['city']
avg = sum(i['value'] for i in items) / len(items)
grades = [k + ':' + str(s['value']) for k, s in sorted(registry.items())]
(city, avg, grades)
"""
        result = interpreter.execute(code, variables={
            "person": person, "items": items, "registry": registry,
        })
        assert result == ["London", 85.0, ["alice:95", "bob:72"]]


def test_pydantic_variable_passed_to_tool():
    """Pydantic model injected as variable (dict), then passed to a pydantic-typed tool."""

    class Tag(pydantic.BaseModel):
        label: str
        priority: int

    def format_tag(tag: Tag) -> str:
        return f"[{tag.priority}] {tag.label}"

    tag_instance = Tag(label="urgent", priority=1)

    with PythonInterpreter(tools={"format_tag": format_tag}) as sandbox:
        code = """
label_from_var = tag["label"]
formatted = format_tag(tag={"label": tag["label"], "priority": tag["priority"]})
f"{label_from_var} -> {formatted}"
"""
        result = sandbox.execute(code, variables={"tag": tag_instance})
        assert result == "urgent -> [1] urgent"


def test_tool_return_roundtrip_and_repeated_calls():
    """Tool returns dict consumed by pydantic tool; repeated calls have no state leak."""

    class Record(pydantic.BaseModel):
        id: int
        label: str

    call_count = {"n": 0}

    def fetch(id: int = 0) -> dict:
        return {"id": id, "label": f"item_{id}"}

    def describe(record: Record) -> str:
        call_count["n"] += 1
        return f"#{record.id}:{record.label}"

    with PythonInterpreter(tools={"fetch": fetch, "describe": describe}) as sandbox:
        code = """
results = []
for i in [1, 2, 3]:
    data = fetch(id=i)
    results.append(describe(record=data))
results
"""
        result = sandbox.execute(code)
        assert result == ["#1:item_1", "#2:item_2", "#3:item_3"]
        assert call_count["n"] == 3


def test_tool_falsy_return_values():
    """Tools returning falsy values (0, False, empty string) should preserve them, not coerce to ''."""

    def return_zero() -> int:
        return 0

    def return_false() -> bool:
        return False

    def return_empty_string() -> str:
        return ""

    def return_none() -> str:
        return None

    with PythonInterpreter(tools={
        "return_zero": return_zero,
        "return_false": return_false,
        "return_empty_string": return_empty_string,
        "return_none": return_none,
    }) as sandbox:
        assert sandbox.execute("return_zero()") == "0"
        assert sandbox.execute("return_false()") == "False"
        assert sandbox.execute("return_empty_string()") == ""
        assert sandbox.execute("return_none()") == ""


def test_tool_returning_pydantic_model():
    """Tools returning pydantic models should serialize to JSON dict, not repr string."""

    class Profile(pydantic.BaseModel):
        name: str
        age: int

    def make_profile(name: str, age: int) -> Profile:
        return Profile(name=name, age=age)

    with PythonInterpreter(tools={"make_profile": make_profile}) as sandbox:
        result = sandbox.execute('make_profile(name="Ada", age=36)')
        assert result == {"name": "Ada", "age": 36}


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
# Battle Tests: Pydantic Models in Sandbox (Phase 3)
# =============================================================================


