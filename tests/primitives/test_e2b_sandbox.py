"""Tests for E2BSandbox with mocked E2B Sandbox."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from dspy.primitives.sandbox import FinalAnswerResult, SandboxError


class MockExecution:
    """Mock E2B execution result."""

    def __init__(self, text=None, stdout=None, error=None):
        self.text = text
        self.error = error
        self.logs = MagicMock()
        self.logs.stdout = stdout or []


class MockError:
    """Mock E2B execution error."""

    def __init__(self, name, message=None, value=None):
        self.name = name
        self.message = message
        self.value = value


class MockSandbox:
    """Mock E2B Sandbox class."""

    def __init__(self):
        self.run_code = MagicMock(return_value=MockExecution())
        self.kill = MagicMock()

    @classmethod
    def create(cls, api_key=None, timeout=None, envs=None):
        instance = cls()
        instance._create_kwargs = {"api_key": api_key, "timeout": timeout, "envs": envs}
        return instance


@pytest.fixture
def mock_e2b_module():
    """Create and install a mock e2b_code_interpreter module."""
    mock_module = MagicMock()
    mock_sandbox_instance = MockSandbox()
    mock_module.Sandbox = MagicMock()
    mock_module.Sandbox.create = MagicMock(return_value=mock_sandbox_instance)

    # Install the mock module
    with patch.dict(sys.modules, {"e2b_code_interpreter": mock_module}):
        # Clear the cached import in the E2BSandbox module if it exists
        yield mock_module, mock_sandbox_instance


def test_execute_simple_code(mock_e2b_module):
    """Test basic code execution with print output."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(stdout=["Hello, World!"])

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("print('Hello, World!')")

    assert result == "Hello, World!"


def test_execute_with_expression_result(mock_e2b_module):
    """Test code execution that returns an expression result."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="42")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("40 + 2")

    assert result == "42"


def test_execute_with_variables(mock_e2b_module):
    """Test variable injection into code."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="15")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("x + y", variables={"x": 10, "y": 5})

    # Verify variables were injected (check the code that was passed)
    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert "x = 10" in code
    assert "y = 5" in code


def test_execute_with_string_variable(mock_e2b_module):
    """Test string variable injection with proper escaping."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="hello world")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("msg", variables={"msg": "hello world"})

    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert "msg = 'hello world'" in code or 'msg = "hello world"' in code


def test_execute_with_dict_variable(mock_e2b_module):
    """Test dict variable injection."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="{'a': 1}")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("data", variables={"data": {"a": 1}})

    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert '"a": 1' in code or "'a': 1" in code


def test_final_answer(mock_e2b_module):
    """Test FINAL() returns FinalAnswerResult."""
    mock_module, mock_sandbox = mock_e2b_module

    # Use a callable side_effect to return FinalAnswer only when FINAL() is called
    final_error = MockError("FinalAnswer", value="the answer")

    def run_code_side_effect(code):
        if "FINAL(" in code and "def FINAL" not in code:
            return MockExecution(error=final_error)
        return MockExecution()

    mock_sandbox.run_code.side_effect = run_code_side_effect

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            result = interpreter.execute("FINAL('the answer')")

    assert isinstance(result, FinalAnswerResult)
    assert result.answer == "the answer"


def test_final_var(mock_e2b_module):
    """Test FINAL_VAR() returns FinalAnswerResult with variable value."""
    mock_module, mock_sandbox = mock_e2b_module

    final_error = MockError("FinalAnswer", value=42)

    def run_code_side_effect(code):
        if "FINAL_VAR(" in code and "def FINAL_VAR" not in code:
            return MockExecution(error=final_error)
        return MockExecution()

    mock_sandbox.run_code.side_effect = run_code_side_effect

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            result = interpreter.execute("answer = 42\nFINAL_VAR('answer')")

    assert isinstance(result, FinalAnswerResult)
    assert result.answer == 42


def test_syntax_error(mock_e2b_module):
    """Test SyntaxError is raised for invalid Python."""
    mock_module, mock_sandbox = mock_e2b_module

    syntax_error = MockError("SyntaxError", message="invalid syntax")

    def run_code_side_effect(code):
        # Return error only for the actual user code (not setup code)
        if "+++" in code:
            return MockExecution(error=syntax_error)
        return MockExecution()

    mock_sandbox.run_code.side_effect = run_code_side_effect

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            with pytest.raises(SyntaxError, match="invalid syntax"):
                interpreter.execute("+++")


def test_runtime_error(mock_e2b_module):
    """Test SandboxError is raised for runtime errors."""
    mock_module, mock_sandbox = mock_e2b_module

    runtime_error = MockError("ZeroDivisionError", message="division by zero")

    def run_code_side_effect(code):
        if "1/0" in code:
            return MockExecution(error=runtime_error)
        return MockExecution()

    mock_sandbox.run_code.side_effect = run_code_side_effect

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            with pytest.raises(SandboxError, match="ZeroDivisionError"):
                interpreter.execute("1/0")


def test_name_error(mock_e2b_module):
    """Test SandboxError is raised for undefined variables."""
    mock_module, mock_sandbox = mock_e2b_module

    name_error = MockError("NameError", message="name 'undefined' is not defined")

    call_count = [0]  # Track number of calls to distinguish setup from user code

    def run_code_side_effect(code):
        call_count[0] += 1
        # User code comes after setup calls; check for the actual undefined variable
        if code.strip() == "undefined" or code.strip().endswith("\nundefined"):
            return MockExecution(error=name_error)
        return MockExecution()

    mock_sandbox.run_code.side_effect = run_code_side_effect

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            with pytest.raises(SandboxError, match="NameError"):
                interpreter.execute("undefined")


def test_invalid_variable_name(mock_e2b_module):
    """Test that invalid variable names are rejected."""
    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        with pytest.raises(SandboxError, match="Invalid variable name"):
            interpreter.execute("x", variables={"123invalid": 1})


def test_unsupported_value_type(mock_e2b_module):
    """Test that unsupported value types are rejected."""
    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        with pytest.raises(SandboxError, match="Unsupported value type"):
            interpreter.execute("x", variables={"x": object()})


def test_shutdown_kills_sandbox(mock_e2b_module):
    """Test that shutdown() kills the sandbox."""
    mock_module, mock_sandbox = mock_e2b_module

    from dspy.primitives.e2b_sandbox import E2BSandbox

    interpreter = E2BSandbox(api_key="test-key")
    interpreter.execute("1")  # Force sandbox creation
    interpreter.shutdown()

    mock_sandbox.kill.assert_called_once()


def test_context_manager(mock_e2b_module):
    """Test context manager calls shutdown on exit."""
    mock_module, mock_sandbox = mock_e2b_module

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        interpreter.execute("1")

    mock_sandbox.kill.assert_called_once()


def test_callable_interface(mock_e2b_module):
    """Test that interpreter can be called directly."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="42")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter("40 + 2")

    assert result == "42"


def test_no_output(mock_e2b_module):
    """Test code that produces no output."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution()

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("x = 1")

    assert result is None


def test_combined_stdout_and_text(mock_e2b_module):
    """Test that stdout and expression result are combined."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(
        stdout=["print output\n"],
        text="expression result"
    )

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        result = interpreter.execute("print('hello')\n42")

    assert "print output" in result
    assert "expression result" in result


def test_missing_api_key(mock_e2b_module):
    """Test that missing API key raises appropriate error."""
    from dspy.primitives.e2b_sandbox import E2BSandbox

    # Create interpreter without API key and ensure env var is also not set
    with patch("dspy.primitives.e2b_sandbox.os.environ.get", return_value=None):
        interpreter = E2BSandbox(api_key=None)
        with pytest.raises(SandboxError, match="E2B API key required"):
            interpreter.execute("1")


def test_llm_setup_with_openai_key(mock_e2b_module):
    """Test that llm_query setup runs when OpenAI key is provided."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution()

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key", openai_api_key="sk-test") as interpreter:
        interpreter.execute("1")

    # Should have run setup code including litellm install and LLM setup
    call_count = mock_sandbox.run_code.call_count
    assert call_count >= 2  # At least FINAL setup and LLM setup


def test_environment_variables_passed(mock_e2b_module):
    """Test that environment variables are passed to sandbox."""
    mock_module, mock_sandbox = mock_e2b_module

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(
        api_key="test-key",
        openai_api_key="sk-test",
        llm_model="gpt-4",
        max_llm_calls=100
    ) as interpreter:
        interpreter.execute("1")  # Trigger sandbox creation

    # Check that Sandbox.create was called with envs
    mock_module.Sandbox.create.assert_called_once()
    call_kwargs = mock_module.Sandbox.create.call_args[1]
    assert "envs" in call_kwargs
    envs = call_kwargs["envs"]
    assert envs["OPENAI_API_KEY"] == "sk-test"
    assert envs["LLM_MODEL"] == "gpt-4"
    assert envs["LLM_MAX_CALLS"] == "100"


def test_state_persists_across_calls(mock_e2b_module):
    """Test that state persists across multiple execute calls (same sandbox)."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="2")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        interpreter.execute("counter = 1")
        interpreter.execute("counter + 1")

    # The sandbox should be reused (create only called once)
    assert mock_module.Sandbox.create.call_count == 1


def test_timeout_passed_to_sandbox(mock_e2b_module):
    """Test that timeout is passed to Sandbox.create."""
    mock_module, mock_sandbox = mock_e2b_module

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key", timeout=600) as interpreter:
        interpreter.execute("1")  # Trigger sandbox creation

    call_kwargs = mock_module.Sandbox.create.call_args[1]
    assert call_kwargs["timeout"] == 600


def test_none_values_serialized_correctly(mock_e2b_module):
    """Test that None values are serialized correctly."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="None")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        interpreter.execute("x", variables={"x": None})

    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert "x = None" in code


def test_bool_values_serialized_correctly(mock_e2b_module):
    """Test that boolean values are serialized correctly."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="True")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        interpreter.execute("x", variables={"x": True, "y": False})

    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert "x = True" in code
    assert "y = False" in code


def test_list_values_serialized_correctly(mock_e2b_module):
    """Test that list values are serialized correctly."""
    mock_module, mock_sandbox = mock_e2b_module
    mock_sandbox.run_code.return_value = MockExecution(text="[1, 2, 3]")

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with E2BSandbox(api_key="test-key") as interpreter:
        interpreter.execute("x", variables={"x": [1, 2, 3]})

    call_args = mock_sandbox.run_code.call_args_list[-1]
    code = call_args[0][0]
    assert "[1, 2, 3]" in code


def test_timeout_reconnection(mock_e2b_module):
    """Test that sandbox is recreated after timeout."""
    mock_module, mock_sandbox = mock_e2b_module

    # Track call count and simulate timeout on second user code execution
    call_count = [0]
    timeout_triggered = [False]

    class TimeoutException(Exception):
        pass

    def run_code_side_effect(code):
        call_count[0] += 1
        # Simulate timeout on a specific call (after setup is done)
        # Setup takes 1 call, first execute is call 2, second execute triggers timeout
        if call_count[0] == 3 and not timeout_triggered[0]:
            timeout_triggered[0] = True
            raise TimeoutException("sandbox was not found")
        return MockExecution(text="ok")

    mock_sandbox.run_code.side_effect = run_code_side_effect

    # Track sandbox creations
    create_count = [0]
    original_create = mock_module.Sandbox.create

    def tracked_create(*args, **kwargs):
        create_count[0] += 1
        return original_create(*args, **kwargs)

    mock_module.Sandbox.create = MagicMock(side_effect=tracked_create)

    from dspy.primitives.e2b_sandbox import E2BSandbox

    with patch("dspy.primitives.e2b_sandbox.os.environ.get", side_effect=lambda k, d=None: "test-key" if k == "E2B_API_KEY" else None):
        with E2BSandbox(api_key="test-key") as interpreter:
            # First call should work (triggers setup + execute)
            result1 = interpreter.execute("1")
            assert result1 == "ok"

            # Second call triggers timeout, reconnects, and succeeds
            result2 = interpreter.execute("2")
            assert result2 == "ok"

    # Verify sandbox was recreated (create should be called twice)
    assert create_count[0] == 2, f"Expected 2 sandbox creations, got {create_count[0]}"
