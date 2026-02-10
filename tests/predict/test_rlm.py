"""
Tests for the RLM (Recursive Language Model) module.

Test organization:
- Unit tests (no Deno required): MockInterpreter, RLM formatting, signatures
- Integration tests (@pytest.mark.deno): PythonInterpreter with Deno
"""

from contextlib import contextmanager

import pytest

from dspy.adapters.types.tool import Tool
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from tests.mock_interpreter import MockInterpreter

# ============================================================================
# Test Helpers and Factories
# ============================================================================


def make_mock_predictor(responses: list[dict], async_mode: bool = False):
    """Factory for mock predictors with scripted responses.

    Args:
        responses: List of dicts with keys like 'reasoning', 'code'.
        async_mode: If True, returns a predictor with acall() instead of __call__().
    """

    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def _next_response(self):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**result)

        def __call__(self, **kwargs):
            return self._next_response()

        async def acall(self, **kwargs):
            return self._next_response()

    return MockPredictor()


@contextmanager
def dummy_lm_context(responses: list[dict]):
    """Context manager for DummyLM setup."""
    import dspy
    from dspy.utils.dummies import DummyLM

    lm = DummyLM(responses)
    with dspy.context(lm=lm):
        yield lm


# Common test tools
def echo_tool(text: str = "") -> str:
    """Echo the input text."""
    return f"Echo: {text}"


def add_tool(a: int = 0, b: int = 0) -> str:
    """Add two numbers."""
    return str(a + b)


def multiply_tool(a: int = 0, b: int = 0) -> str:
    """Multiply two numbers."""
    return str(a * b)

# ============================================================================
# Unit Tests: MockInterpreter
# ============================================================================


class TestMockInterpreter:
    """Unit tests for MockInterpreter."""

    def test_scripted_responses(self):
        """Test that MockInterpreter returns scripted responses in order."""
        mock = MockInterpreter(responses=["first", "second", "third"])
        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

    def test_returns_final_output_result(self):
        """Test that MockInterpreter can return FinalOutput."""
        mock = MockInterpreter(responses=["exploring", FinalOutput("42")])
        assert mock.execute("print(len(data))") == "exploring"
        result = mock.execute("SUBMIT('42')")
        assert isinstance(result, FinalOutput)
        assert result.output == "42"

    def test_raises_exception_from_responses(self):
        """Test that MockInterpreter raises exceptions from responses."""
        mock = MockInterpreter(responses=["ok", CodeInterpreterError("undefined variable")])
        assert mock.execute("code1") == "ok"
        with pytest.raises(CodeInterpreterError, match="undefined variable"):
            mock.execute("code2")

    def test_records_call_history(self):
        """Test that MockInterpreter records call history for test assertions."""
        mock = MockInterpreter(responses=["resp"])
        mock.execute("print(1)", variables={"x": 10})
        assert mock.call_history == [("print(1)", {"x": 10})]


# ============================================================================
# Unit Tests: RLM Module (no interpreter needed)
# ============================================================================


class TestRLMInitialization:
    """Tests for RLM module initialization."""

    def test_basic_initialization(self):
        """Test RLM module initializes correctly with signature."""
        rlm = RLM("context, query -> answer", max_iterations=5)
        assert rlm.max_iterations == 5
        assert rlm.generate_action is not None
        assert rlm.extract is not None
        assert rlm.tools == {}  # No user tools provided
        assert "context" in rlm.signature.input_fields
        assert "query" in rlm.signature.input_fields
        assert "answer" in rlm.signature.output_fields

    def test_custom_signature(self):
        """Test RLM with custom signature."""
        rlm = RLM("document, question -> summary, key_facts", max_iterations=5)
        assert "document" in rlm.signature.input_fields
        assert "question" in rlm.signature.input_fields
        assert "summary" in rlm.signature.output_fields
        assert "key_facts" in rlm.signature.output_fields

    def test_custom_tools(self):
        """Test RLM with custom tools."""
        def custom_tool(x: str = "") -> str:
            return x.upper()

        rlm = RLM("context -> answer", max_iterations=5, tools=[custom_tool])
        assert "custom_tool" in rlm.tools
        assert len(rlm.tools) == 1  # Only user tools, not internal llm_query/llm_query_batched

    @pytest.mark.parametrize("tool_name", ["invalid-name", "123start"])
    def test_tool_validation_invalid_identifier(self, tool_name):
        """Test RLM rejects tool names that aren't valid Python identifiers."""
        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, name=tool_name)
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            RLM("context -> answer", tools=[tool])

    @pytest.mark.parametrize("tool_name", ["llm_query", "SUBMIT", "print"])
    def test_tool_validation_reserved_names(self, tool_name):
        """Test RLM rejects tool names that conflict with built-in functions."""
        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, name=tool_name)
        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM("context -> answer", tools=[tool])

    @pytest.mark.parametrize("invalid_value", ["not a function", 123])
    def test_tool_validation_not_callable(self, invalid_value):
        """Test RLM rejects tools that aren't callable."""
        with pytest.raises(TypeError, match="must be callable"):
            RLM("context -> answer", tools=[invalid_value])

    def test_tools_dict_rejected(self):
        """Test RLM rejects dict format for tools with helpful error."""
        def my_tool() -> str:
            return "result"

        with pytest.raises(TypeError, match="tools must be a list, not a dict"):
            RLM("context -> answer", tools={"my_tool": my_tool})

    def test_optional_parameters(self):
        """Test RLM optional parameters and their defaults."""
        import dspy

        # Test defaults
        rlm = RLM("context -> answer")
        assert rlm.max_llm_calls == 50
        assert rlm.sub_lm is None
        assert rlm._interpreter is None

        # Test custom values
        mock = MockInterpreter()
        mock_lm = dspy.LM("openai/gpt-4o-mini")
        rlm = RLM("context -> answer", max_llm_calls=100, sub_lm=mock_lm, interpreter=mock)
        assert rlm.max_llm_calls == 100
        assert rlm.sub_lm is mock_lm
        assert rlm._interpreter is mock

    def test_forward_validates_required_inputs(self):
        """Test that forward() raises ValueError for missing required inputs."""
        mock = MockInterpreter(responses=["result"])

        # Single missing input
        rlm = RLM("context, query -> answer", max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError, match="Missing required input"):
            rlm.forward(context="some context")  # Missing 'query'

        # Multiple missing inputs - all should be reported
        rlm = RLM("a, b, c -> answer", max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError) as exc_info:
            rlm.forward(a="only a")  # Missing 'b' and 'c'
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_batched_query_errors_have_clear_markers(self):
        """Test that errors in llm_query_batched are prefixed with [ERROR]."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock()
        mock_lm.side_effect = RuntimeError("LM failed")

        rlm = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm._make_llm_tools()

        results = tools["llm_query_batched"](prompts=["test prompt"])
        assert len(results) == 1
        assert results[0].startswith("[ERROR]")
        assert "LM failed" in results[0]

    def test_tools_call_counter_is_thread_safe(self):
        """Test that the LLM call counter is thread-safe for concurrent llm_query_batched calls.

        The call counter must be protected by a lock since llm_query_batched uses
        ThreadPoolExecutor for concurrent execution.
        """
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import MagicMock

        mock_lm = MagicMock()
        mock_lm.return_value = ["response"]

        rlm = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm._make_llm_tools()

        call_count = [0]
        errors = []

        def make_call():
            try:
                tools["llm_query"](prompt="test")
                call_count[0] += 1
            except RuntimeError as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_call) for _ in range(10)]
            for f in futures:
                f.result()

        assert call_count[0] == 10, f"Expected 10 successful calls, got {call_count[0]}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
            tools["llm_query"](prompt="one more")


class TestRLMFormatting:
    """Tests for RLM formatting helpers."""

    def test_format_history(self):
        """Test history formatting using REPLHistory."""
        history = REPLHistory()
        history = history.append(reasoning="Need to check the data", code="print(1)", output="1")
        history = history.append(reasoning="Now calculate", code="x = 2", output="")
        formatted = history.format()
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "print(1)" in formatted
        assert "Need to check" in formatted

    def test_format_history_empty(self):
        """Test history formatting with empty history."""
        history = REPLHistory()
        formatted = history.format()
        assert "have not interacted with the REPL" in formatted

    def test_action_signature_has_iteration_field(self):
        """Test action signature includes iteration input field."""
        rlm = RLM("context -> answer")
        action_sig = rlm.generate_action.signature
        assert "iteration" in action_sig.input_fields

    def test_format_output(self):
        """Test output formatting."""
        rlm = RLM("context -> answer")
        formatted = rlm._format_output("output text")
        assert "output text" in formatted

    def test_format_output_empty(self):
        """Test output formatting with empty output."""
        rlm = RLM("context -> answer")
        formatted = rlm._format_output("")
        assert "no output" in formatted.lower()

    def test_format_output_passthrough(self):
        """Test that _format_output passes through non-empty output without truncation."""
        rlm = RLM("context -> answer", max_output_chars=100)
        long_output = "a" * 200
        formatted = rlm._format_output(long_output)
        assert formatted == long_output

    def test_format_variable_info_string(self):
        """Test variable info formatting for string value using REPLVariable."""
        var = REPLVariable.from_value("context", "Hello world", preview_chars=5)
        formatted = var.format()
        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "11" in formatted  # length
        assert "He" in formatted  # head
        assert "ld" in formatted  # tail
        assert "..." in formatted  # truncation indicator

    def test_format_variable_info_dict(self):
        """Test variable info formatting for dict value using REPLVariable."""
        var = REPLVariable.from_value("data", {"key": "value"})
        formatted = var.format()
        assert "Variable: `data`" in formatted
        assert "Type: dict" in formatted
        assert "key" in formatted

    def test_build_variables_multiple(self):
        """Test building multiple variables."""
        rlm = RLM("context, query -> answer")
        variables = rlm._build_variables(
            context="Hello world",
            query="What is this?"
        )
        assert len(variables) == 2
        formatted = "\n\n".join(v.format() for v in variables)
        assert "Variable: `context`" in formatted
        assert "Variable: `query`" in formatted
        assert "Hello world" in formatted
        assert "What is this?" in formatted


class TestREPLTypes:
    """Tests for the REPL type classes."""

    def test_repl_history_immutability(self):
        """Test that REPLHistory.append() returns new instance."""
        h1 = REPLHistory()
        h2 = h1.append(code="print(1)", output="1")
        assert len(h1) == 0  # Original unchanged
        assert len(h2) == 1  # New has entry

    def test_repl_history_len_iter_bool(self):
        """Test REPLHistory list-like interface."""
        h = REPLHistory()
        assert len(h) == 0
        assert not bool(h)

        h = h.append(code="x = 1", output="")
        h = h.append(code="x = 2", output="")
        assert len(h) == 2
        assert bool(h)

        codes = [e.code for e in h]
        assert codes == ["x = 1", "x = 2"]

    def test_repl_entry_format(self):
        """Test REPLEntry formatting."""
        entry = REPLEntry(reasoning="test reason", code="print(1)", output="1")
        formatted = entry.format(index=0)
        assert "Step 1" in formatted
        assert "test reason" in formatted
        assert "print(1)" in formatted
        assert "1" in formatted

    def test_repl_entry_format_truncation(self):
        """Test REPLEntry.format() truncates with head+tail and shows true length."""
        output = "a" * 100 + "b" * 100
        entry = REPLEntry(code="print('a' + 'b')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        # Head and tail preserved
        assert "a" * 50 in formatted
        assert "b" * 50 in formatted
        assert "100 characters omitted" in formatted
        # True original length shown in header
        assert "200 chars" in formatted

    def test_repl_entry_format_no_truncation(self):
        """Test REPLEntry.format() passes short output through without truncation."""
        output = "a" * 50
        entry = REPLEntry(code="print('a')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        assert output in formatted
        assert "omitted" not in formatted

    def test_repl_history_threads_max_output_chars(self):
        """Test REPLHistory carries max_output_chars through append()."""
        h = REPLHistory(max_output_chars=50)
        h2 = h.append(code="print(1)", output="a" * 100)
        assert h2.max_output_chars == 50
        # Formatting should truncate at 50 chars
        formatted = h2.format()
        assert "50 characters omitted" in formatted

    def test_repl_variable_from_value(self):
        """Test REPLVariable.from_value() factory."""
        var = REPLVariable.from_value("test", "hello world")
        assert var.name == "test"
        assert var.type_name == "str"
        assert var.total_length == 11
        assert "hello world" in var.preview

    def test_repl_variable_truncation(self):
        """Test REPLVariable preview shows head and tail."""
        var = REPLVariable.from_value("big", "a" * 500 + "b" * 500, preview_chars=50)
        assert var.preview.startswith("a" * 25)
        assert var.preview.endswith("b" * 25)
        assert "..." in var.preview

    def test_repl_variable_with_field_info(self):
        """Test REPLVariable includes desc and constraints from field_info."""
        import dspy

        # Create a field with description and constraints
        field = dspy.InputField(desc="The user's question", ge=0, le=100)

        var = REPLVariable.from_value("query", "What is 2+2?", field_info=field)
        assert var.desc == "The user's question"
        assert "greater than or equal to" in var.constraints

        # Verify format includes the metadata
        formatted = var.format()
        assert "Description: The user's question" in formatted
        assert "Constraints:" in formatted

    def test_repl_variable_without_field_info(self):
        """Test REPLVariable works without field_info."""
        var = REPLVariable.from_value("data", [1, 2, 3])
        assert var.desc == ""
        assert var.constraints == ""

        # Format should not include empty desc/constraints lines
        formatted = var.format()
        assert "Description:" not in formatted
        assert "Constraints:" not in formatted

    def test_build_variables_includes_field_metadata(self):
        """Test _build_variables passes field_info to REPLVariable."""
        import dspy

        class QASig(dspy.Signature):
            """Answer questions."""
            context: str = dspy.InputField(desc="Background information")
            question: str = dspy.InputField(desc="The question to answer")
            answer: str = dspy.OutputField()

        rlm = RLM(QASig, max_iterations=3)
        variables = rlm._build_variables(context="Some text", question="What?")

        # Find the context variable
        context_var = next(v for v in variables if v.name == "context")
        assert context_var.desc == "Background information"

        question_var = next(v for v in variables if v.name == "question")
        assert question_var.desc == "The question to answer"


class TestRLMCallMethod:
    """Tests for RLM __call__ method."""

    def test_call_is_alias_for_forward(self):
        """Test that __call__ is an alias for forward()."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = rlm(query="What is the answer?")
        assert result.answer == "42"


class TestRLMMaxIterationsFallback:
    """Tests for max_iterations reached and extract fallback."""

    def test_max_iterations_triggers_extract(self):
        """Test that reaching max_iterations uses extract fallback."""
        mock = MockInterpreter(responses=[
            "exploring...",
            "still exploring...",
            "more exploring...",
        ])
        rlm = RLM("query -> answer", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore 1", "code": "print('exploring')"},
            {"reasoning": "Explore 2", "code": "print('exploring')"},
            {"reasoning": "Explore 3", "code": "print('exploring')"},
        ])
        # Mock the extract predictor to return a value
        rlm.extract = make_mock_predictor([
            {"answer": "extracted_answer"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "extracted_answer"
        assert result.final_reasoning == "Extract forced final output"


class TestRLMToolExceptions:
    """Tests for tool exception handling."""

    def test_tool_exception_returns_error_in_output(self):
        """Test that tool exceptions are caught and returned as errors."""
        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        mock = MockInterpreter(responses=[
            CodeInterpreterError("RuntimeError: Tool failed!"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock, tools=[failing_tool])
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Call tool", "code": "failing_tool()"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"


class TestRLMDynamicSignature:
    """Tests for the dynamically built RLM signatures."""

    def test_action_signature_structure(self):
        """Test action signature has required fields and instructions."""
        rlm = RLM("document, question -> summary, answer")
        action_sig = rlm.generate_action.signature

        # Required input/output fields
        assert "variables_info" in action_sig.input_fields
        assert "repl_history" in action_sig.input_fields
        assert "reasoning" in action_sig.output_fields
        assert "code" in action_sig.output_fields

        # Instructions mention key tools and variables
        instructions = action_sig.instructions
        assert "llm_query" in instructions
        assert "llm_query_batched" in instructions
        assert "SUBMIT" in instructions
        assert "`document`" in instructions
        assert "`question`" in instructions
        assert "`summary`" in instructions
        assert "`answer`" in instructions

    def test_extract_signature_structure(self):
        """Test extract signature has required fields for all outputs."""
        rlm = RLM("document, question -> summary, key_facts, confidence")
        extract_sig = rlm.extract.signature
        assert "variables_info" in extract_sig.input_fields
        assert "repl_history" in extract_sig.input_fields
        assert "summary" in extract_sig.output_fields
        assert "key_facts" in extract_sig.output_fields
        assert "confidence" in extract_sig.output_fields


# ============================================================================
# Integration Tests: PythonInterpreter (require Deno)
# ============================================================================


@pytest.mark.deno
class TestPythonInterpreter:
    """Integration tests for the secure sandbox with tool support."""

    def test_start_prewarms_sandbox(self):
        """Test that start() pre-warms the sandbox."""
        interp = PythonInterpreter()
        try:
            # Before start, deno_process should be None
            assert interp.deno_process is None
            # After start, it should be running
            interp.start()
            assert interp.deno_process is not None
            assert interp.deno_process.poll() is None  # Still running
            # Execute should work
            result = interp.execute("print(42)")
            assert "42" in result
        finally:
            interp.shutdown()

    def test_start_is_idempotent(self):
        """Test that start() can be called multiple times safely."""
        interp = PythonInterpreter()
        try:
            interp.start()
            first_process = interp.deno_process
            interp.start()  # Second call - should be idempotent
            assert interp.deno_process is first_process  # Same process
        finally:
            interp.shutdown()

    def test_basic_execution(self):
        """Test basic code execution."""
        with PythonInterpreter() as interp:
            result = interp.execute("print(1 + 1)")
            assert "2" in result

    def test_variable_injection(self):
        """Test variable injection."""
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute(
                "print(x + y)",
                variables={"x": 10, "y": 5}
            )
            assert "15" in result

    def test_variable_injection_with_none_values(self):
        """Test variable injection with None values in dicts/lists (JSON null -> Python None)."""
        with PythonInterpreter(tools={}) as interp:
            # Test None in dict
            result = interp.execute(
                "print(data['key'] is None)",
                variables={"data": {"key": None, "other": "value"}}
            )
            assert "True" in result

            # Test None in list
            result = interp.execute(
                "print(items[1] is None)",
                variables={"items": [1, None, 3]}
            )
            assert "True" in result

            # Test nested None
            result = interp.execute(
                "print(nested['inner']['value'] is None)",
                variables={"nested": {"inner": {"value": None}}}
            )
            assert "True" in result

    def test_tool_call_kwargs(self):
        """Test tool call with keyword arguments."""
        def echo(message: str = "") -> str:
            return f"Echo: {message}"

        with PythonInterpreter(tools={"echo": echo}) as interp:
            result = interp.execute('print(echo(message="hello"))')
            assert "Echo: hello" in result

    def test_tool_call_positional(self):
        """Test tool call with positional arguments."""
        def greet(name: str) -> str:
            return f"Hello: {name}"

        with PythonInterpreter(tools={"greet": greet}) as interp:
            result = interp.execute('print(greet("world"))')
            assert "Hello: world" in result

    def test_multiple_tools(self):
        """Test multiple tools."""
        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        def multiply(a: int = 0, b: int = 0) -> str:
            return str(a * b)

        with PythonInterpreter(tools={"add": add, "multiply": multiply}) as interp:
            result = interp.execute("""
sum_result = add(a=3, b=4)
prod_result = multiply(a=3, b=4)
print(f"Sum: {sum_result}, Product: {prod_result}")
""")
            assert "Sum: 7" in result
            assert "Product: 12" in result

    def test_tool_returns_list(self):
        """Test tool that returns a list (like llm_query_batched)."""
        def batch_process(items: list | None = None) -> list:
            items = items or []
            return [f"processed_{item}" for item in items]

        with PythonInterpreter(tools={"batch_process": batch_process}) as interp:
            result = interp.execute("""
results = batch_process(items=["a", "b", "c"])
print(f"Type: {type(results).__name__}")
print(f"Length: {len(results)}")
print(f"First: {results[0]}")
print(f"All: {results}")
""")
            assert "Type: list" in result
            assert "Length: 3" in result
            assert "First: processed_a" in result

    def test_tool_returns_dict(self):
        """Test tool that returns a dict."""
        def get_info() -> dict:
            return {"name": "test", "count": 42}

        with PythonInterpreter(tools={"get_info": get_info}) as interp:
            result = interp.execute("""
info = get_info()
print(f"Type: {type(info).__name__}")
print(f"Name: {info['name']}")
print(f"Count: {info['count']}")
""")
            assert "Type: dict" in result
            assert "Name: test" in result
            assert "Count: 42" in result

    def test_state_persists(self):
        """Test that state persists across executions."""
        with PythonInterpreter(tools={}) as interp:
            interp.execute("x = 10")
            result = interp.execute("print(x + 5)")
            assert "15" in result

    def test_syntax_error(self):
        """Test syntax error handling."""
        with PythonInterpreter(tools={}) as interp:
            with pytest.raises(SyntaxError):
                interp.execute("def incomplete(")

    def test_runtime_error(self):
        """Test runtime error handling."""
        with PythonInterpreter(tools={}) as interp:
            with pytest.raises(CodeInterpreterError):
                interp.execute("undefined_variable")


@pytest.mark.deno
class TestSandboxSecurity:
    """Integration tests for sandbox security restrictions."""

    def test_no_network_access(self):
        """Test that network access is blocked."""
        with PythonInterpreter(tools={}) as interp:
            with pytest.raises(CodeInterpreterError) as exc_info:
                interp.execute("""
from pyodide.http import pyfetch
import asyncio
asyncio.get_event_loop().run_until_complete(pyfetch("https://example.com"))
""")
            assert "net access" in str(exc_info.value).lower() or "allow-net" in str(exc_info.value).lower()

    def test_imports_work(self):
        """Test that standard library imports work."""
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute("""
import json
import re
from collections import Counter
data = {"key": "value"}
print(json.dumps(data))
""")
            assert "key" in result


# ============================================================================
# Unit Tests: RLM with MockInterpreter (no Deno required)
# ============================================================================


class TestRLMAsyncMock:
    """Unit tests for RLM aforward() using MockInterpreter (no Deno required)."""

    @pytest.mark.asyncio
    async def test_aforward_basic(self):
        """Test aforward() returns Prediction with expected output (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = await rlm.aforward(query="What is the answer?")
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_aforward_int_output_mock(self):
        """Test aforward() returns int when signature expects int (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"count": 42})])
        rlm = RLM("query -> count: int", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return count", "code": "SUBMIT(42)"},
        ])

        result = await rlm.aforward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_iteration_mock(self):
        """Test aforward() handles multiple iterations before SUBMIT (MockInterpreter)."""
        mock = MockInterpreter(responses=[
            "explored data",
            FinalOutput({"answer": "done"}),
        ])
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore first", "code": "print('exploring')"},
            {"reasoning": "Now finish", "code": 'SUBMIT("done")'},
        ])

        result = await rlm.aforward(query="test")
        assert result.answer == "done"


class TestRLMTypeCoercionMock:
    """Unit tests for RLM type coercion using MockInterpreter (no Deno required)."""

    @pytest.mark.parametrize("output_field,output_type,final_value,code,expected", [
        ("count", "int", 42, "SUBMIT(42)", 42),
        ("score", "float", 3.14, "SUBMIT(3.14)", 3.14),
        ("valid", "bool", True, "SUBMIT(True)", True),
        ("numbers", "list[int]", [1, 2, 3], "SUBMIT([1, 2, 3])", [1, 2, 3]),
        ("answer", "Literal['yes', 'no']", "yes", 'SUBMIT("yes")', "yes"),
    ])
    def test_type_coercion(self, output_field, output_type, final_value, code, expected):
        """Test RLM type coercion for various types (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({output_field: final_value})])
        rlm = RLM(f"query -> {output_field}: {output_type}", max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return value", "code": code},
        ])

        result = rlm.forward(query="test")
        assert getattr(result, output_field) == expected

    def test_type_error_retries(self):
        """Test RLM retries when type validation fails (MockInterpreter)."""
        mock = MockInterpreter(responses=[
            FinalOutput({"answer": "maybe"}),  # Invalid for Literal
            FinalOutput({"answer": "yes"}),    # Valid
        ])
        rlm = RLM("query -> answer: Literal['yes', 'no']", max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Try maybe", "code": 'SUBMIT("maybe")'},
            {"reasoning": "Try yes", "code": 'SUBMIT("yes")'},
        ])

        result = rlm.forward(query="is it yes?")
        assert result.answer == "yes"


# ============================================================================
# Integration Tests: RLM Type Coercion with PythonInterpreter
# ============================================================================


@pytest.mark.deno
class TestRLMTypeCoercion:
    """Tests for RLM type coercion through full forward pass with PythonInterpreter.

    Note: These tests let RLM create its own PythonInterpreter so it can register
    typed output_fields for SUBMIT based on the signature.
    """

    @pytest.mark.parametrize("output_field,output_type,code,expected,expected_type", [
        ("count", "int", "SUBMIT(42)", 42, int),
        ("score", "float", "SUBMIT(3.14)", 3.14, float),
        ("valid", "bool", "SUBMIT(True)", True, bool),
        ("numbers", "list[int]", "SUBMIT([1, 2, 3])", [1, 2, 3], list),
        ("data", "dict[str, str]", 'SUBMIT({"key": "value"})', {"key": "value"}, dict),
        ("answer", "Literal['yes', 'no']", 'SUBMIT("yes")', "yes", str),
    ])
    def test_type_coercion(self, output_field, output_type, code, expected, expected_type):
        """Test RLM type coercion for various types with PythonInterpreter."""
        rlm = RLM(f"query -> {output_field}: {output_type}", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return value", "code": code},
        ])

        result = rlm.forward(query="test")
        assert getattr(result, output_field) == expected
        assert isinstance(getattr(result, output_field), expected_type)

    def test_submit_extracts_typed_value(self):
        """Test RLM SUBMIT correctly extracts typed value."""
        rlm = RLM("query -> count: int", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Compute and return", "code": "result = 42\nSUBMIT(result)"},
        ])

        result = rlm.forward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)


# ============================================================================
# Integration Tests: RLM Multiple Output Fields
# ============================================================================


@pytest.mark.deno
class TestRLMMultipleOutputs:
    """Tests for signatures with multiple typed output fields.

    Tests SUBMIT() calling patterns with multi-output signatures.
    """

    def test_multi_output_final_kwargs(self):
        """SUBMIT(field1=val1, field2=val2) with keyword args."""
        rlm = RLM("query -> name: str, count: int", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return both outputs", "code": 'SUBMIT(name="alice", count=5)'},
        ])

        result = rlm.forward(query="test")
        assert result.name == "alice"
        assert result.count == 5
        assert isinstance(result.count, int)

    def test_multi_output_final_positional(self):
        """SUBMIT(val1, val2) with positional args mapped to field order."""
        rlm = RLM("query -> name: str, count: int", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return both outputs positionally", "code": 'SUBMIT("bob", 10)'},
        ])

        result = rlm.forward(query="test")
        assert result.name == "bob"
        assert result.count == 10

    def test_multi_output_three_fields(self):
        """Signature with 3+ output fields of different types."""
        rlm = RLM("query -> name: str, age: int, active: bool", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return all three", "code": 'SUBMIT(name="carol", age=30, active=True)'},
        ])

        result = rlm.forward(query="test")
        assert result.name == "carol"
        assert result.age == 30
        assert result.active is True

    def test_multi_output_final_missing_field_errors(self):
        """SUBMIT() with missing field should return error in output."""
        rlm = RLM("query -> name: str, count: int", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Missing count field", "code": 'SUBMIT(name="alice")'},
            {"reasoning": "Now provide both", "code": 'SUBMIT(name="alice", count=5)'},
        ])

        # RLM should retry after getting error for missing field
        result = rlm.forward(query="test")
        assert result.name == "alice"
        assert result.count == 5

    def test_multi_output_submit_vars(self):
        """SUBMIT can pass variables directly for multiple outputs."""
        rlm = RLM("query -> name: str, count: int", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Use SUBMIT", "code": 'n = "dave"\nc = 15\nSUBMIT(n, c)'},
        ])

        result = rlm.forward(query="test")
        assert result.name == "dave"
        assert result.count == 15

    def test_multi_output_type_coercion(self):
        """Each output field is coerced to its declared type."""
        rlm = RLM("query -> count: int, ratio: float, flag: bool", max_iterations=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return mixed types", "code": "SUBMIT(count=42, ratio=3.14, flag=True)"},
        ])

        result = rlm.forward(query="test")
        assert result.count == 42
        assert isinstance(result.count, int)
        assert result.ratio == 3.14
        assert isinstance(result.ratio, float)
        assert result.flag is True
        assert isinstance(result.flag, bool)


# ============================================================================
# Integration Tests: RLM with DummyLM and PythonInterpreter
# ============================================================================


@pytest.mark.deno
class TestRLMWithDummyLM:
    """End-to-end tests using DummyLM with RLM and PythonInterpreter.

    Note: These tests let RLM create its own PythonInterpreter so it can register
    typed output_fields for SUBMIT based on the signature.
    """

    def test_simple_computation_e2e(self):
        """Test full RLM pipeline: DummyLM -> RLM -> PythonInterpreter -> result."""
        with dummy_lm_context([
            {"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"},
        ]):
            rlm = RLM("query -> answer: int", max_iterations=3)
            result = rlm.forward(query="What is 2 + 3?")

            assert result.answer == 5
            assert isinstance(result.answer, int)

    def test_multi_turn_computation_e2e(self):
        """Test RLM with multiple turns before SUBMIT."""
        with dummy_lm_context([
            {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
            {"reasoning": "Now compute and return", "code": "y = x * 2\nSUBMIT(y)"},
        ]):
            rlm = RLM("query -> answer: int", max_iterations=5)
            result = rlm.forward(query="Double ten")

            assert result.answer == 20
            assert len(result.trajectory) == 2

    def test_with_input_variables_e2e(self):
        """Test RLM with input variables passed to sandbox."""
        with dummy_lm_context([
            {"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"},
        ]):
            rlm = RLM("numbers: list[int] -> total: int", max_iterations=3)
            result = rlm.forward(numbers=[1, 2, 3, 4, 5])

            assert result.total == 15

    def test_with_tool_e2e(self):
        """Test RLM calling a host-side tool through the sandbox."""
        def lookup(key: str) -> str:
            return {"apple": "red", "banana": "yellow"}.get(key, "unknown")

        with dummy_lm_context([
            {"reasoning": "Look up the color of apple", "code": 'color = lookup(key="apple")\nSUBMIT(color)'},
        ]):
            rlm = RLM("fruit -> color: str", max_iterations=3, tools=[lookup])
            result = rlm.forward(fruit="apple")

            assert result.color == "red"

    @pytest.mark.asyncio
    async def test_aforward_simple_computation_e2e(self):
        """Test aforward() full pipeline: DummyLM -> RLM -> PythonInterpreter -> result."""
        with dummy_lm_context([
            {"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"},
        ]):
            rlm = RLM("query -> answer: int", max_iterations=3)
            result = await rlm.aforward(query="What is 2 + 3?")

            assert result.answer == 5
            assert isinstance(result.answer, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_turn_e2e(self):
        """Test aforward() with multiple turns before SUBMIT."""
        with dummy_lm_context([
            {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
            {"reasoning": "Now compute and return", "code": "y = x * 2\nSUBMIT(y)"},
        ]):
            rlm = RLM("query -> answer: int", max_iterations=5)
            result = await rlm.aforward(query="Double ten")

            assert result.answer == 20
            assert len(result.trajectory) == 2

    @pytest.mark.asyncio
    async def test_aforward_with_input_variables_e2e(self):
        """Test aforward() with input variables passed to sandbox."""
        with dummy_lm_context([
            {"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"},
        ]):
            rlm = RLM("numbers: list[int] -> total: int", max_iterations=3)
            result = await rlm.aforward(numbers=[1, 2, 3, 4, 5])

            assert result.total == 15


# ============================================================================
# Integration Tests: RLM with real LM (require API key and Deno)
# ============================================================================


@pytest.mark.skip(reason="Requires actual LM and Deno - run manually")
class TestRLMIntegration:
    """Integration tests that require a configured LM."""

    def test_simple_computation(self):
        """Test RLM on simple computation."""
        import dspy
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        rlm = RLM("context, query -> answer", max_iterations=5)
        result = rlm(
            context={"numbers": [1, 2, 3, 4, 5]},
            query="What is the sum of the numbers?"
        )
        assert "15" in result.answer

    def test_with_llm_query(self):
        """Test RLM using the llm_query tool."""
        import dspy
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        rlm = RLM("context, query -> answer", max_iterations=5)
        result = rlm(
            context="The quick brown fox jumps over the lazy dog.",
            query="Use llm_query to describe what animal is mentioned as lazy."
        )
        assert "dog" in result.answer.lower()


# ============================================================================
# Unit Tests: Multimodal Media Support (Audio/Image)
# ============================================================================


class TestMediaDetection:
    """Unit tests for media field detection and registry building."""

    def test_detect_audio_field(self):
        """Test _detect_media_fields finds Audio-typed inputs."""
        import dspy
        from dspy.adapters.types.audio import Audio

        class TranscribeSig(dspy.Signature):
            """Transcribe audio."""
            audio_input: Audio = dspy.InputField()
            transcription: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM(TranscribeSig, max_iterations=3)
            media_fields = rlm._detect_media_fields()

        assert "audio_input" in media_fields
        assert media_fields["audio_input"] == "Audio"

    def test_detect_image_field(self):
        """Test _detect_media_fields finds Image-typed inputs."""
        import dspy
        from dspy.adapters.types.image import Image

        class DescribeSig(dspy.Signature):
            """Describe an image."""
            photo: Image = dspy.InputField()
            description: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('a cat')"}]):
            rlm = RLM(DescribeSig, max_iterations=3)
            media_fields = rlm._detect_media_fields()

        assert "photo" in media_fields
        assert media_fields["photo"] == "Image"

    def test_detect_mixed_media_fields(self):
        """Test _detect_media_fields finds both Audio and Image in same signature."""
        import dspy
        from dspy.adapters.types.audio import Audio
        from dspy.adapters.types.image import Image

        class MultimodalSig(dspy.Signature):
            """Process audio and image together."""
            audio_clip: Audio = dspy.InputField()
            photo: Image = dspy.InputField()
            analysis: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('done')"}]):
            rlm = RLM(MultimodalSig, max_iterations=3)
            media_fields = rlm._detect_media_fields()

        assert len(media_fields) == 2
        assert media_fields["audio_clip"] == "Audio"
        assert media_fields["photo"] == "Image"

    def test_no_media_fields(self):
        """Test _detect_media_fields returns empty dict for text-only signatures."""

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            media_fields = rlm._detect_media_fields()

        assert media_fields == {}

    def test_build_media_registry_with_audio(self):
        """Test _build_media_registry extracts Audio objects from inputs."""
        import dspy
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")

        class TranscribeSig(dspy.Signature):
            """Transcribe audio."""
            audio_input: Audio = dspy.InputField()
            transcription: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM(TranscribeSig, max_iterations=3)
            registry = rlm._build_media_registry({"audio_input": audio})

        assert "audio_input" in registry
        assert registry["audio_input"] is audio

    def test_build_media_registry_ignores_text(self):
        """Test _build_media_registry skips non-media values."""

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            registry = rlm._build_media_registry({"query": "hello world"})

        assert registry == {}


class TestLLMQueryWithMedia:
    """Unit tests for llm_query_with_media tool creation and validation."""

    def test_media_tool_available_when_registry_populated(self):
        """Test llm_query_with_media is created when media registry is non-empty."""
        import dspy
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")

        class TranscribeSig(dspy.Signature):
            """Transcribe audio."""
            audio_input: Audio = dspy.InputField()
            transcription: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM(TranscribeSig, max_iterations=3)
            tools = rlm._make_llm_tools(media_registry={"audio_input": audio})

        assert "llm_query_with_media" in tools
        assert "llm_query" in tools
        assert "llm_query_batched" in tools

    def test_media_tool_absent_when_no_media(self):
        """Test llm_query_with_media is NOT created when media registry is empty."""

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            tools = rlm._make_llm_tools(media_registry={})

        assert "llm_query_with_media" not in tools
        assert "llm_query" in tools

    def test_media_tool_absent_when_registry_none(self):
        """Test llm_query_with_media is NOT created when media registry is None."""

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            tools = rlm._make_llm_tools(media_registry=None)

        assert "llm_query_with_media" not in tools

    def test_media_tool_rejects_empty_prompt(self):
        """Test llm_query_with_media raises on empty prompt."""
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            tools = rlm._make_llm_tools(media_registry={"audio_input": audio})

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            tools["llm_query_with_media"]("")

    def test_media_tool_rejects_no_media_vars(self):
        """Test llm_query_with_media raises when no media var names given."""
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            tools = rlm._make_llm_tools(media_registry={"audio_input": audio})

        with pytest.raises(ValueError, match="At least one media variable"):
            tools["llm_query_with_media"]("transcribe this")

    def test_media_tool_rejects_unknown_var(self):
        """Test llm_query_with_media raises on nonexistent media variable."""
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            tools = rlm._make_llm_tools(media_registry={"audio_input": audio})

        with pytest.raises(ValueError, match="not found"):
            tools["llm_query_with_media"]("transcribe", "nonexistent_var")

    def test_reserved_tool_names_includes_media(self):
        """Test llm_query_with_media is in the reserved tool names set."""
        assert "llm_query_with_media" in RLM._RESERVED_TOOL_NAMES


class TestMediaInstructions:
    """Unit tests for media-specific instruction injection in signatures."""

    def test_media_tools_in_action_instructions(self):
        """Test that media fields cause llm_query_with_media docs to appear."""
        import dspy
        from dspy.adapters.types.audio import Audio

        class TranscribeSig(dspy.Signature):
            """Transcribe audio."""
            audio_input: Audio = dspy.InputField()
            transcription: str = dspy.OutputField()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM(TranscribeSig, max_iterations=3)
            action_sig, _extract_sig = rlm._build_signatures()

        instructions = action_sig.instructions
        assert "llm_query_with_media" in instructions
        assert "audio_input" in instructions

    def test_no_media_instructions_for_text_only(self):
        """Test that text-only signatures do NOT include media instructions."""

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            action_sig, _extract_sig = rlm._build_signatures()

        instructions = action_sig.instructions
        assert "llm_query_with_media" not in instructions


class TestMultiModelSubCalls:
    """Unit tests for multi-model sub-call routing via sub_lms parameter."""

    def test_sub_lms_stored_on_init(self):
        """Test sub_lms dict is stored on the RLM instance."""
        from unittest.mock import MagicMock

        lm1 = MagicMock()
        lm2 = MagicMock()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3, sub_lms={"flash": lm1, "pro": lm2})

        assert rlm.sub_lms == {"flash": lm1, "pro": lm2}

    def test_sub_lms_defaults_to_empty_dict(self):
        """Test sub_lms defaults to empty dict when not provided."""
        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
        assert rlm.sub_lms == {}

    def test_llm_query_routes_to_named_model(self):
        """Test llm_query(prompt, model='name') routes to the correct LM."""
        from unittest.mock import MagicMock

        mock_flash = MagicMock(return_value=["flash response"])
        mock_pro = MagicMock(return_value=["pro response"])
        mock_default = MagicMock(return_value=["default response"])

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_default,
                   sub_lms={"flash": mock_flash, "pro": mock_pro})
        tools = rlm._make_llm_tools()

        result = tools["llm_query"]("test prompt", model="flash")
        assert result == "flash response"
        mock_flash.assert_called_once()

        result = tools["llm_query"]("test prompt", model="pro")
        assert result == "pro response"
        mock_pro.assert_called_once()

    def test_llm_query_default_model_uses_sub_lm(self):
        """Test llm_query without model param falls back to sub_lm."""
        from unittest.mock import MagicMock

        mock_sub = MagicMock(return_value=["sub_lm response"])
        mock_flash = MagicMock(return_value=["flash response"])

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_sub, sub_lms={"flash": mock_flash})
        tools = rlm._make_llm_tools()

        # model=None should use sub_lm, not flash
        result = tools["llm_query"]("test prompt")
        assert result == "sub_lm response"
        mock_sub.assert_called_once()
        mock_flash.assert_not_called()

    def test_llm_query_raises_on_unknown_model(self):
        """Test llm_query raises ValueError for unknown model name."""
        from unittest.mock import MagicMock

        mock_flash = MagicMock(return_value=["flash"])

        rlm = RLM("query -> answer", max_iterations=3, sub_lms={"flash": mock_flash})
        tools = rlm._make_llm_tools()

        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            tools["llm_query"]("test", model="nonexistent")

    def test_llm_query_batched_routes_to_named_model(self):
        """Test llm_query_batched with model param routes all prompts to named LM."""
        from unittest.mock import MagicMock

        mock_default = MagicMock(return_value=["default response"])
        mock_pro = MagicMock(return_value=["pro response"])

        rlm = RLM("query -> answer", max_iterations=3, max_llm_calls=10,
                   sub_lm=mock_default, sub_lms={"pro": mock_pro})
        tools = rlm._make_llm_tools()

        results = tools["llm_query_batched"](["prompt1", "prompt2"], model="pro")
        assert len(results) == 2
        # Both should come from the pro LM
        assert all(r == "pro response" for r in results)
        assert mock_pro.call_count == 2
        mock_default.assert_not_called()

    def test_llm_query_with_media_routes_to_named_model(self):
        """Test llm_query_with_media with model kwarg routes to named LM."""
        from unittest.mock import MagicMock

        from dspy.adapters.types.audio import Audio

        mock_default = MagicMock(return_value=["default response"])
        mock_pro = MagicMock(return_value=["pro media response"])

        audio = Audio(data="dGVzdA==", audio_format="wav")

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_default,
                   sub_lms={"pro": mock_pro})
        tools = rlm._make_llm_tools(media_registry={"audio_input": audio})

        result = tools["llm_query_with_media"]("transcribe", "audio_input", model="pro")
        assert result == "pro media response"
        mock_pro.assert_called_once()
        mock_default.assert_not_called()

    def test_model_docs_in_instructions_when_sub_lms(self):
        """Test that model names appear in action instructions when sub_lms is set."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock()

        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3, sub_lms={"flash": mock_lm, "pro": mock_lm})
            action_sig, _ = rlm._build_signatures()

        instructions = action_sig.instructions
        assert "flash" in instructions
        assert "pro" in instructions
        assert "model=" in instructions

    def test_no_model_docs_when_no_sub_lms(self):
        """Test that model docs are absent when sub_lms is not set."""
        with dummy_lm_context([{"reasoning": "test", "code": "SUBMIT('hi')"}]):
            rlm = RLM("query -> answer", max_iterations=3)
            action_sig, _ = rlm._build_signatures()

        instructions = action_sig.instructions
        assert "Available models:" not in instructions

    def test_call_count_shared_across_models(self):
        """Test that the call counter is shared across all model choices."""
        from unittest.mock import MagicMock

        mock_default = MagicMock(return_value=["default"])
        mock_flash = MagicMock(return_value=["flash"])
        mock_pro = MagicMock(return_value=["pro"])

        rlm = RLM("query -> answer", max_iterations=3, max_llm_calls=3,
                   sub_lm=mock_default, sub_lms={"flash": mock_flash, "pro": mock_pro})
        tools = rlm._make_llm_tools()

        tools["llm_query"]("p1", model="flash")
        tools["llm_query"]("p2", model="pro")
        tools["llm_query"]("p3")  # default

        # 4th call should exceed the limit of 3
        with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
            tools["llm_query"]("p4", model="flash")


class TestBudgetTracking:
    """Tests for budget() tool and max_time enforcement."""

    def test_max_time_initialization(self):
        """Test that max_time is stored on the RLM instance."""
        rlm = RLM("query -> answer", max_time=60.0)
        assert rlm.max_time == 60.0

    def test_max_time_default_none(self):
        """Test that max_time defaults to None (no limit)."""
        rlm = RLM("query -> answer")
        assert rlm.max_time is None

    def test_budget_tool_created(self):
        """Test that the budget tool is included in execution tools."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=10)
        tools = rlm._make_llm_tools()
        assert "budget" in tools
        assert callable(tools["budget"])

    def test_budget_returns_string(self):
        """Test that budget() returns a human-readable string."""
        rlm = RLM("query -> answer", max_iterations=10, max_llm_calls=20)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 3}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert isinstance(result, str)
        assert "Iterations:" in result
        assert "LLM calls:" in result

    def test_budget_reflects_iteration(self):
        """Test that budget() shows correct remaining iterations."""
        rlm = RLM("query -> answer", max_iterations=10, max_llm_calls=20)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 7}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        # iteration=7, max=10, remaining = 10 - 7 - 1 = 2
        assert "2/10 remaining" in result

    def test_budget_reflects_llm_calls(self):
        """Test that budget() shows correct remaining LLM calls after usage."""
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}

        from unittest.mock import MagicMock
        mock_lm = MagicMock(return_value=["response"])
        rlm_with_lm = RLM("query -> answer", max_iterations=5, max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm_with_lm._make_llm_tools(execution_state=execution_state)

        # Use 3 LLM calls
        tools["llm_query"]("prompt1")
        tools["llm_query"]("prompt2")
        tools["llm_query"]("prompt3")

        result = tools["budget"]()
        assert "7/10 remaining" in result

    def test_budget_shows_time_when_max_time_set(self):
        """Test that budget() includes time info when max_time is configured."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=10, max_time=120.0)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "Time:" in result
        assert "/120.0s remaining" in result

    def test_budget_no_time_when_max_time_none(self):
        """Test that budget() shows 'no limit' when max_time is None."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=10)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "no limit" in result

    def test_budget_reserved_name(self):
        """Test that 'budget' is a reserved tool name."""
        def budget() -> str:
            return "custom"

        from dspy.adapters.types.tool import Tool
        tool = Tool(budget, name="budget")
        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM("query -> answer", tools=[tool])

    def test_budget_in_action_instructions(self):
        """Test that the action instructions mention budget()."""
        rlm = RLM("query -> answer", max_iterations=5)
        action_sig = rlm.generate_action.signature
        assert "budget()" in action_sig.instructions

    def test_max_time_triggers_extract_fallback(self):
        """Test that exceeding max_time triggers extract fallback (not exception)."""

        mock = MockInterpreter(responses=[
            "exploring...",
            "still exploring...",
        ])
        # Set max_time to 0 so it's already exceeded on first check
        rlm = RLM("query -> answer", max_iterations=5, max_time=0.0, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
        ])
        rlm.extract = make_mock_predictor([
            {"answer": "timeout_fallback"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "timeout_fallback"
        assert result.final_reasoning == "Extract forced final output"

    def test_max_time_none_no_timeout(self):
        """Test that max_time=None means no time checking."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iterations=5, max_time=None, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "42"

    def test_budget_iteration_updates_via_tool(self):
        """Test that budget() reports decreasing iterations across a forward() run."""
        budget_reports = []

        class BudgetCapturingInterpreter(MockInterpreter):
            def __init__(self):
                super().__init__(responses=["output1", "output2", FinalOutput({"answer": "done"})])

            def execute(self, code, variables=None):
                # Call the budget tool if available
                if "budget" in self.tools:
                    budget_reports.append(self.tools["budget"]())
                return super().execute(code, variables)

        mock_interp = BudgetCapturingInterpreter()
        rlm = RLM("query -> answer", max_iterations=5, interpreter=mock_interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Step 1", "code": "print('a')"},
            {"reasoning": "Step 2", "code": "print('b')"},
            {"reasoning": "Step 3", "code": 'SUBMIT("done")'},
        ])

        result = rlm(query="test")
        assert result.answer == "done"
        # We got 3 iterations (0, 1, 2), should have 3 budget reports
        assert len(budget_reports) == 3
        # Each report should show decreasing remaining iterations
        for report in budget_reports:
            assert "Iterations:" in report
            assert "LLM calls:" in report


    def test_max_cost_initialization(self):
        """Test that max_cost is stored on the RLM instance."""
        rlm = RLM("query -> answer", max_cost=0.10)
        assert rlm.max_cost == 0.10

    def test_max_cost_default_none(self):
        """Test that max_cost defaults to None (no limit)."""
        rlm = RLM("query -> answer")
        assert rlm.max_cost is None

    def test_budget_shows_cost_when_max_cost_set(self):
        """Test that budget() includes cost info when max_cost is configured."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=10, max_cost=0.50)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "Cost:" in result
        assert "$0.50" in result

    def test_budget_no_cost_when_max_cost_none_and_no_spending(self):
        """Test that budget() omits cost when max_cost is None and nothing spent."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=10)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        # No cost tracking when max_cost is None and no LM history entries with cost
        assert "Cost:" not in result or "no limit" in result

    def test_max_cost_zero_triggers_immediate_fallback(self):
        """Test that max_cost=0 triggers extract fallback immediately."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        # Give the mock LM a history with a cost entry
        mock_lm.history = [{"cost": 0.001, "usage": {"total_tokens": 100}}]

        mock = MockInterpreter(responses=["exploring..."])
        rlm = RLM("query -> answer", max_iterations=5, max_cost=0.0, sub_lm=mock_lm, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
        ])
        rlm.extract = make_mock_predictor([
            {"answer": "cost_fallback"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "cost_fallback"


    def test_byok_cost_upstream_inference_cost(self):
        """Test cost tracking includes usage.cost_details.upstream_inference_cost for BYOK."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        # Start with empty history — entries added after tool creation
        mock_lm.history = []

        rlm = RLM("query -> answer", max_cost=1.0, sub_lm=mock_lm)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)

        # Simulate BYOK responses arriving after tool creation
        mock_lm.history.extend([
            {
                "cost": 0,
                "usage": {
                    "total_tokens": 500,
                    "is_byok": True,
                    "cost_details": {"upstream_inference_cost": 0.0025},
                },
            },
            {
                "cost": None,
                "usage": {
                    "total_tokens": 300,
                    "cost_details": {"upstream_inference_cost": 0.0015},
                },
            },
        ])

        result = tools["budget"]()
        # Should pick up the upstream_inference_cost values: 0.0025 + 0.0015
        assert "$0.0040" in result
        assert "800 tokens" in result  # 500 + 300

    def test_cost_sums_provider_and_upstream(self):
        """Test cost tracking sums both provider cost and upstream inference cost."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        mock_lm.history = []

        rlm = RLM("query -> answer", max_cost=1.0, sub_lm=mock_lm)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)

        # Non-BYOK: both provider cost and upstream cost are nonzero
        mock_lm.history.append({
            "cost": 0.01,
            "usage": {
                "total_tokens": 1000,
                "cost_details": {"upstream_inference_cost": 0.005},
            },
        })

        result = tools["budget"]()
        # Sum of 0.01 + 0.005 = 0.015
        assert "$0.0150" in result

    def test_budget_warning_low_iterations(self):
        """Test that budget() shows warning when iterations are low."""
        rlm = RLM("query -> answer", max_iterations=5, max_llm_calls=50)
        # iteration=4 means 0 remaining (5 - 4 - 1)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 4}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "LOW" in result
        assert "iterations" in result

    def test_budget_warning_low_time(self):
        """Test that budget() shows warning when time is running low."""
        import time
        rlm = RLM("query -> answer", max_iterations=20, max_llm_calls=50, max_time=10.0)
        # Start time 9 seconds ago — only 1s remaining (10%)
        execution_state = {"start_time": time.monotonic() - 9.0, "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "LOW" in result
        assert "time" in result

    def test_budget_no_warning_when_plenty_remaining(self):
        """Test that budget() has no warning when resources are plentiful."""
        rlm = RLM("query -> answer", max_iterations=20, max_llm_calls=50, max_time=60.0)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(execution_state=execution_state)
        result = tools["budget"]()
        assert "LOW" not in result


# ============================================================================
# Integration Tests: RLM + LocalInterpreter
# ============================================================================


class TestRLMWithLocalInterpreter:
    """Integration tests proving RLM and LocalInterpreter work together end-to-end."""

    def test_basic_forward(self):
        """Test RLM forward() with LocalInterpreter produces a result."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=3, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("hello")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "hello"

    def test_state_persists_across_iterations(self):
        """Test that variables set in one iteration survive to the next."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=5, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Set a variable", "code": "x = 42\nprint(x)"},
            {"reasoning": "Use it", "code": "SUBMIT(str(x * 2))"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "84"

    def test_tools_accessible_in_code(self):
        """Test that llm_query and budget tools are callable from LocalInterpreter code."""
        from unittest.mock import MagicMock

        from dspy.primitives.local_interpreter import LocalInterpreter

        mock_lm = MagicMock(return_value=["mocked response"])
        mock_lm.history = []

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=5, sub_lm=mock_lm, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Use tools", "code": 'b = budget()\nresult = llm_query("hi")\nprint(result)'},
            {"reasoning": "Submit", "code": "SUBMIT(result)"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "mocked response"

    def test_stdlib_imports_work(self):
        """Test that LocalInterpreter allows stdlib imports inside RLM."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=3, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Use stdlib", "code": 'import json\nSUBMIT(json.dumps({"a": 1}))'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == '{"a": 1}'

    def test_error_recovery(self):
        """Test that a runtime error in one iteration doesn't kill the session."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=5, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "This will error", "code": "1 / 0"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "recovered"

    def test_max_time_with_local_interpreter(self):
        """Test that max_time budget enforcement works with LocalInterpreter."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=10, max_time=0.0, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
        ])
        rlm.extract = make_mock_predictor([
            {"answer": "timeout_fallback"},
        ])

        result = rlm.forward(query="test")
        assert result.answer == "timeout_fallback"

    @pytest.mark.asyncio
    async def test_aforward_with_local_interpreter(self):
        """Test RLM aforward() works with LocalInterpreter."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        rlm = RLM("query -> answer", max_iterations=3, interpreter=interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("async_hello")'},
        ])

        result = await rlm.aforward(query="test")
        assert result.answer == "async_hello"


# ============================================================================
# Tests: llm_query_with_media content construction
# ============================================================================


class TestMediaContentConstruction:
    """Test that llm_query_with_media builds correct multimodal content for the LM."""

    def test_media_content_parts_sent_to_lm(self):
        """Test that llm_query_with_media sends multimodal content parts to the LM."""
        from unittest.mock import MagicMock

        from dspy.adapters.types.audio import Audio

        mock_lm = MagicMock(return_value=["transcription result"])
        mock_lm.history = []

        audio = Audio(data="dGVzdA==", audio_format="wav")

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_lm)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(
            media_registry={"my_audio": audio},
            execution_state=execution_state,
        )

        result = tools["llm_query_with_media"]("describe this audio", "my_audio")
        assert result == "transcription result"

        # Verify the LM was called with messages containing multimodal content
        mock_lm.assert_called_once()
        call_kwargs = mock_lm.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        # First part should be text
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "describe this audio"
        # Remaining parts should be from audio.format()
        assert len(content) > 1

    def test_image_content_parts_sent_to_lm(self):
        """Test that llm_query_with_media sends image content parts to the LM."""
        from unittest.mock import MagicMock

        from dspy.adapters.types.image import Image

        mock_lm = MagicMock(return_value=["a cat sitting on a mat"])
        mock_lm.history = []

        image = Image(url="https://example.com/cat.jpg")

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_lm)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(
            media_registry={"my_image": image},
            execution_state=execution_state,
        )

        result = tools["llm_query_with_media"]("what is in this image?", "my_image")
        assert result == "a cat sitting on a mat"

        # Verify multimodal message structure
        mock_lm.assert_called_once()
        call_kwargs = mock_lm.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        assert content[0]["type"] == "text"
        assert len(content) > 1

    def test_multiple_media_objects_in_one_call(self):
        """Test llm_query_with_media with multiple media variables."""
        from unittest.mock import MagicMock

        from dspy.adapters.types.audio import Audio
        from dspy.adapters.types.image import Image

        mock_lm = MagicMock(return_value=["combined analysis"])
        mock_lm.history = []

        audio = Audio(data="dGVzdA==", audio_format="wav")
        image = Image(url="https://example.com/photo.jpg")

        rlm = RLM("query -> answer", max_iterations=3, sub_lm=mock_lm)
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(
            media_registry={"audio_in": audio, "image_in": image},
            execution_state=execution_state,
        )

        result = tools["llm_query_with_media"]("analyze both", "audio_in", "image_in")
        assert result == "combined analysis"

        # Verify both media objects were included
        call_kwargs = mock_lm.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        # text + audio parts + image parts
        assert len(content) >= 3

    def test_media_with_model_routing(self):
        """Test llm_query_with_media routes to named model when specified."""
        from unittest.mock import MagicMock

        from dspy.adapters.types.audio import Audio

        mock_default = MagicMock(return_value=["default"])
        mock_default.history = []
        mock_pro = MagicMock(return_value=["pro result"])
        mock_pro.history = []

        audio = Audio(data="dGVzdA==", audio_format="wav")

        rlm = RLM("query -> answer", max_iterations=3,
                   sub_lm=mock_default, sub_lms={"pro": mock_pro})
        execution_state = {"start_time": __import__("time").monotonic(), "iteration": 0}
        tools = rlm._make_llm_tools(
            media_registry={"audio": audio},
            execution_state=execution_state,
        )

        result = tools["llm_query_with_media"]("transcribe", "audio", model="pro")
        assert result == "pro result"
        mock_pro.assert_called_once()
        mock_default.assert_not_called()


# ============================================================================
# Tests: max_cost mid-run fallback
# ============================================================================


class TestMaxCostMidRunFallback:
    """Test that max_cost triggers extract fallback during a multi-iteration run."""

    def test_cost_exceeded_mid_run_triggers_fallback(self):
        """Test that exceeding max_cost mid-run triggers extract fallback, not crash."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        mock_lm.history = []

        mock = MockInterpreter(responses=[
            "first iteration output",
            "second iteration output",  # cost exceeded before this
        ])
        rlm = RLM("query -> answer", max_iterations=10, max_cost=0.05,
                   sub_lm=mock_lm, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
            {"reasoning": "More", "code": "print('more')"},
        ])
        rlm.extract = make_mock_predictor([
            {"answer": "cost_fallback"},
        ])

        # Simulate cost appearing after first iteration
        # The forward loop checks cost at the start of each iteration
        # After iteration 0, we inject cost into LM history
        original_execute = mock.execute

        def execute_with_cost_injection(code, variables=None):
            result = original_execute(code, variables)
            # After first execute, inject cost exceeding budget
            if mock.call_count == 1:
                mock_lm.history.append({
                    "cost": 0.10,  # exceeds max_cost=0.05
                    "usage": {"total_tokens": 5000},
                })
            return result

        mock.execute = execute_with_cost_injection

        result = rlm.forward(query="test")
        assert result.answer == "cost_fallback"
        assert result.final_reasoning == "Extract forced final output"


# ============================================================================
# Tests: Async budget/time/cost
# ============================================================================


class TestAsyncBudgetTimeCost:
    """Test that budget, max_time, and max_cost work correctly in aforward()."""

    @pytest.mark.asyncio
    async def test_aforward_max_time_triggers_fallback(self):
        """Test that aforward() respects max_time and triggers extract fallback."""
        mock = MockInterpreter(responses=["exploring..."])
        rlm = RLM("query -> answer", max_iterations=5, max_time=0.0, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
        ], async_mode=True)
        rlm.extract = make_mock_predictor([
            {"answer": "async_timeout_fallback"},
        ], async_mode=True)

        result = await rlm.aforward(query="test")
        assert result.answer == "async_timeout_fallback"

    @pytest.mark.asyncio
    async def test_aforward_max_cost_triggers_fallback(self):
        """Test that aforward() respects max_cost and triggers extract fallback."""
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        mock_lm.history = [{"cost": 1.0, "usage": {"total_tokens": 50000}}]

        mock = MockInterpreter(responses=["exploring..."])
        rlm = RLM("query -> answer", max_iterations=5, max_cost=0.0,
                   sub_lm=mock_lm, interpreter=mock)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore", "code": "print('exploring')"},
        ], async_mode=True)
        rlm.extract = make_mock_predictor([
            {"answer": "async_cost_fallback"},
        ], async_mode=True)

        result = await rlm.aforward(query="test")
        assert result.answer == "async_cost_fallback"

    @pytest.mark.asyncio
    async def test_aforward_budget_tool_works(self):
        """Test that budget() tool is accessible in aforward() via MockInterpreter."""
        budget_reports = []

        class BudgetCapturingInterpreter(MockInterpreter):
            def __init__(self):
                super().__init__(responses=["output", FinalOutput({"answer": "done"})])

            def execute(self, code, variables=None):
                if "budget" in self.tools:
                    budget_reports.append(self.tools["budget"]())
                return super().execute(code, variables)

        mock_interp = BudgetCapturingInterpreter()
        rlm = RLM("query -> answer", max_iterations=10, max_llm_calls=30,
                   max_time=60.0, interpreter=mock_interp)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Step 1", "code": "print('a')"},
            {"reasoning": "Done", "code": 'SUBMIT("done")'},
        ], async_mode=True)

        result = await rlm.aforward(query="test")
        assert result.answer == "done"
        assert len(budget_reports) >= 1
        assert "Iterations:" in budget_reports[0]
        assert "Time:" in budget_reports[0]


# ============================================================================
# Tests: bootstrap_trace resilience for non-parse exceptions
# ============================================================================


class TestBootstrapTraceResilience:
    """Test that bootstrap_trace_data handles non-parse exceptions gracefully."""

    def test_runtime_error_captured_as_failed_prediction(self):
        """Test that a RuntimeError from forward() is captured, not propagated."""
        from unittest.mock import patch

        import dspy
        from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data

        class CrashingModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict("query -> answer")

            def forward(self, **kwargs):
                raise RuntimeError("RLM timeout exceeded")

        program = CrashingModule()
        dataset = [
            dspy.Example(query="test1").with_inputs("query"),
            dspy.Example(query="test2").with_inputs("query"),
        ]

        def metric(example, prediction, trace=None):
            if isinstance(prediction, FailedPrediction):
                return 0.0
            return 1.0

        # Mock the Evaluate class to directly call the program
        class DirectEvaluate:
            def __init__(self, **kwargs):
                self.devset = kwargs.get("devset", [])
                self.failure_score = kwargs.get("failure_score", 0)

            def __call__(self, program, metric=None, **kwargs):
                results = []
                for example in self.devset:
                    inputs = {k: example[k] for k in example.inputs()}
                    prediction = program(**inputs)
                    score = metric(example, prediction) if metric else None
                    results.append((example, prediction, score))

                class Result:
                    pass
                r = Result()
                r.results = results
                return r

        with patch("dspy.teleprompt.bootstrap_trace.Evaluate", DirectEvaluate):
            import dspy as _dspy
            with _dspy.context(lm=_dspy.LM(model="openai/gpt-4o-mini"), trace=[]):
                results = bootstrap_trace_data(
                    program=program,
                    dataset=dataset,
                    metric=metric,
                    raise_on_error=False,
                )

        assert len(results) == 2
        for result in results:
            pred = result["prediction"]
            assert isinstance(pred, FailedPrediction)
            assert "RLM timeout exceeded" in pred.completion_text
            # Trace should be preserved (even if empty, it shouldn't be None)
            assert isinstance(result["trace"], list)

    def test_cost_overrun_captured_as_failed_prediction(self):
        """Test that a cost overrun exception is captured with partial trace."""
        from unittest.mock import patch

        import dspy
        from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data

        class CostOverrunModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict("query -> answer")

            def forward(self, **kwargs):
                # Simulate partial work before cost overrun
                # Add something to the trace before crashing
                dspy.settings.trace.append(
                    (self.predictor, kwargs, dspy.Prediction(answer="partial"))
                )
                raise RuntimeError("Cost budget exceeded ($0.55 > $0.50)")

        program = CostOverrunModule()
        dataset = [dspy.Example(query="test").with_inputs("query")]

        def metric(example, prediction, trace=None):
            return 0.0

        class DirectEvaluate:
            def __init__(self, **kwargs):
                self.devset = kwargs.get("devset", [])

            def __call__(self, program, metric=None, **kwargs):
                results = []
                for example in self.devset:
                    inputs = {k: example[k] for k in example.inputs()}
                    prediction = program(**inputs)
                    score = metric(example, prediction) if metric else None
                    results.append((example, prediction, score))

                class Result:
                    pass
                r = Result()
                r.results = results
                return r

        with patch("dspy.teleprompt.bootstrap_trace.Evaluate", DirectEvaluate):
            import dspy as _dspy
            with _dspy.context(lm=_dspy.LM(model="openai/gpt-4o-mini"), trace=[]):
                results = bootstrap_trace_data(
                    program=program,
                    dataset=dataset,
                    metric=metric,
                    raise_on_error=False,
                )

        assert len(results) == 1
        pred = results[0]["prediction"]
        assert isinstance(pred, FailedPrediction)
        assert "Cost budget exceeded" in pred.completion_text
        # The partial trace from before the crash should be preserved
        trace = results[0]["trace"]
        assert isinstance(trace, list)
        assert len(trace) == 1  # the one entry we appended before crashing

    def test_keyboard_interrupt_not_swallowed(self):
        """Test that KeyboardInterrupt is NOT caught (it's BaseException, not Exception)."""
        from unittest.mock import patch

        import dspy
        from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

        class InterruptingModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict("query -> answer")

            def forward(self, **kwargs):
                raise KeyboardInterrupt()

        program = InterruptingModule()
        dataset = [dspy.Example(query="test").with_inputs("query")]

        class DirectEvaluate:
            def __init__(self, **kwargs):
                self.devset = kwargs.get("devset", [])

            def __call__(self, program, metric=None, **kwargs):
                results = []
                for example in self.devset:
                    inputs = {k: example[k] for k in example.inputs()}
                    prediction = program(**inputs)
                    results.append((example, prediction, None))

                class Result:
                    pass
                r = Result()
                r.results = results
                return r

        with patch("dspy.teleprompt.bootstrap_trace.Evaluate", DirectEvaluate):
            import dspy as _dspy
            with _dspy.context(lm=_dspy.LM(model="openai/gpt-4o-mini"), trace=[]):
                with pytest.raises(KeyboardInterrupt):
                    bootstrap_trace_data(
                        program=program,
                        dataset=dataset,
                        raise_on_error=False,
                    )


# ============================================================================
# Tests: LocalInterpreter output_fields via setter
# ============================================================================


class TestLocalInterpreterOutputFieldsSetter:
    """Test LocalInterpreter output_fields configuration paths."""

    def test_output_fields_set_after_init(self):
        """Test that output_fields can be set after construction and SUBMIT uses them."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        interp.output_fields = [{"name": "answer"}, {"name": "confidence"}]
        interp.start()

        result = interp.execute('SUBMIT("hello", "high")')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "hello", "confidence": "high"}

    def test_output_fields_none_defaults_to_single_output(self):
        """Test that without output_fields, single-arg SUBMIT wraps in 'output' key."""
        from dspy.primitives.local_interpreter import LocalInterpreter

        interp = LocalInterpreter()
        interp.start()

        result = interp.execute('SUBMIT("hello")')
        assert isinstance(result, FinalOutput)
        assert result.output == {"output": "hello"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
