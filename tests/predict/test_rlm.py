"""
Tests for the RLM (Recursive Language Model) module.

Test organization:
- Unit tests (no Deno required): MockSandbox, RLM formatting, signatures
- Integration tests (@pytest.mark.integration): LocalSandbox with Deno
"""

import random

import pytest

from dspy.predict.rlm import RLM, REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.local_sandbox import LocalSandbox
from dspy.primitives.mock_sandbox import MockSandbox
from dspy.primitives.sandbox import FinalAnswerResult, SandboxError

# ============================================================================
# Unit Tests: MockSandbox
# ============================================================================


class TestMockSandbox:
    """Unit tests for MockSandbox."""

    def test_start_is_idempotent(self):
        """Test that start() can be called multiple times safely."""
        mock = MockSandbox(responses=["result"])
        mock.start()  # First call
        mock.start()  # Second call - should be safe
        assert mock.execute("code") == "result"

    def test_scripted_responses(self):
        """Test that MockSandbox returns scripted responses in order."""
        mock = MockSandbox(responses=["first", "second", "third"])
        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

    def test_returns_empty_when_exhausted(self):
        """Test that MockSandbox returns empty string when responses exhausted."""
        mock = MockSandbox(responses=["only"])
        assert mock.execute("code1") == "only"
        assert mock.execute("code2") == ""

    def test_records_call_history(self):
        """Test that MockSandbox records call history."""
        mock = MockSandbox(responses=["resp"])
        mock.execute("print(1)", variables={"x": 10})
        assert len(mock.call_history) == 1
        assert mock.call_history[0] == ("print(1)", {"x": 10})

    def test_call_count(self):
        """Test that MockSandbox tracks call count."""
        mock = MockSandbox(responses=["a", "b", "c"])
        assert mock.call_count == 0
        mock.execute("code1")
        assert mock.call_count == 1
        mock.execute("code2")
        assert mock.call_count == 2

    def test_returns_final_answer_result(self):
        """Test that MockSandbox can return FinalAnswerResult."""
        mock = MockSandbox(responses=[
            "exploring",
            FinalAnswerResult("42"),
        ])
        result1 = mock.execute("print(len(data))")
        assert result1 == "exploring"

        result2 = mock.execute("FINAL('42')")
        assert isinstance(result2, FinalAnswerResult)
        assert result2.answer == "42"

    def test_raises_exception_from_responses(self):
        """Test that MockSandbox raises exceptions from responses."""
        mock = MockSandbox(responses=[
            "ok",
            SandboxError("undefined variable"),
        ])
        assert mock.execute("code1") == "ok"
        with pytest.raises(SandboxError) as exc_info:
            mock.execute("code2")
        assert "undefined variable" in str(exc_info.value)

    def test_custom_execute_fn(self):
        """Test MockSandbox with custom execute function."""
        def custom_exec(code, variables):
            if "FINAL" in code:
                return FinalAnswerResult("done")
            return f"executed: {len(code)} chars"

        mock = MockSandbox(execute_fn=custom_exec)
        result1 = mock.execute("print(1)")
        assert "executed:" in result1

        result2 = mock.execute("FINAL('done')")
        assert isinstance(result2, FinalAnswerResult)

    def test_shutdown_prevents_further_execution(self):
        """Test that shutdown prevents further execution."""
        mock = MockSandbox(responses=["resp"])
        mock.shutdown()
        with pytest.raises(SandboxError):
            mock.execute("code")

    def test_reset(self):
        """Test that reset clears state."""
        mock = MockSandbox(responses=["a", "b"])
        mock.execute("code1")
        mock.shutdown()

        mock.reset()
        mock.responses = ["x", "y"]
        assert mock.call_count == 0
        assert mock.call_history == []
        assert mock.execute("code") == "x"

    def test_context_manager(self):
        """Test MockSandbox as context manager."""
        with MockSandbox(responses=["resp"]) as mock:
            assert mock.execute("code") == "resp"
        # After exiting, should be shutdown
        with pytest.raises(SandboxError):
            mock.execute("code")


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
        assert "llm_query" in rlm.tools
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

        rlm = RLM("context -> answer", max_iterations=5, tools={"custom_tool": custom_tool})
        assert "llm_query" in rlm.tools
        assert "custom_tool" in rlm.tools

    def test_interpreter_parameter(self):
        """Test RLM accepts interpreter parameter."""
        mock = MockSandbox()
        rlm = RLM("context -> answer", interpreter=mock)
        assert rlm._interpreter is mock

    def test_max_llm_calls_parameter(self):
        """Test RLM accepts max_llm_calls parameter."""
        rlm = RLM("context -> answer", max_llm_calls=100)
        assert rlm.max_llm_calls == 100

    def test_default_max_llm_calls(self):
        """Test RLM has default max_llm_calls of 50."""
        rlm = RLM("context -> answer")
        assert rlm.max_llm_calls == 50

    def test_sub_lm_parameter(self):
        """Test RLM accepts sub_lm parameter."""
        import dspy
        mock_lm = dspy.LM("openai/gpt-4o-mini")
        rlm = RLM("context -> answer", sub_lm=mock_lm)
        assert rlm.sub_lm is mock_lm

    def test_sub_lm_default_none(self):
        """Test RLM defaults sub_lm to None."""
        rlm = RLM("context -> answer")
        assert rlm.sub_lm is None

    def test_forward_validates_required_inputs(self):
        """Test that forward() raises ValueError for missing required inputs."""
        mock = MockSandbox(responses=["result"])
        rlm = RLM("context, query -> answer", max_iterations=3, interpreter=mock)

        with pytest.raises(ValueError, match="Missing required input"):
            rlm.forward(context="some context")  # Missing 'query'

    def test_forward_validates_all_missing_inputs(self):
        """Test that forward() reports all missing inputs."""
        mock = MockSandbox(responses=["result"])
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

    def test_format_output_truncation(self):
        """Test that long output is truncated."""
        rlm = RLM("context -> answer", max_output_chars=100)
        formatted = rlm._format_output("x" * 200)
        assert "truncated" in formatted.lower()

    def test_format_variable_info_string(self):
        """Test variable info formatting for string value using REPLVariable."""
        var = REPLVariable.from_value("context", "Hello world", preview_chars=5)
        formatted = var.format()
        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "11" in formatted  # length
        assert "Hello" in formatted
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
        """Test REPLEntry output truncation."""
        entry = REPLEntry(code="print('x' * 1000)", output="x" * 1000)
        formatted = entry.format(index=0, max_output_chars=50)
        assert "truncated" in formatted

    def test_repl_variable_from_value(self):
        """Test REPLVariable.from_value() factory."""
        var = REPLVariable.from_value("test", "hello world")
        assert var.name == "test"
        assert var.type_name == "str"
        assert var.total_length == 11
        assert "hello world" in var.preview

    def test_repl_variable_truncation(self):
        """Test REPLVariable preview truncation."""
        var = REPLVariable.from_value("big", "x" * 1000, preview_chars=50)
        assert len(var.preview) == 53  # 50 + "..."
        assert var.preview.endswith("...")

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


class TestRLMDynamicSignature:
    """Tests for the dynamically built RLM signatures."""

    def test_action_signature_has_required_fields(self):
        """Test that action signature has required fields."""
        rlm = RLM("context -> answer")
        action_sig = rlm.generate_action.signature
        assert "variables_info" in action_sig.input_fields
        assert "repl_history" in action_sig.input_fields
        assert "reasoning" in action_sig.output_fields
        assert "code" in action_sig.output_fields

    def test_extract_signature_has_required_fields(self):
        """Test that extract signature has required fields."""
        rlm = RLM("context -> answer")
        extract_sig = rlm.extract.signature
        assert "variables_info" in extract_sig.input_fields
        assert "repl_history" in extract_sig.input_fields
        assert "answer" in extract_sig.output_fields

    def test_extract_signature_has_multiple_outputs(self):
        """Test that extract signature includes all output fields."""
        rlm = RLM("document, question -> summary, key_facts, confidence")
        extract_sig = rlm.extract.signature
        assert "summary" in extract_sig.output_fields
        assert "key_facts" in extract_sig.output_fields
        assert "confidence" in extract_sig.output_fields

    def test_action_signature_mentions_llm_query(self):
        """Test that action signature instructions mention llm_query."""
        rlm = RLM("context -> answer")
        instructions = rlm.generate_action.signature.instructions
        assert "llm_query" in instructions

    def test_action_signature_mentions_llm_query_batched(self):
        """Test that action signature instructions mention llm_query_batched."""
        rlm = RLM("context -> answer")
        instructions = rlm.generate_action.signature.instructions
        assert "llm_query_batched" in instructions

    def test_rlm_has_llm_query_batched_tool(self):
        """Test that RLM includes llm_query_batched in default tools."""
        rlm = RLM("context -> answer")
        assert "llm_query_batched" in rlm.tools
        assert callable(rlm.tools["llm_query_batched"])

    def test_action_signature_mentions_final(self):
        """Test that action signature instructions mention FINAL."""
        rlm = RLM("context -> answer")
        instructions = rlm.generate_action.signature.instructions
        assert "FINAL" in instructions

    def test_action_signature_mentions_input_vars(self):
        """Test that action signature mentions the input variable names."""
        rlm = RLM("document, question -> answer")
        instructions = rlm.generate_action.signature.instructions
        assert "`document`" in instructions
        assert "`question`" in instructions

    def test_action_signature_mentions_output_vars(self):
        """Test that action signature mentions the output variable names."""
        rlm = RLM("context, query -> summary, analysis")
        instructions = rlm.generate_action.signature.instructions
        assert "`summary`" in instructions
        assert "`analysis`" in instructions


# ============================================================================
# Integration Tests: LocalSandbox (require Deno)
# ============================================================================


@pytest.mark.integration
class TestLocalSandbox:
    """Integration tests for the secure sandbox with tool support."""

    def test_start_prewarms_sandbox(self):
        """Test that start() pre-warms the sandbox."""
        interp = LocalSandbox()
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
        interp = LocalSandbox()
        try:
            interp.start()
            first_process = interp.deno_process
            interp.start()  # Second call - should be idempotent
            assert interp.deno_process is first_process  # Same process
        finally:
            interp.shutdown()

    def test_basic_execution(self):
        """Test basic code execution."""
        with LocalSandbox() as interp:
            result = interp.execute("print(1 + 1)")
            assert "2" in result

    def test_variable_injection(self):
        """Test variable injection."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute(
                "print(x + y)",
                variables={"x": 10, "y": 5}
            )
            assert "15" in result

    def test_tool_call_kwargs(self):
        """Test tool call with keyword arguments."""
        def echo(message: str = "") -> str:
            return f"Echo: {message}"

        with LocalSandbox(tools={"echo": echo}) as interp:
            result = interp.execute('print(echo(message="hello"))')
            assert "Echo: hello" in result

    def test_tool_call_positional(self):
        """Test tool call with positional arguments."""
        def greet(name: str) -> str:
            return f"Hello: {name}"

        with LocalSandbox(tools={"greet": greet}) as interp:
            result = interp.execute('print(greet("world"))')
            assert "Hello: world" in result

    def test_multiple_tools(self):
        """Test multiple tools."""
        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        def multiply(a: int = 0, b: int = 0) -> str:
            return str(a * b)

        with LocalSandbox(tools={"add": add, "multiply": multiply}) as interp:
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

        with LocalSandbox(tools={"batch_process": batch_process}) as interp:
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

        with LocalSandbox(tools={"get_info": get_info}) as interp:
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
        with LocalSandbox(tools={}) as interp:
            interp.execute("x = 10")
            result = interp.execute("print(x + 5)")
            assert "15" in result

    def test_syntax_error(self):
        """Test syntax error handling."""
        with LocalSandbox(tools={}) as interp:
            with pytest.raises(SyntaxError):
                interp.execute("def incomplete(")

    def test_runtime_error(self):
        """Test runtime error handling."""
        with LocalSandbox(tools={}) as interp:
            with pytest.raises(SandboxError):
                interp.execute("undefined_variable")

    def test_final_answer(self):
        """Test FINAL() returns FinalAnswerResult with dict format."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute('FINAL("the answer is 42")')
            assert isinstance(result, FinalAnswerResult)
            # Default FINAL wraps value in {"answer": value}
            assert result.answer == {"answer": "the answer is 42"}

    def test_final_var(self):
        """Test FINAL_VAR() returns variable value as FinalAnswerResult with dict format."""
        with LocalSandbox(tools={}) as interp:
            interp.execute("my_answer = 'computed result'")
            result = interp.execute('FINAL_VAR("my_answer")')
            assert isinstance(result, FinalAnswerResult)
            # Default FINAL_VAR wraps value in {"answer": value}
            assert result.answer == {"answer": "computed result"}

    def test_final_var_not_found(self):
        """Test FINAL_VAR() raises error for undefined variable."""
        with LocalSandbox(tools={}) as interp:
            with pytest.raises(SandboxError) as exc_info:
                interp.execute('FINAL_VAR("nonexistent")')
            assert "not found" in str(exc_info.value)

    @pytest.mark.parametrize("value,expected", [
        (None, None),
        ("hello", "hello"),
        (True, True),
        (False, False),
        (42, 42),
        (-17, -17),
        (3.14, 3.14),
        (0.0, 0.0),
        ([1, 2, 3], [1, 2, 3]),
        ([], []),
        (["a", "b"], ["a", "b"]),
        ({"key": "value"}, {"key": "value"}),
        ({}, {}),
        ({"nested": {"a": 1}}, {"nested": {"a": 1}}),
    ])
    def test_final_answer_types(self, value, expected):
        """Test FINAL() correctly returns values of all supported types."""
        with LocalSandbox(tools={}) as interp:
            code = f"FINAL({value!r})"
            result = interp.execute(code)
            assert isinstance(result, FinalAnswerResult)
            # Default FINAL wraps value in {"answer": value}
            assert result.answer == {"answer": expected}

    @pytest.mark.parametrize("value,expected", [
        (None, None),
        ("hello", "hello"),
        (True, True),
        (False, False),
        (42, 42),
        (-17, -17),
        (3.14, 3.14),
        (0.0, 0.0),
        ([1, 2, 3], [1, 2, 3]),
        ([], []),
        (["a", "b"], ["a", "b"]),
        ({"key": "value"}, {"key": "value"}),
        ({}, {}),
        ({"nested": {"a": 1}}, {"nested": {"a": 1}}),
    ])
    def test_final_var_types(self, value, expected):
        """Test FINAL_VAR() correctly returns values of all supported types."""
        with LocalSandbox(tools={}) as interp:
            interp.execute(f"my_var = {value!r}")
            result = interp.execute('FINAL_VAR("my_var")')
            assert isinstance(result, FinalAnswerResult)
            # Default FINAL_VAR wraps value in {"answer": value}
            assert result.answer == {"answer": expected}

    def test_final_answer_empty_string(self):
        """Test FINAL() with empty string."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute('FINAL("")')
            assert isinstance(result, FinalAnswerResult)
            assert result.answer == {"answer": ""}

    def test_final_answer_unicode(self):
        """Test FINAL() with unicode characters."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute('FINAL("こんにちは 🚀")')
            assert isinstance(result, FinalAnswerResult)
            assert result.answer == {"answer": "こんにちは 🚀"}

    def test_final_answer_special_chars(self):
        """Test FINAL() with special characters and escapes."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute(r'FINAL("line1\nline2\ttab")')
            assert isinstance(result, FinalAnswerResult)
            assert result.answer == {"answer": "line1\nline2\ttab"}

    def test_final_var_complex_nested(self):
        """Test FINAL_VAR() with complex nested structures."""
        with LocalSandbox(tools={}) as interp:
            interp.execute('data = {"list": [1, 2, {"nested": True}], "count": 42}')
            result = interp.execute('FINAL_VAR("data")')
            assert isinstance(result, FinalAnswerResult)
            assert result.answer == {"answer": {"list": [1, 2, {"nested": True}], "count": 42}}


@pytest.mark.integration
class TestSandboxSecurity:
    """Integration tests for sandbox security restrictions."""

    def test_no_network_access(self):
        """Test that network access is blocked."""
        with LocalSandbox(tools={}) as interp:
            with pytest.raises(SandboxError) as exc_info:
                interp.execute("""
from pyodide.http import pyfetch
import asyncio
asyncio.get_event_loop().run_until_complete(pyfetch("https://example.com"))
""")
            assert "net access" in str(exc_info.value).lower() or "allow-net" in str(exc_info.value).lower()

    def test_imports_work(self):
        """Test that standard library imports work."""
        with LocalSandbox(tools={}) as interp:
            result = interp.execute("""
import json
import re
from collections import Counter
data = {"key": "value"}
print(json.dumps(data))
""")
            assert "key" in result


# ============================================================================
# Unit Tests: RLM Type Validation (parse_value from adapters)
# ============================================================================


class TestRLMTypeValidation:
    """Tests for RLM type validation using parse_value from adapters."""

    def test_parse_value_literal_valid(self):
        """Test parse_value accepts valid Literal values."""
        from typing import Literal

        from dspy.adapters.utils import parse_value

        result = parse_value("yes", Literal["yes", "no"])
        assert result == "yes"

    def test_parse_value_literal_invalid(self):
        """Test parse_value rejects invalid Literal values."""
        from typing import Literal

        from dspy.adapters.utils import parse_value

        with pytest.raises(ValueError):
            parse_value("maybe", Literal["yes", "no"])

    def test_parse_value_int_from_string(self):
        """Test parse_value coerces string to int."""
        from dspy.adapters.utils import parse_value

        result = parse_value("42", int)
        assert result == 42
        assert isinstance(result, int)

    def test_parse_value_int_from_int(self):
        """Test parse_value preserves int from int."""
        from dspy.adapters.utils import parse_value

        result = parse_value(42, int)
        assert result == 42
        assert isinstance(result, int)

    def test_parse_value_float_from_string(self):
        """Test parse_value coerces string to float."""
        from dspy.adapters.utils import parse_value

        result = parse_value("3.14", float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_value_float_from_float(self):
        """Test parse_value preserves float from float."""
        from dspy.adapters.utils import parse_value

        result = parse_value(3.14, float)
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_value_bool_from_string(self):
        """Test parse_value coerces string to bool."""
        from dspy.adapters.utils import parse_value

        result = parse_value("true", bool)
        assert result is True

    def test_parse_value_bool_from_bool(self):
        """Test parse_value preserves bool from bool."""
        from dspy.adapters.utils import parse_value

        result = parse_value(True, bool)
        assert result is True

    def test_parse_value_list_from_string(self):
        """Test parse_value coerces JSON string to list."""
        from dspy.adapters.utils import parse_value

        result = parse_value("[1, 2, 3]", list[int])
        assert result == [1, 2, 3]

    def test_parse_value_list_from_list(self):
        """Test parse_value validates list from list."""
        from dspy.adapters.utils import parse_value

        result = parse_value([1, 2, 3], list[int])
        assert result == [1, 2, 3]

    def test_parse_value_dict_from_string(self):
        """Test parse_value coerces JSON string to dict."""
        from dspy.adapters.utils import parse_value

        result = parse_value('{"a": 1}', dict[str, int])
        assert result == {"a": 1}

    def test_parse_value_dict_from_dict(self):
        """Test parse_value validates dict from dict."""
        from dspy.adapters.utils import parse_value

        result = parse_value({"a": 1}, dict[str, int])
        assert result == {"a": 1}

    def test_parse_value_enum_by_value(self):
        """Test parse_value finds enum by value."""
        from enum import Enum

        from dspy.adapters.utils import parse_value

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = parse_value("red", Color)
        assert result == Color.RED

    def test_parse_value_enum_by_name(self):
        """Test parse_value finds enum by name."""
        from enum import Enum

        from dspy.adapters.utils import parse_value

        class Color(Enum):
            RED = "red"
            BLUE = "blue"

        result = parse_value("RED", Color)
        assert result == Color.RED

    def test_parse_value_optional_none(self):
        """Test parse_value handles Optional with None."""
        from dspy.adapters.utils import parse_value

        result = parse_value(None, str | None)
        assert result is None

    def test_parse_value_optional_value(self):
        """Test parse_value handles Optional with value."""
        from dspy.adapters.utils import parse_value

        result = parse_value("hello", str | None)
        assert result == "hello"


# ============================================================================
# Unit Tests: RLM Type Coercion with MockSandbox (no Deno required)
# ============================================================================


class TestRLMTypeCoercionMock:
    """Unit tests for RLM type coercion using MockSandbox (no Deno required)."""

    def test_rlm_int_output_mock(self):
        """Test RLM returns int when signature expects int (MockSandbox)."""
        from dspy.primitives.prediction import Prediction

        # FinalAnswerResult must use dict format with field names matching signature
        mock = MockSandbox(responses=[FinalAnswerResult({"count": 42})])
        rlm = RLM("query -> count: int", max_iterations=3, interpreter=mock)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return count", code="FINAL(42)")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)

    def test_rlm_literal_output_mock(self):
        """Test RLM returns valid Literal value (MockSandbox)."""
        from dspy.primitives.prediction import Prediction

        # FinalAnswerResult must use dict format with field names matching signature
        mock = MockSandbox(responses=[FinalAnswerResult({"answer": "yes"})])
        rlm = RLM("query -> answer: Literal['yes', 'no']", max_iterations=3, interpreter=mock)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Answer yes", code='FINAL("yes")')

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="is it yes?")
        assert result.answer == "yes"

    def test_rlm_list_output_mock(self):
        """Test RLM returns list when signature expects list (MockSandbox)."""
        from dspy.primitives.prediction import Prediction

        # FinalAnswerResult must use dict format with field names matching signature
        mock = MockSandbox(responses=[FinalAnswerResult({"numbers": [1, 2, 3]})])
        rlm = RLM("query -> numbers: list[int]", max_iterations=3, interpreter=mock)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return list", code="FINAL([1, 2, 3])")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="get numbers")
        assert result.numbers == [1, 2, 3]
        assert isinstance(result.numbers, list)

    def test_rlm_type_error_retries_mock(self):
        """Test RLM retries when type validation fails (MockSandbox)."""
        from dspy.primitives.prediction import Prediction

        # MockSandbox returns responses in order
        # FinalAnswerResult must use dict format with field names matching signature
        mock = MockSandbox(responses=[
            FinalAnswerResult({"answer": "maybe"}),  # Invalid for Literal
            FinalAnswerResult({"answer": "yes"}),    # Valid
        ])
        rlm = RLM("query -> answer: Literal['yes', 'no']", max_iterations=5, interpreter=mock)

        call_count = [0]

        class MockPredictor:
            def __call__(self, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return Prediction(reasoning="Try maybe", code='FINAL("maybe")')
                else:
                    return Prediction(reasoning="Try yes", code='FINAL("yes")')

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="is it yes?")
        assert result.answer == "yes"
        assert call_count[0] >= 2  # Should have retried


# ============================================================================
# Integration Tests: RLM Type Coercion with LocalSandbox
# ============================================================================


@pytest.mark.integration
class TestRLMTypeCoercion:
    """Tests for RLM type coercion through full forward pass with LocalSandbox.

    Note: These tests let RLM create its own LocalSandbox so it can register
    typed output_fields for FINAL based on the signature.
    """

    def test_rlm_int_output_from_int_final(self):
        """Test RLM returns int when signature expects int and FINAL returns int."""
        from dspy.primitives.prediction import Prediction

        # Let RLM create its own sandbox with output_fields
        rlm = RLM("query -> count: int", max_iterations=3)

        # Mock generate_action to return code that calls FINAL with an int
        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="I'll return the count", code="FINAL(42)")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)

    def test_rlm_literal_output_valid(self):
        """Test RLM returns valid Literal value."""
        from dspy.primitives.prediction import Prediction

        rlm = RLM("query -> answer: Literal['yes', 'no']", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Answer is yes", code='FINAL("yes")')

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="is it yes?")
        assert result.answer == "yes"

    def test_rlm_list_output_from_list_final(self):
        """Test RLM returns list when signature expects list and FINAL returns list."""
        from dspy.primitives.prediction import Prediction

        # Use 'numbers' instead of 'items' to avoid conflict with Prediction.items() method
        rlm = RLM("query -> numbers: list[int]", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return list", code="FINAL([1, 2, 3])")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="get items")
        assert result.numbers == [1, 2, 3]
        assert isinstance(result.numbers, list)

    def test_rlm_dict_output_from_dict_final(self):
        """Test RLM returns dict when signature expects dict and FINAL returns dict."""
        from dspy.primitives.prediction import Prediction

        rlm = RLM("query -> data: dict[str, str]", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return dict", code='FINAL({"key": "value"})')

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="get data")
        assert result.data == {"key": "value"}
        assert isinstance(result.data, dict)

    def test_rlm_float_output(self):
        """Test RLM returns float when signature expects float."""
        from dspy.primitives.prediction import Prediction

        rlm = RLM("query -> score: float", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return score", code="FINAL(3.14)")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="get score")
        assert result.score == 3.14
        assert isinstance(result.score, float)

    def test_rlm_bool_output(self):
        """Test RLM returns bool when signature expects bool."""
        from dspy.primitives.prediction import Prediction

        rlm = RLM("query -> valid: bool", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(reasoning="Return bool", code="FINAL(True)")

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="is valid?")
        assert result.valid is True
        assert isinstance(result.valid, bool)

    def test_rlm_final_var_int_output(self):
        """Test RLM FINAL_VAR correctly extracts typed value."""
        from dspy.primitives.prediction import Prediction

        rlm = RLM("query -> count: int", max_iterations=3)

        class MockPredictor:
            def __call__(self, **kwargs):
                return Prediction(
                    reasoning="Compute and return",
                    code='result = 42\nFINAL_VAR("result")'
                )

        rlm.generate_action = MockPredictor()

        result = rlm.forward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)


# ============================================================================
# Integration Tests: RLM with DummyLM and LocalSandbox
# ============================================================================


@pytest.mark.integration
class TestRLMWithDummyLM:
    """End-to-end tests using DummyLM with RLM and LocalSandbox.

    Note: These tests let RLM create its own LocalSandbox so it can register
    typed output_fields for FINAL based on the signature.
    """

    def test_simple_computation_e2e(self):
        """Test full RLM pipeline: DummyLM -> RLM -> LocalSandbox -> result."""
        import dspy
        from dspy.utils.dummies import DummyLM

        # DummyLM returns responses with reasoning and code fields
        lm = DummyLM([
            {"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nFINAL(result)"},
        ])

        with dspy.context(lm=lm):
            # Let RLM create its own sandbox with output_fields
            rlm = RLM("query -> answer: int", max_iterations=3)
            result = rlm.forward(query="What is 2 + 3?")

            assert result.answer == 5
            assert isinstance(result.answer, int)

    def test_multi_turn_computation_e2e(self):
        """Test RLM with multiple turns before FINAL."""
        import dspy
        from dspy.utils.dummies import DummyLM

        # First turn: explore the data, second turn: compute and return
        lm = DummyLM([
            {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
            {"reasoning": "Now compute and return", "code": "y = x * 2\nFINAL(y)"},
        ])

        with dspy.context(lm=lm):
            rlm = RLM("query -> answer: int", max_iterations=5)
            result = rlm.forward(query="Double ten")

            assert result.answer == 20
            # Verify trajectory has 2 entries
            assert len(result.trajectory) == 2

    def test_with_input_variables_e2e(self):
        """Test RLM with input variables passed to sandbox."""
        import dspy
        from dspy.utils.dummies import DummyLM

        lm = DummyLM([
            {"reasoning": "Sum the numbers in the list", "code": "FINAL(sum(numbers))"},
        ])

        with dspy.context(lm=lm):
            rlm = RLM("numbers: list[int] -> total: int", max_iterations=3)
            result = rlm.forward(numbers=[1, 2, 3, 4, 5])

            assert result.total == 15

    def test_with_tool_e2e(self):
        """Test RLM calling a host-side tool through the sandbox."""
        import dspy
        from dspy.utils.dummies import DummyLM

        def lookup(key: str) -> str:
            return {"apple": "red", "banana": "yellow"}.get(key, "unknown")

        lm = DummyLM([
            {"reasoning": "Look up the color of apple", "code": 'color = lookup(key="apple")\nFINAL(color)'},
        ])

        with dspy.context(lm=lm):
            # Pass custom tools to RLM, which will pass them to its LocalSandbox
            rlm = RLM("fruit -> color: str", max_iterations=3, tools={"lookup": lookup})
            result = rlm.forward(fruit="apple")

            assert result.color == "red"


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
# Benchmark Data Generators (for manual testing)
# ============================================================================


def generate_needle_haystack(
    num_lines: int = 10000,
    needle_value: str | None = None,
    needle_position: float = 0.5,
) -> tuple[str, str]:
    """Generate a needle-in-haystack benchmark."""
    if needle_value is None:
        needle_value = str(random.randint(1000000, 9999999))

    random_words = ["blah", "random", "text", "data", "content", "info", "sample", "test"]
    lines = []

    for _ in range(num_lines):
        num_words = random.randint(3, 8)
        line = " ".join(random.choice(random_words) for _ in range(num_words))
        lines.append(line)

    needle_pos = int(num_lines * needle_position)
    lines[needle_pos] = f"The magic number is {needle_value}"

    return "\n".join(lines), needle_value


def generate_label_counting_context(
    num_items: int = 10,
    labels: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Generate a label counting benchmark (Oolong-style)."""
    if labels is None:
        labels = ["spam", "ham"]

    items = []
    counts = dict.fromkeys(labels, 0)

    for i in range(num_items):
        label = random.choice(labels)
        counts[label] += 1
        text = f"Item {i+1}: This is a {label} message with some content."
        items.append(text)

    context = "\n".join(items)
    return context, counts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
