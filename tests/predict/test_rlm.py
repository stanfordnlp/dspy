"""
Tests for the RLM (Recursive Language Model) module.

Test organization:
- Unit tests (no Deno required): MockInterpreter, RLM formatting, signatures
- Integration tests (@pytest.mark.deno): PythonInterpreter with Deno
"""

import base64
from contextlib import contextmanager
from pathlib import Path

import pytest

import dspy
from dspy.adapters.types.tool import Tool
from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory

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
        rlm = RLM("context, query -> answer", max_iters=5)
        assert rlm.max_iters == 5
        assert rlm.generate_action is not None
        assert rlm.extract is not None
        assert rlm.tools == {}  # No user tools provided
        assert "context" in rlm.signature.input_fields
        assert "query" in rlm.signature.input_fields
        assert "answer" in rlm.signature.output_fields

    def test_custom_signature(self):
        """Test RLM with custom signature."""
        rlm = RLM("document, question -> summary, key_facts", max_iters=5)
        assert "document" in rlm.signature.input_fields
        assert "question" in rlm.signature.input_fields
        assert "summary" in rlm.signature.output_fields
        assert "key_facts" in rlm.signature.output_fields

    def test_custom_tools(self):
        """Test RLM with custom tools."""
        def custom_tool(x: str = "") -> str:
            return x.upper()

        rlm = RLM("context -> answer", max_iters=5, tools=[custom_tool])
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

    def test_tool_validation_rejects_python_keyword(self):
        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, name="for")
        with pytest.raises(ValueError, match="not a keyword"):
            RLM("context -> answer", tools=[tool])

    @pytest.mark.parametrize("tool_name", ["llm_query", "llm_query_batched", "__dspy_llm_query_batched", "__dspy_replay_llm_query", "__dspy_frames", "SUBMIT", "print"])
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

    def test_duplicate_tool_names_rejected(self):
        def first() -> str:
            return "first"

        def second() -> str:
            return "second"

        with pytest.raises(ValueError, match="Duplicate tool name 'lookup'"):
            RLM("context -> answer", tools=[Tool(first, name="lookup"), Tool(second, name="lookup")])

    @pytest.mark.parametrize("input_name", ["llm_query", "llm_query_batched", "__dspy_frames", "SUBMIT", "print"])
    def test_input_names_cannot_shadow_sandbox_functions(self, input_name):
        with pytest.raises(ValueError, match="Input fields conflict with built-in sandbox functions"):
            RLM(f"{input_name} -> answer")

    def test_input_name_cannot_shadow_user_tool(self):
        def lookup() -> str:
            return "result"

        with pytest.raises(ValueError, match="Input fields conflict with user tools: \\['lookup'\\]"):
            RLM("lookup -> answer", tools=[lookup])

    @pytest.mark.parametrize("output_name", ["trajectory", "final_reasoning"])
    def test_output_names_cannot_shadow_result_metadata(self, output_name):
        with pytest.raises(ValueError, match=f"Output fields conflict with RLM result metadata: \\['{output_name}'\\]"):
            RLM(f"context -> {output_name}")

    def test_optional_parameters(self):
        """Test RLM optional parameters and their defaults."""
        import dspy

        # Test defaults
        rlm = RLM("context -> answer")
        assert rlm.max_llm_calls == 50
        assert rlm.sub_lm is None
        assert rlm._interpreter_factory is PythonInterpreter

        # Test custom values
        mock_lm = dspy.LM("openai/gpt-4o-mini")
        rlm = RLM(
            "context -> answer",
            max_llm_calls=100,
            sub_lm=mock_lm,
            interpreter_factory=MockInterpreter,
        )
        assert rlm.max_llm_calls == 100
        assert rlm.sub_lm is mock_lm
        assert rlm._interpreter_factory is MockInterpreter

    def test_forward_validates_required_inputs(self):
        """Test that forward() raises ValueError for missing required inputs."""
        # Single missing input
        rlm = RLM("context, query -> answer", max_iters=3)
        with pytest.raises(ValueError, match="Missing required input"):
            rlm.forward(context="some context")  # Missing 'query'

        # Multiple missing inputs - all should be reported
        rlm = RLM("a, b, c -> answer", max_iters=3)
        with pytest.raises(ValueError) as exc_info:
            rlm.forward(a="only a")  # Missing 'b' and 'c'
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_interpreter_instance_is_rejected_as_factory(self):
        with pytest.raises(TypeError, match="first positional argument when calling the module"):
            RLM("context -> answer", interpreter_factory=MockInterpreter())

    def test_constructor_interpreter_keyword_is_removed(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'interpreter'"):
            RLM("context -> answer", interpreter=MockInterpreter())

    def test_factory_return_value_is_validated(self):
        rlm = RLM("query -> answer", interpreter_factory=lambda: None)

        with pytest.raises(TypeError, match="interpreter_factory must return a CodeInterpreter, not NoneType"):
            rlm(query="test")

    def test_keyword_interpreter_override_has_clear_error(self):
        rlm = RLM("query -> answer")

        with pytest.raises(TypeError, match="first positional argument"):
            rlm(query="test", interpreter=MockInterpreter())

    def test_llm_query_returns_legacy_response_text(self):
        from dspy.utils.dummies import DummyLM

        tools = RLM("context -> answer", sub_lm=DummyLM([{"answer": "legacy answer"}]))._make_llm_tools()

        assert tools["llm_query"]("test prompt") == "[[ ## answer ## ]]\nlegacy answer"

    def test_llm_query_returns_typed_response_text(self):
        import dspy
        from dspy.utils.dummies import DummyLM

        tools = RLM("context -> answer", sub_lm=DummyLM([{"answer": "typed answer"}]))._make_llm_tools()

        with dspy.context(experimental=True):
            result = tools["llm_query"]("test prompt")

        assert result == "[[ ## answer ## ]]\ntyped answer"

    def test_llm_query_rejects_unsupported_response_shape(self):
        from unittest.mock import MagicMock

        tools = RLM("context -> answer", sub_lm=MagicMock(return_value="untyped response"))._make_llm_tools()

        with pytest.raises(TypeError, match="Sub-LM must return dspy.LMResponse or a non-empty list"):
            tools["llm_query"]("test prompt")

    def test_llm_query_reports_textless_response_type(self):
        from unittest.mock import MagicMock

        tools = RLM(
            "context -> answer",
            sub_lm=MagicMock(return_value=[{"tool_calls": []}]),
        )._make_llm_tools()

        with pytest.raises(TypeError, match="Sub-LM response must contain text, got NoneType"):
            tools["llm_query"]("test prompt")
    @pytest.mark.parametrize("unexpected_name", ["SUBMIT", "lookup", "tools"])
    def test_forward_rejects_undeclared_inputs_before_interpreter_execution(self, unexpected_name):
        def lookup() -> str:
            return "tool result"

        factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "done"})])
        rlm = RLM("context -> answer", tools=[lookup], interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("done")'},
        ])

        with pytest.raises(ValueError, match=f"Unexpected inputs not declared in the signature: \\['{unexpected_name}'\\]"):
            rlm(context="some context", **{unexpected_name: "shadowed value"})

        assert factory.instances == []

    def test_batched_query_propagates_lm_errors(self):
        from unittest.mock import MagicMock

        import dspy

        mock_lm = MagicMock()
        mock_lm.side_effect = dspy.LMTransportError("LM failed")

        rlm = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)
        tools = rlm._make_llm_tools()

        with pytest.raises(dspy.LMTransportError, match="LM failed"):
            tools["llm_query_batched"](prompts=["test prompt"])

    def test_batched_query_propagates_missing_lm_configuration(self):
        import dspy

        with dspy.context(lm=None):
            tools = RLM("context -> answer")._make_llm_tools()
            with pytest.raises(dspy.LMNotConfiguredError, match="No LM configured"):
                tools["llm_query_batched"](["test prompt"])

    def test_batched_query_propagates_programming_errors(self):
        from unittest.mock import MagicMock

        mock_lm = MagicMock()
        mock_lm.side_effect = TypeError("invalid LM implementation")
        tools = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)._make_llm_tools()

        with pytest.raises(TypeError, match="invalid LM implementation"):
            tools["llm_query_batched"](["test prompt"])

    def test_batched_query_replays_validation_after_successful_prefix(self):
        from unittest.mock import MagicMock

        mock_lm = MagicMock(return_value=["response"])
        tools = RLM("context -> answer", max_llm_calls=10, sub_lm=mock_lm)._make_llm_tools()

        with pytest.raises(ValueError, match="prompt cannot be empty"):
            tools["llm_query_batched"](["first", ""])

        mock_lm.assert_called_once_with("first")

    def test_batched_query_inherits_request_context(self):
        import contextvars

        import dspy

        request_marker = contextvars.ContextVar("request_marker", default="global")

        class TaggedLM:
            def __init__(self, tag):
                self.tag = tag

            def __call__(self, prompt):
                return [f"{self.tag}:{request_marker.get()}"]

        dspy.configure(lm=TaggedLM("global"))
        tools = RLM("context -> answer")._make_llm_tools()

        with dspy.context(lm=TaggedLM("request-local")):
            request_marker.set("request-local")
            assert tools["llm_query"]("one") == "request-local:request-local"
            assert tools["llm_query_batched"](["one", "two"]) == [
                "request-local:request-local",
                "request-local:request-local",
            ]

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


class TestRLMQueryBatchCompiler:
    @staticmethod
    def execute_compiled(code, items):
        batches = []

        def query_batch(prompts):
            batches.append(prompts)
            return [{"value": f"answer:{prompt}"} for prompt in prompts]

        compiled, count = RLM._compile_llm_query_loops(code)
        namespace = {
            "items": items,
            "llm_query": lambda prompt: f"answer:{prompt}",
            "__dspy_llm_query_batched": query_batch,
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }
        exec(compiled, namespace)
        return namespace, batches, compiled, count

    def test_leaves_llm_query_list_comprehension_unchanged(self):
        code = 'answers = [llm_query(f"Analyze: {item}") for item in items]'

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_compiles_prompt_formatting_loop_and_preserves_final_locals(self):
        code = """answers = []
for item in items:
    prompt = "Analyze: {}".format(item)
    answer = llm_query(prompt)
    answers.append(answer)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a", "b"])

        assert count == 1
        assert batches == [["Analyze: a", "Analyze: b"]]
        assert namespace["answers"] == ["answer:Analyze: a", "answer:Analyze: b"]
        assert namespace["item"] == "b"
        assert namespace["prompt"] == "Analyze: b"
        assert namespace["answer"] == "answer:Analyze: b"
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_compiles_singleton_destructuring_target(self):
        code = """answers = []
for (item,) in items:
    answer = llm_query(item)
    answers.append(answer)
"""
        namespace, batches, _, count = self.execute_compiled(code, [["ab"], ["cd"]])

        assert count == 1
        assert batches == [["ab", "cd"]]
        assert namespace["answers"] == ["answer:ab", "answer:cd"]
        assert namespace["item"] == "cd"

    def test_compiles_iteration_local_list_prompt_builder(self):
        code = """answers = []
for batch in items:
    lines = []
    for index, item in enumerate(batch):
        rendered = f"{index + 1}. {item}"
        lines.append(rendered)
    prompt = "\\n".join(lines)
    answer = llm_query(prompt)
    answers.append(answer)
"""
        namespace, batches, _, count = self.execute_compiled(code, [["a", "b"], ["c"]])

        assert count == 1
        assert batches == [["1. a\n2. b", "1. c"]]
        assert namespace["answers"] == ["answer:1. a\n2. b", "answer:1. c"]
        assert namespace["lines"] == ["1. c"]
        assert namespace["rendered"] == "1. c"

    def test_compiles_iteration_local_string_builder(self):
        code = """answers = []
for start in range(0, len(items) + 2, 2):
    batch = items[start:start + 2]
    if len(batch) == 0:
        break
    prompt = "Items:"
    for item in batch:
        if len(item) > 3:
            item = item[:3]
        prompt += f" {item}"
    answer = llm_query(prompt)
    answers.append(answer)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["alpha", "b", "charlie"])

        assert count == 1
        assert batches == [["Items: alp b", "Items: cha"]]
        assert namespace["answers"] == ["answer:Items: alp b", "answer:Items: cha"]
        assert namespace["start"] == 4
        assert namespace["batch"] == []
        assert namespace["prompt"] == "Items: cha"
        assert namespace["item"] == "cha"

    def test_leaves_loop_using_current_block_prompt_helper_unchanged(self):
        code = """def make_prompt(batch):
    text = "Items:"
    for item in batch:
        text += f" {item}"
    return text

answers = []
for batch in items:
    prompt = make_prompt(batch)
    answers.append(llm_query(prompt))
"""
        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize("instrument", ["trace", "profile"])
    def test_runtime_instrumentation_uses_scalar_loop(self, instrument):
        import sys

        code = """answers = []
for item in items:
    answers.append(llm_query(item))
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        calls, batches = [], []
        namespace = {
            "items": ["a", "b"],
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }
        get_instrument, set_instrument = getattr(sys, f"get{instrument}"), getattr(sys, f"set{instrument}")
        previous = get_instrument()

        try:
            set_instrument(lambda *args: None)
            exec(compiled, namespace)
        finally:
            set_instrument(previous)

        assert count == 1
        assert calls == ["a", "b"]
        assert batches == []
        assert namespace["answers"] == ["answer:a", "answer:b"]

    @pytest.mark.parametrize("instrument", ["trace", "profile"])
    def test_rebound_runtime_instrumentation_getter_uses_scalar_without_calling(self, instrument):
        import sys

        code = """answers = []
for item in items:
    answers.append(llm_query(item))
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        calls, batches = [], []
        namespace = {
            "items": ["a", "b"],
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }
        getter_name = f"get{instrument}"
        original = getattr(sys, getter_name)
        called = False

        def retained_getter():
            nonlocal called
            called = True
            raise RuntimeError(f"retained {getter_name} called")

        try:
            setattr(sys, getter_name, retained_getter)
            exec(compiled, namespace)
        finally:
            setattr(sys, getter_name, original)

        assert count == 1
        assert not called
        assert calls == ["a", "b"]
        assert batches == []
        assert namespace["answers"] == ["answer:a", "answer:b"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_persistent_compiler_name_collision_uses_scalar_loop_without_deleting_state(self):
        import ast

        code = """answers = []
for item in items:
    answers.append(llm_query(item))
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        tool_names = {"__dspy_llm_query_batched", "__dspy_replay_llm_query"}
        generated_names = {node.id for node in ast.walk(ast.parse(compiled)) if isinstance(node, ast.Name) and node.id.startswith("__dspy_")} - tool_names
        retained = {name: f"retained:{name}" for name in generated_names}
        calls, batches = [], []
        namespace = {
            **retained,
            "items": ["a", "b"],
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert generated_names
        assert calls == ["a", "b"]
        assert batches == []
        assert namespace["answers"] == ["answer:a", "answer:b"]
        assert {name: namespace[name] for name in generated_names} == retained

    def test_rejects_rebound_current_block_prompt_helper(self):
        code = """state = []
def helper(item):
    return item
def replacement(item):
    state.append("helper")
    return f"{item}|{len(state)}"
helper.__code__ = replacement.__code__
answers = []
for item in items:
    answers.append(llm_query(helper(item)))
    state.append("continuation")
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize("mutation", ["holder.value", "mutate()", "import json"])
    def test_rejects_effectful_code_before_current_block_prompt_helper_use(self, mutation):
        code = f"""def helper(item):
    return item
{mutation}
answers = []
for item in items:
    answers.append(llm_query(helper(item)))
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_rejects_class_registry_reflection(self):
        code = """def helper(item):
    return item
answers = []
for item in items:
    answers.append(llm_query(helper(item)))
    str.__base__.__subclasses__()
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_rejects_frame_builtins_reflection(self):
        code = """answers = []
for item in items:
    answers.append(llm_query(item))
    try:
        1 / 0
    except Exception as error:
        error.__traceback__.tb_frame.f_builtins["__im" + "port__"]("pydoc")
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize(("constructor", "empty"), [("dict", "{}"), ("list", "[]")])
    def test_rejects_shadowed_owned_value_constructor_before_prompt_helper_use(self, constructor, empty):
        code = f"""def helper(item):
    return item
def {constructor}():
    mutate()
    return {empty}
owned = {constructor}()
answers = []
for item in items:
    answers.append(llm_query(helper(item)))
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_rejects_current_block_prompt_helper_defined_after_use(self):
        code = """answers = []
for item in items:
    answers.append(llm_query(helper(item)))
def helper(item):
    return item
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize(
        "setup, mutation",
        [
            ("", 'replacement.__globals__["helper"].__code__ = replacement.__code__'),
            ("", 'replacement.__getattribute__("__globals__")["helper"].__code__ = replacement.__code__'),
            ("", 'object.__getattribute__(replacement, "__globals__")["helper"].__code__ = replacement.__code__'),
            ("import inspect\n", 'inspect.currentframe().f_globals["helper"].__code__ = replacement.__code__'),
            ("import builtins\n", 'builtins.globals()["helper"].__code__ = replacement.__code__'),
            ("import operator\n", 'operator.attrgetter("__globals__")(replacement)["helper"].__code__ = replacement.__code__'),
            ("import operator\n", 'operator.methodcaller("__getattribute__", "__globals__")(replacement)["helper"].__code__ = replacement.__code__'),
            ("import sys\n", "sys.modules[__name__].helper = replacement"),
            ("", 'print.__self__.__import__("pydoc").locate("dspy_rlm_builtin_self_probe").helper = replacement'),
        ],
    )
    def test_rejects_dynamic_namespace_helper_rebinding(self, setup, mutation):
        code = setup + f"""def helper(item):
    return item
def replacement(item):
    return f"changed:{{item}}"
{mutation}
answers = []
for item in items:
    answers.append(llm_query(helper(item)))
"""

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_runtime_guard_preserves_query_parsing_and_aggregation(self):
        code = """import json
chunk_results = []
totals = {"count": 0}
for start in range(0, len(items), 2):
    chunk = items[start:start + 2]
    numbered = "|".join(chunk)
    prompt = f"Classify: {numbered}"
    print(f"querying {start}")
    result = llm_query(prompt)
    clean_json = result.strip()
    parsed = json.loads(clean_json)
    chunk_results.append(parsed)
    for category, count in parsed.items():
        totals[category] += count
"""
        batches = []

        def query_batch(prompts):
            batches.append(prompts)
            return [{"value": '{"count": 2}'}, {"value": '{"count": 1}'}]

        compiled, count = RLM._compile_llm_query_loops(code)
        namespace = {
            "items": ["a", "b", "c"],
            "llm_query": lambda prompt: '{"count": 2}' if prompt.endswith("a|b") else '{"count": 1}',
            "__dspy_llm_query_batched": query_batch,
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }
        exec(compiled, namespace)

        assert count == 1
        assert batches == []
        assert namespace["chunk_results"] == [{"count": 2}, {"count": 1}]
        assert namespace["totals"] == {"count": 3}

    def test_runtime_receiver_guard_handles_retained_json_monkeypatch(self):
        import json

        code = """import json
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{len(state)}"))
    parsed = json.loads('"ok"')
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        original_loads = json.loads
        state, calls, batches = [], [], []

        def retained_loads(value):
            state.append("loads")
            return "ok"

        namespace = {
            "items": ["a", "b"],
            "state": state,
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        try:
            json.loads = retained_loads
            exec(compiled, namespace)
        finally:
            json.loads = original_loads

        assert count == 1
        assert calls == ["a|0", "b|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:a|0", "answer:b|1"]
        assert state == ["loads", "loads"]
        assert namespace["parsed"] == "ok"
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_runtime_receiver_guard_handles_retained_json_prompt_monkeypatch(self):
        import json

        code = """import json
answers = []
for item in items:
    prompt = json.dumps(item)
    answer = llm_query(prompt)
    answers.append(answer)
    state.append(answer)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        original_dumps = json.dumps
        state, calls, batches = [], [], []

        def retained_dumps(value):
            return f"{value}|{len(state)}"

        namespace = {
            "items": ["a", "b"],
            "state": state,
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        try:
            json.dumps = retained_dumps
            exec(compiled, namespace)
        finally:
            json.dumps = original_dumps

        assert count == 1
        assert calls == ["a|0", "b|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:a|0", "answer:b|1"]
        assert state == namespace["answers"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_compiles_multiple_independent_queries_per_iteration(self):
        code = """answers = []
for item in items:
    subject_prompt = f"Subject: {item}"
    subject = llm_query(subject_prompt)
    mood_prompt = f"Mood: {item}"
    mood = llm_query(mood_prompt)
    answers.append((subject, mood))
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a", "b"])

        assert count == 1
        assert batches == [["Subject: a", "Mood: a", "Subject: b", "Mood: b"]]
        assert namespace["answers"] == [
            ("answer:Subject: a", "answer:Mood: a"),
            ("answer:Subject: b", "answer:Mood: b"),
        ]

    def test_compiles_loop_inside_helper_definition(self):
        code = """def classify(values):
    results = []
    for value in values:
        prompt = f"Classify: {value}"
        results.append(llm_query(prompt))
    return results

answers = classify(items)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a", "b"])

        assert count == 1
        assert batches == [["Classify: a", "Classify: b"]]
        assert namespace["answers"] == ["answer:Classify: a", "answer:Classify: b"]

    def test_compiles_empty_chunk_guard_and_postquery_continue(self):
        code = """answers = []
for index in range(len(items) + 1):
    chunk = items[index:index + 1]
    if len(chunk) == 0:
        break
    prompt = chr(10).join(chunk)
    result = llm_query(prompt)
    if "skip" in result:
        continue
    answers.append(result)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a", "skip", "b"])

        assert count == 1
        assert batches == [["a", "skip", "b"]]
        assert namespace["answers"] == ["answer:a", "answer:b"]
        assert namespace["index"] == 3
        assert namespace["chunk"] == []
        assert namespace["prompt"] == "b"
        assert namespace["result"] == "answer:b"

    def test_replays_ordered_lm_failure_and_cleans_temporaries(self):
        import dspy

        calls = []

        class FailingLM:
            def __call__(self, prompt):
                calls.append(prompt)
                if prompt == "p:b":
                    raise dspy.LMTransportError("LM failed")
                return [f"R:{prompt}"]

        code = """import dspy
answers = []
result = "before"
try:
    for item in items:
        prompt = f"p:{item}"
        result = llm_query(prompt)
        answers.append(result)
except dspy.LMTransportError as error:
    caught = str(error)
"""
        rlm = RLM("items -> answer", max_llm_calls=10, sub_lm=FailingLM())
        tools = rlm._make_llm_tools()
        compiled, count = rlm._compile_llm_query_loops(code)
        namespace = {"items": ["a", "b", "c"], **tools}
        exec(compiled, namespace)

        assert count == 1
        assert set(calls) == {"p:a", "p:b", "p:c"}  # Later independent calls may already be in flight.
        assert namespace["answers"] == ["R:p:a"]
        assert namespace["item"] == "b"
        assert namespace["prompt"] == "p:b"
        assert namespace["result"] == "R:p:a"
        assert namespace["caught"] == "LM failed"
        assert not any(name.startswith("__dspy_") and name not in tools for name in namespace)

    def test_cleans_temporaries_when_replay_postprocessing_fails(self):
        calls = []

        class ParsingLM:
            def __call__(self, prompt):
                calls.append(prompt)
                return [{"a": "1", "b": "bad", "c": "3"}[prompt]]

        code = """answers = []
raw = "before"
try:
    for item in items:
        raw = llm_query(item)
        answers.append(int(raw))
except ValueError as error:
    caught = str(error)
"""
        rlm = RLM("items -> answer", max_llm_calls=10, sub_lm=ParsingLM())
        tools = rlm._make_llm_tools()
        compiled, count = rlm._compile_llm_query_loops(code)
        namespace = {"items": ["a", "b", "c"], **tools}
        exec(compiled, namespace)

        assert count == 1
        assert set(calls) == {"a", "b", "c"}
        assert namespace["answers"] == [1]
        assert namespace["item"] == "b"
        assert namespace["raw"] == "bad"
        assert "invalid literal" in namespace["caught"]
        assert not any(name.startswith("__dspy_") and name not in tools for name in namespace)

    def test_lm_failure_does_not_replay_later_prompt_assignments(self):
        import dspy

        calls = []

        class FailingLM:
            def __call__(self, prompt):
                calls.append(prompt)
                if prompt == "first:b":
                    raise dspy.LMTransportError("LM failed")
                return [f"R:{prompt}"]

        code = """import dspy
answers = []
first = second = first_prompt = second_prompt = "before"
try:
    for item in items:
        first_prompt = f"first:{item}"
        first = llm_query(first_prompt)
        second_prompt = f"second:{item}"
        second = llm_query(second_prompt)
        answers.append((first, second))
except dspy.LMTransportError as error:
    caught = str(error)
"""
        rlm = RLM("items -> answer", max_llm_calls=10, sub_lm=FailingLM())
        tools = rlm._make_llm_tools()
        compiled, count = rlm._compile_llm_query_loops(code)
        namespace = {"items": ["a", "b", "c"], **tools}
        exec(compiled, namespace)

        assert count == 1
        assert set(calls) == {"first:a", "second:a", "first:b", "second:b", "first:c", "second:c"}
        assert namespace["answers"] == [("R:first:a", "R:second:a")]
        assert (namespace["item"], namespace["first_prompt"], namespace["second_prompt"]) == ("b", "first:b", "second:a")
        assert (namespace["first"], namespace["second"], namespace["caught"]) == ("R:first:a", "R:second:a", "LM failed")

    def test_compiled_batch_preserves_scalar_budget_prefix(self):
        calls = []

        class EchoLM:
            def __call__(self, prompt):
                calls.append(prompt)
                return [f"R:{prompt}"]

        code = """answers = []
try:
    for item in ["a", "b", "c"]:
        result = llm_query(item)
        answers.append(result)
except Exception as error:
    caught = str(error)
try:
    after = llm_query("after")
except Exception as error:
    after_error = str(error)
"""
        rlm = RLM("items -> answer", max_llm_calls=2, sub_lm=EchoLM())
        tools = rlm._make_llm_tools()
        compiled, count = rlm._compile_llm_query_loops(code)
        namespace = dict(tools)
        exec(compiled, namespace)

        assert count == 1
        assert calls == ["a", "b"]
        assert namespace["answers"] == ["R:a", "R:b"]
        assert namespace["item"] == "c"
        assert "2 + 1 > 2" in namespace["caught"]
        assert "2 + 1 > 2" in namespace["after_error"]

    def test_replays_gather_exception_after_successful_scalar_prefix(self):
        code = """answers = []
try:
    for index in [0, 1]:
        first = llm_query(f"first:{index}")
        prompt = items[index]
        second = llm_query(prompt)
        answers.append((first, second))
except IndexError as error:
    caught = str(error)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a"])

        assert count == 1
        assert batches == [["first:0", "a", "first:1"]]
        assert namespace["answers"] == [("answer:first:0", "answer:a")]
        assert (namespace["index"], namespace["first"], namespace["prompt"], namespace["second"]) == (1, "answer:first:1", "a", "answer:a")
        assert namespace["caught"] == "list index out of range"

    def test_replays_prefix_and_restores_locals_before_iterator_failure(self):
        code = """answers = []
try:
    for item in items:
        if item == "skip":
            continue
        answers.append(llm_query(item))
except RuntimeError as error:
    caught = str(error)
"""

        def failing_items():
            yield "a"
            yield "skip"
            raise RuntimeError("iteration failed")

        namespace, batches, _, count = self.execute_compiled(code, failing_items())

        assert count == 1
        assert batches == [["a"]]
        assert namespace["answers"] == ["answer:a"]
        assert (namespace["item"], namespace["caught"]) == ("skip", "iteration failed")

    def test_stages_prequery_print_conversion_failure_before_batch(self, capsys):
        code = """answers = []
try:
    for item in items:
        print(int(item))
        answers.append(llm_query(item))
except ValueError as error:
    caught = str(error)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["1", "bad"])

        assert count == 1
        assert batches == [["1"]]
        assert namespace["answers"] == ["answer:1"]
        assert namespace["item"] == "bad"
        assert "invalid literal" in namespace["caught"]
        assert capsys.readouterr().out == "1\n"

    def test_stages_prequery_print_subscript_failure_before_batch(self, capsys):
        code = """answers = []
try:
    for index in range(len(items) + 1):
        print(items[index])
        answers.append(llm_query(str(index)))
except IndexError as error:
    caught = str(error)
"""
        namespace, batches, _, count = self.execute_compiled(code, ["a"])

        assert count == 1
        assert batches == [["0"]]
        assert namespace["answers"] == ["answer:0"]
        assert namespace["index"] == 1
        assert namespace["caught"] == "list index out of range"
        assert capsys.readouterr().out == "a\n"

    def test_runtime_alias_guard_handles_retained_plain_list_aliases(self):
        code = """answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        summary = []
        batches = []
        namespace = {
            "items": ["a", "b"],
            "summary": summary,
            "alias": summary,
            "llm_query": lambda prompt: f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert batches == []
        assert namespace["answers"] == ["answer:a|", "answer:b|answer:a|"]
        assert summary == namespace["answers"]

    def test_runtime_callback_guard_handles_retained_callback(self):
        code = """import re
answers = []
for item in items:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    re.sub("1", callback, "1")
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        state, calls, batches = [], [], []

        def callback(match):
            state.append(match.group())
            return match.group()

        namespace = {
            "items": ["a", "b"],
            "state": state,
            "callback": callback,
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert calls == ["a|0", "b|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:a|0", "answer:b|1"]
        assert state == ["1", "1"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_runtime_callback_guard_handles_retained_container_callback(self):
        code = """import re
answers = []
for item in ["a", "b"]:
    prompt = f"{item}|{len(state)}"
    answers.append(llm_query(prompt))
    re.sub("1", callbacks[0], "1")
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        state, calls, batches = [], [], []

        def callback(match):
            state.append(match.group())
            return match.group()

        namespace = {
            "state": state,
            "callbacks": [callback],
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert calls == ["a|0", "b|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:a|0", "answer:b|1"]
        assert state == ["1", "1"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    def test_runtime_callback_guard_handles_retained_container_callee(self):
        code = """answers = []
for item in ["a", "b"]:
    prompt = f"{item}|{len(state)}"
    answers.append(llm_query(prompt))
    callbacks[0]()
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        state, calls, batches = [], [], []

        def callback():
            state.append("x")

        namespace = {
            "state": state,
            "callbacks": [callback],
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert calls == ["a|0", "b|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:a|0", "answer:b|1"]
        assert state == ["x", "x"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    @pytest.mark.parametrize(
        "continuation",
        [
            "recorder.record()",
            '{"recorder": recorder}["recorder"].record()',
            "marker = recorder.value",
            'marker = {"recorder": recorder}["recorder"].value',
            "marker = -recorder",
            "marker = recorder + 1",
            "marker = recorder == 1",
            "marker = bool(recorder)",
            "marker = len(recorder)",
            "marker = {recorder: 1}",
            "if recorder:\n        marker = 1",
            "recorder += 1",
            "scratch = [recorder for recorder in []]\n    marker = recorder + 1",
            "recorder = -recorder",
            "marker = (recorder := -recorder)",
        ],
    )
    def test_runtime_receiver_guard_handles_retained_custom_access(self, continuation):
        code = """answers = []
for item in items:
    answers.append(llm_query(f"{item}|{len(state)}"))
    CONTINUATION
""".replace("CONTINUATION", continuation)
        compiled, count = RLM._compile_llm_query_loops(code)
        state, calls, batches = [], [], []

        class Recorder:
            def record(self):
                state.append("x")

            @property
            def value(self):
                state.append("x")
                return len(state)

            def __neg__(self):
                state.append("x")
                return self

            def __add__(self, other):
                return -self

            __eq__ = __add__

            def __iadd__(self, other):
                self.__neg__()
                return self

            def __bool__(self):
                state.append("x")
                return True

            def __hash__(self):
                state.append("x")
                return len(state)

            def __len__(self):
                state.append("x")
                return len(state)

        namespace = {
            "items": ["x", "y"],
            "state": state,
            "recorder": Recorder(),
            "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert calls == ["x|0", "y|1"]
        assert batches == []
        assert namespace["answers"] == ["answer:x|0", "answer:y|1"]
        assert state == ["x", "x"]
        assert not any(name.startswith("__dspy_") and name not in {"__dspy_llm_query_batched", "__dspy_replay_llm_query"} for name in namespace)

    @pytest.mark.parametrize(
        "definition",
        [
            "def marker(value=holder.value):\n        pass",
            "def marker(*, value=holder.value):\n        pass",
            "def marker(value: holder.value) -> holder.value:\n        pass",
            "def marker(*args: holder.value, **kwargs: holder.value):\n        pass",
            "async def marker(value: holder.value):\n        pass",
            "fn = lambda value=holder.value: value",
            "fn = lambda *, value=holder.value: value",
        ],
    )
    def test_leaves_eager_function_definition_expressions_unchanged(self, definition):
        code = """answers = []
for item in ["a", "b"]:
    prompt = f"{item}|{len(state)}"
    answers.append(llm_query(prompt))
    DEFINITION
""".replace("DEFINITION", definition)

        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize(
        "code",
        [
            """answers = []
for item in ["a", "b"]:
    answers.append(llm_query(item))
    with manager:
        pass
""",
            """async def classify():
    answers = []
    for item in ["a", "b"]:
        answers.append(llm_query(item))
        async with manager:
            pass
""",
        ],
    )
    def test_leaves_context_managers_unchanged(self, code):
        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_runtime_alias_guard_handles_imported_list_aliases(self):
        import sys

        code = """import sys
from sys import path as source
sink = sys.path
answers = []
for item in items:
    prompt = f"{item}|{len(source)}"
    answer = llm_query(prompt)
    answers.append(answer)
    sink.append(answer)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        original_path = list(sys.path)
        calls = []
        batches = []

        try:
            exec(compiled, {
                "items": ["a", "b"],
                "llm_query": lambda prompt: calls.append(prompt) or f"answer:{prompt}",
                "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
                "__dspy_replay_llm_query": lambda outcome: outcome["value"],
            })
        finally:
            sys.path[:] = original_path

        assert count == 1
        assert batches == []
        assert calls == [f"a|{len(original_path)}", f"b|{len(original_path) + 1}"]

    def test_runtime_alias_guard_handles_retained_nested_aliases(self):
        code = """answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(box['alias'])}")
    answers.append(answer)
    summary.append(answer)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        summary = []
        batches = []
        namespace = {
            "items": ["a", "b"],
            "summary": summary,
            "box": {"alias": summary},
            "llm_query": lambda prompt: f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert batches == []
        assert namespace["answers"] == ["answer:a|", "answer:b|answer:a|"]
        assert summary == namespace["answers"]

    def test_runtime_alias_guard_handles_local_for_binding_aliases(self):
        code = """summary = []
choices = [summary]
for alias in choices:
    pass
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        batches = []
        namespace = {
            "items": ["a", "b"],
            "llm_query": lambda prompt: f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert batches == []
        assert namespace["answers"] == ["answer:a|", "answer:b|answer:a|"]
        assert namespace["summary"] == namespace["answers"]

    def test_runtime_alias_guard_does_not_read_same_named_retained_global(self):
        code = """def classify(items):
    answers = []
    for item in items:
        answer = llm_query(item)
        answers.append(answer)
    return answers
result = classify(items)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        batches = []
        namespace = {
            "items": ["a", "b"],
            "answer": "retained",
            "llm_query": lambda prompt: f"answer:{prompt}",
            "__dspy_llm_query_batched": lambda prompts: batches.append(prompts) or [{"value": f"answer:{prompt}"} for prompt in prompts],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert batches == [["a", "b"]]
        assert namespace["result"] == ["answer:a", "answer:b"]
        assert namespace["answer"] == "retained"

    def test_runtime_alias_guard_does_not_read_unbound_mutable_local(self):
        code = """def classify(items):
    answers = []
    for item in items:
        raw = llm_query(item)
        parsed = []
        parsed.append(raw)
        answers.append(parsed)
    return answers
result = classify(items)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        namespace = {
            "items": ["a", "b"],
            "parsed": [],
            "__dspy_llm_query_batched": lambda prompts: [{"value": f"answer:{prompt}"} for prompt in prompts],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert namespace["result"] == [["answer:a"], ["answer:b"]]
        assert namespace["parsed"] == []

    def test_runtime_alias_guard_replays_prefix_before_unbound_augassign(self):
        code = """def classify(items):
    answers = []
    try:
        for item in items:
            answer = llm_query(item)
            answers.append(answer)
            total += 1
    except UnboundLocalError:
        return answers
result = classify(items)
"""
        compiled, count = RLM._compile_llm_query_loops(code)
        calls = []

        def query(prompt):
            calls.append(prompt)
            return f"answer:{prompt}"

        namespace = {
            "items": ["a", "b"],
            "total": 0,
            "__dspy_llm_query_batched": lambda prompts: [{"value": query(prompt)} for prompt in prompts],
            "__dspy_replay_llm_query": lambda outcome: outcome["value"],
        }

        exec(compiled, namespace)

        assert count == 1
        assert calls == ["a", "b"]  # Later calls may be in flight before the replayed continuation fails.
        assert namespace["result"] == ["answer:a"]
        assert namespace["total"] == 0

    @pytest.mark.parametrize(
        "code",
        [
            """answers = []
for item in items:
    first = llm_query(f"First: {item}")
    second = llm_query(f"Second: {first}")
    answers.append((first, second))
""",
            """answers = []
summary = ""
for item in items:
    answer = llm_query(f"Prior: {summary}; item: {item}")
    answers.append(answer)
    summary += answer
""",
        ],
    )
    def test_rejects_query_and_cross_iteration_dependencies(self, code):
        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    @pytest.mark.parametrize(
        "code",
        [
            """answers = []
answer = ""
for item in items:
    prompt = answer + item
    answer = llm_query(prompt)
    answers.append(answer)
""",
            """answers = []
for item in items:
    prompt = make_prompt(item)
    answers.append(llm_query(prompt))
""",
            """answers = []
shared_lines = []
for batch in items:
    lines = shared_lines
    for item in batch:
        lines.append(item)
    answers.append(llm_query("\\n".join(lines)))
""",
            """answers = []
for batch in items:
    lines = []
    for item in batch:
        lines.append(item)
    if len(lines) == 0:
        continue
    answers.append(llm_query("\\n".join(lines)))
""",
            """summary = []
alias = summary
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
""",
            """summary = []
(alias,) = (summary,)
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
""",
            """summary = []
alias: list = summary
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
""",
            """summary = []
alias = summary or []
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(alias)}")
    answers.append(answer)
    summary.append(answer)
""",
            """summary = []
box = {"alias": summary}
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box['alias'])}"))
    summary.append(answers[-1])
""",
            """summary = []
box = dict(alias=summary)
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box['alias'])}"))
    summary.append(answers[-1])
""",
            """summary = []
box = list([summary])
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box[0])}"))
    summary.append(answers[-1])
""",
            """summary = []
box = dict({"alias": summary})
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box['alias'])}"))
    summary.append(answers[-1])
""",
            """summary = []
original = {"alias": summary}
box = original.copy()
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box['alias'])}"))
    summary.append(answers[-1])
""",
            """summary = []
def identity(value):
    return value
alias = identity(summary)
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(alias)}"))
    summary.append(answers[-1])
""",
            """summary = []
for alias in [summary]:
    pass
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(alias)}"))
    summary.append(answers[-1])
""",
            """summary = []
for first_alias in [summary]:
    pass
for alias in [first_alias]:
    pass
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(alias)}"))
    summary.append(answers[-1])
""",
            """history = []
answers = []
for item in items:
    lines = []
    for prior in history:
        lines.append(prior)
    lines.append(item)
    answers.append(llm_query("|".join(lines)))
    history.append(answers[-1])
""",
            """answers = []
x = ["seed"]
for item in ["a", "b"]:
    values = [x for x in x]
    prompt = f"{item}|{values}"
    answer = llm_query(prompt)
    answers.append(answer)
    x.append(answer)
""",
            """answers = []
x = ["seed"]
for item in ["a", "b"]:
    values = list(x for x in x)
    prompt = f"{item}|{values}"
    answer = llm_query(prompt)
    answers.append(answer)
    x.append(answer)
""",
            """summary = []
box = {"alias": summary}
wrapper = {"box": box}
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(wrapper['box']['alias'])}"))
    summary.append(answers[-1])
""",
            """box = {}
summary = box.setdefault("alias", [])
answers = []
for item in items:
    answers.append(llm_query(f"{item}|{'/'.join(box['alias'])}"))
    summary.append(answers[-1])
""",
            """class Box:
    pass
summary = []
box = Box()
box.alias = summary
answers = []
for item in items:
    answer = llm_query(f"{item}|{'/'.join(box.alias)}")
    answers.append(answer)
    summary.append(answer)
""",
            """answers = []
for item in items:
    answers.append(llm_query(str(item)))
    if item < 3:
        items.append(item + 1)
""",
            """state = [1]
def generate():
    for value in state:
        yield value
answers = []
for item in generate():
    answer = llm_query(str(item))
    answers.append(answer)
    if item < 3:
        state.append(item + 1)
""",
            """answers = []
class Item:
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return f"{self.value}:{len(answers)}"
items = [Item("a"), Item("b")]
for item in items:
    answers.append(llm_query(str(item)))
""",
            """answers = []
Item = type("Item", (), {"__str__": lambda self: str(len(answers))})
items = [Item(), Item()]
for item in items:
    answers.append(llm_query(str(item)))
""",
            """def make_prompt(item):
    return f"Prompt: {item}"
def classify(make_prompt, items):
    answers = []
    for item in items:
        answers.append(llm_query(make_prompt(item)))
    return answers
""",
            """import json
from helpers import json
answers = []
for item in items:
    answers.append(llm_query(json.dumps(item)))
""",
            """import json
import helpers as json
answers = []
for item in items:
    answers.append(llm_query(json.dumps(item)))
""",
            """answers = []
for item in items:
    answers.append(llm_query(builder.get(item)))
""",
            """shared = []
answers = []
for item in items:
    holder = [shared]
    holder[0].append(item)
    answers.append(llm_query("|".join(holder[0])))
""",
            """source = iter(["x"])
answers = []
for item in items:
    lines = []
    lines.extend(source)
    answers.append(llm_query("|".join(lines)))
""",
            """source = iter(["x"])
answers = []
for item in items:
    lines = []
    for value in source:
        lines.append(value)
    answers.append(llm_query(f"{item}|{''.join(lines)}"))
""",
            """state = []
def make_prompt(item, state=state):
    return f"{item}|{len(state)}"
answers = []
for item in items:
    answers.append(llm_query(make_prompt(item)))
    state.append(answers[-1])
""",
            """phase = 0
def key(value):
    return ord(value) if phase == 0 else -ord(value)
answers = []
for item in items:
    prompt = "".join(sorted(item, key=key))
    answer = llm_query(prompt)
    answers.append(answer)
    phase += 1
""",
            """import json
phase = 0
def parse(value):
    return int(value) + phase
answers = []
for item in items:
    prompt = str(json.loads(item, parse_int=parse))
    answer = llm_query(prompt)
    answers.append(answer)
    phase += 1
""",
            """import json
state = []
def parse(value):
    state.append(value)
    return int(value)
answers = []
for item in items:
    prompt = f"{item}|{len(state)}"
    answers.append(llm_query(prompt))
    json.loads("1", parse_int=parse)
""",
            """state = []
def key(value):
    state.append(value)
    return value
values = ["b", "a"]
answers = []
for item in ["x", "y"]:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    values.sort(key=key)
""",
            """state = []
key = lambda value: (state.append(value), value)[1]
values = ["b", "a"]
answers = []
for item in ["x", "y"]:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    values.sort(key=key)
""",
            """state = []
def key(value):
    state.append(value)
    return value
def classify(items, callback):
    values = ["b", "a"]
    answers = []
    for item in items:
        answers.append(llm_query(f"{item}|{len(state)}"))
        values.sort(key=callback)
    return answers
result = classify(items, key)
""",
            """state = []
def make_prompt(item):
    def unused():
        state = []
    return f"{item}|{len(state)}"
answers = []
for item in items:
    answer = llm_query(make_prompt(item))
    answers.append(answer)
    state.append(answer)
""",
            """state = []
def record(value, target):
    target.append(value)
answers = []
for item in items:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    record(answer, target=state)
""",
            """def stop(*, target):
    target.clear()
answers = []
for item in items:
    answers.append(llm_query(item))
    stop(target=items)
""",
            """state = []
def record(value, target):
    target.append(value)
answers = []
for item in items:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    record(answer, *[state])
""",
            """state = []
def record(value, state=state):
    state.append(value)
answers = []
for item in items:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    record(answer)
""",
            """state = []
def make_prompt(item):
    return str(item)
def make_prompt(item):
    return f"{item}|{len(state)}"
answers = []
for item in items:
    answer = llm_query(make_prompt(item))
    answers.append(answer)
    state.append(answer)
""",
            """events = []
def print(value):
    events.append(value)
answers = []
for item in items:
    print(item)
    answers.append(llm_query(f"{item}|{len(events)}"))
""",
            """source = iter(["x"])
answers = []
for item in items:
    lines = []
    lines += source
    answers.append(llm_query("|".join(lines)))
""",
            """answers = []
for item in items:
    answer = (record_side_effect(), llm_query(item))
    answers.append(answer)
""",
            """answers = []
for item in items:
    item.append(llm_query(str(len(items[0]))))
""",
            """answers = []
for item in items:
    answer = llm_query(str(len(items[0])))
    answers.append(answer)
    item.append(answer)
""",
            """try:
    for item in items:
        total += llm_query(item)
except NameError:
    pass
""",
            """slots = []
for item in items:
    slots[1] = llm_query(item)
""",
            """for item in items:
    answer: Missing = llm_query(item)
""",
            """answers = {}
for item in items:
    answers.append(llm_query(item))
""",
            """answers = []
for item in items:
    answers.insert(llm_query(item))
""",
            """answers = []
answers = None
for item in items:
    answers.append(llm_query(item))
""",
            """answers = []
for item in items:
    answers.append(llm_query("fixed"))
    item = "changed"
""",
            """answers = []
for item in items:
    lines = []
    for item in ["x", "y"]:
        lines.append(item)
    answers.append(llm_query("|".join(lines)))
""",
            """answers = []
for divisor in items:
    answers.append((10 // divisor, llm_query(str(divisor))))
""",
            """answers = []
for item in items:
    print(item, end=1)
    answers.append(llm_query(item))
""",
            """answers = []
for item in items:
    print(f"{enumerate(items)}")
    answers.append(llm_query(item))
""",
            """source = []
answers = []
for item in ["a", "b"]:
    answer = llm_query(f"{item}|{len(source)}")
    answers.append(answer)
    from sys import path as source
""",
            """source = []
answers = []
for item in ["a", "b"]:
    answers.append(llm_query(f"{item}|{len(source)}"))
    def source():
        pass
""",
            """source = []
answers = []
for item in ["a", "b"]:
    answers.append(llm_query(f"{item}|{len(source)}"))
    async def source():
        pass
""",
            """state = []
def record(function):
    state.append(len(state))
    return function
answers = []
for item in ["a", "b"]:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    @record
    def decorated():
        pass
""",
            """state = []
def record(function):
    state.append(len(state))
    return function
answers = []
for item in ["a", "b"]:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    @record
    async def decorated():
        pass
""",
            """source = []
answers = []
for item in ["a", "b"]:
    prompt = f"{item}|{len(source)}"
    answer = llm_query(prompt)
    answers.append(answer)
    try:
        1 / 0
    except Exception as source:
        pass
""",
            """state = []
def mutate_global():
    state.append("changed")
answers = []
for item in items:
    answer = llm_query(f"{item}|{len(state)}")
    answers.append(answer)
    mutate_global()
""",
            """answers = []
for item in items:
    if item:
        answers.append(llm_query(item))
""",
            """try:
    answers = llm_query_batched(items)
except Exception:
    answers = [llm_query(item) for item in items]
""",
            """from helpers import llm_query
answers = [llm_query(item) for item in items]
""",
            """from helpers import *
answers = [llm_query(item) for item in items]
""",
            """exec("marker = 1")
answers = [llm_query(item) for item in items]
""",
            """run = exec
run("marker = 1")
answers = [llm_query(item) for item in items]
""",
            """globals()["llm_query"] = lambda prompt: "local:" + prompt
answers = [llm_query(item) for item in items]
""",
            """__dspy_frames = "keep"
answers = []
for item in items:
    answers.append(llm_query(item))
""",
            """del llm_query
answers = [llm_query(item) for item in items]
""",
            """from helpers import str
answers = [llm_query(str(item)) for item in items]
""",
            """answers = []
for item in items:
    prompt = f"Classify: {item}"
    answer = llm_query(prompt)
    prompt = answer
    answers.append(answer)
""",
        ],
    )
    def test_leaves_uncertain_loops_unchanged(self, code):
        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_reserves_generated_name_prefix(self):
        code = """def classify(__dspy_frames, items):
    answers = []
    for item in items:
        answers.append(llm_query(item))
    return answers, __dspy_frames

result = classify("keep", items)
"""
        compiled, count = RLM._compile_llm_query_loops(code)

        assert count == 0
        assert compiled == code

    def test_executes_compiled_code_but_keeps_original_code_in_trajectory(self):
        code = 'answers = []\nfor item in items:\n    answers.append(llm_query(f"Analyze: {item}"))'
        factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "done"})])
        rlm = RLM("items -> answer", max_iters=1, interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Analyze", "code": code}])

        result = rlm(items=["a", "b"])

        assert "llm_query_batched" in factory.instances[0].call_history[0][0]
        assert result.trajectory[0]["code"] == code


class TestRLMInterpreterLifecycle:
    def test_execution_instructions_are_part_of_action_prompt(self):
        class Factory(MockInterpreterFactory):
            execution_instructions = "Use this runtime."

        signature = RLM("query -> answer", interpreter_factory=Factory()).generate_action.signature
        optimized = signature.with_instructions("Optimized instructions")

        assert "Use this runtime." in signature.instructions
        assert "standard libraries" not in signature.instructions
        assert optimized.instructions == "Optimized instructions"

    def test_python_interpreter_lm_request_bytes(self):
        class StopLMCall(BaseException):
            pass

        class CapturingLM(dspy.BaseLM):
            forward_contract = "typed_lm"

            def __init__(self):
                super().__init__("snapshot-model", temperature=0.0, max_tokens=1000, cache=False)
                self.request = None

            def forward(self, request):
                self.request = request
                raise StopLMCall

        lm = CapturingLM()
        rlm = RLM("query -> answer", interpreter_factory=PythonInterpreter)

        with pytest.raises(StopLMCall), dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            rlm.generate_action(
                variables_info=["query: str"],
                repl_history=REPLHistory(),
                iteration="1/20",
            )

        request = lm.request.model_dump_json(indent=2).encode()
        snapshot = Path(__file__).with_name("snapshots") / "rlm_python_interpreter_lm_request.json"

        assert request == snapshot.read_bytes().removesuffix(b"\n")

    def test_interpreter_remains_available_as_signature_input(self):
        factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "CPython"})])
        rlm = RLM("interpreter -> answer", max_iters=1, interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return input", "code": "SUBMIT(interpreter)"},
        ])

        result = rlm(interpreter="CPython")

        assert result.answer == "CPython"
        assert factory.instances[0].call_history[0][1] == {"interpreter": "CPython"}

    @pytest.mark.asyncio
    async def test_factory_creates_and_shuts_down_one_interpreter_per_call(self):
        factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iters=1, interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        sync_result = rlm(query="sync")
        async_result = await rlm.acall(query="async")

        assert sync_result.answer == "42"
        assert async_result.answer == "42"
        assert len(factory.instances) == 2
        assert factory.instances[0] is not factory.instances[1]
        for interpreter in factory.instances:
            with pytest.raises(CodeInterpreterError, match="shutdown"):
                interpreter.execute("print('closed')")

    @pytest.mark.asyncio
    async def test_caller_owned_interpreter_can_be_reused_across_sequential_calls(self):
        factory = MockInterpreterFactory()
        interpreter = MockInterpreter(
            responses=[
                FinalOutput({"answer": "first"}),
                FinalOutput({"answer": "second"}),
            ]
        )
        rlm = RLM("query -> answer", max_iters=1, interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": "SUBMIT(answer)"},
        ])

        try:
            sync_result = rlm(interpreter, query="sync")
            async_result = await rlm.acall(interpreter, query="async")

            assert sync_result.answer == "first"
            assert async_result.answer == "second"
            assert factory.instances == []
            assert interpreter.execute("print('still open')") == ""
        finally:
            interpreter.shutdown()

    def test_factory_interpreter_is_shutdown_when_prediction_raises(self):
        class RaisingPredictor:
            def __call__(self, **kwargs):
                raise ValueError("unexpected predictor failure")

        factory = MockInterpreterFactory()
        rlm = RLM("query -> answer", max_iters=1, interpreter_factory=factory)
        rlm.generate_action = RaisingPredictor()

        with pytest.raises(ValueError, match="unexpected predictor failure"):
            rlm(query="test")

        assert len(factory.instances) == 1
        with pytest.raises(CodeInterpreterError, match="shutdown"):
            factory.instances[0].execute("print('closed')")


class TestRLMCodeFenceParsing:
    """Tests for robust fenced-code extraction."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Standard python fence
            ("```python\nprint(1)\n```", "print(1)"),
            ("```py\nx = 1\nprint(x)\n```", "x = 1\nprint(x)"),
            # Bare fence (no language tag)
            ("```\nprint('no lang')\n```", "print('no lang')"),
            # No fences at all
            ("not fenced code", "not fenced code"),
            # Text before fence (preamble is skipped)
            ("I'll inspect first.\n```python\nprint('hello')\n```\nThen I will submit.", "print('hello')"),
            # Text after closing fence (ignored)
            ("```python\nprint(1)\n```\nsome trailing text", "print(1)"),
            # Unclosed fence (just return the body)
            ("```python\nprint('oops')", "print('oops')"),
            # Double fences (outer decorative ```)
            ("```\n```python\nprint(1)\n```\n```", "print(1)"),
            ("```\n```\nprint(2)\n```\n```", "print(2)"),
        ],
    )
    def test_strip_code_fences(self, raw, expected):
        assert _strip_code_fences(raw) == expected

    def test_strip_code_fences_rejects_non_python_lang(self):
        with pytest.raises(SyntaxError, match="json"):
            _strip_code_fences('```json\n{"a": 1}\n```')


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

        rlm = RLM(QASig, max_iters=3)
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
        rlm = RLM("query -> answer", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = rlm(mock, query="What is the answer?")
        assert result.answer == "42"


class TestRLMMaxIterationsFallback:
    """Tests for max_iters reached and extract fallback."""

    def test_max_iters_triggers_extract(self):
        """Test that reaching max_iters uses extract fallback."""
        mock = MockInterpreter(responses=[
            "exploring...",
            "still exploring...",
            "more exploring...",
        ])
        rlm = RLM("query -> answer", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore 1", "code": "print('exploring')"},
            {"reasoning": "Explore 2", "code": "print('exploring')"},
            {"reasoning": "Explore 3", "code": "print('exploring')"},
        ])
        # Mock the extract predictor to return a value
        rlm.extract = make_mock_predictor([
            {"answer": "extracted_answer"},
        ])

        result = rlm.forward(mock, query="test")
        assert result.answer == "extracted_answer"
        assert result.final_reasoning == "Extract forced final output"


class TestRLMToolExceptions:
    """Tests for tool exception handling."""

    def test_tool_exception_returns_error_in_output(self):
        """Test that tool exceptions are caught and returned as errors."""
        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        mock = MockInterpreter(responses=[
            CodeExecutionError("RuntimeError: Tool failed!"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iters=5, tools=[failing_tool])
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Call tool", "code": "failing_tool()"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(mock, query="test")
        assert result.answer == "recovered"

    def test_runtime_error_history_uses_stripped_code(self):
        """Runtime execution failures should preserve stripped code in history."""
        mock = MockInterpreter(responses=[
            CodeExecutionError("NameError: name 'x' is not defined"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iters=5)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Will fail", "code": "```python\nprint(x)\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(mock, query="test")
        assert result.answer == "recovered"
        first_step = result.trajectory[0]
        assert first_step["code"] == "print(x)"

    def test_syntax_error_from_execute_is_recoverable(self):
        """SyntaxError from interpreter.execute should be surfaced as an iteration error."""
        mock = MockInterpreter(responses=[
            SyntaxError("invalid syntax"),
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iters=5)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Bad code", "code": "```python\ndef incomplete(\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(mock, query="test")
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error] invalid syntax")

    def test_syntax_error_from_strip_code_fences_is_recoverable(self):
        """SyntaxError raised by _strip_code_fences (e.g. non-Python fence tag) should be recoverable."""
        mock = MockInterpreter(responses=[
            FinalOutput({"answer": "recovered"}),
        ])
        rlm = RLM("query -> answer", max_iters=5)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Wrong language", "code": "```bash\nls -la\n```"},
            {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
        ])

        result = rlm.forward(mock, query="test")
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error]")

    def test_interpreter_failure_propagates(self):
        """Process and protocol failures must not fall through to LM extraction."""
        def fail_generated_code(code, variables):
            if code == "pass":
                return ""
            raise CodeInterpreterError("protocol corrupt")

        mock = MockInterpreter(execute_fn=fail_generated_code)
        rlm = RLM("query -> answer", max_iters=1)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Try code", "code": "print('test')"},
        ])
        rlm.extract = make_mock_predictor([{"answer": "hallucinated"}])

        with pytest.raises(CodeInterpreterError, match="protocol corrupt"):
            rlm.forward(mock, query="test")

    @pytest.mark.asyncio
    async def test_interpreter_failure_propagates_async(self):
        """Async process and protocol failures must not fall through to LM extraction."""
        def fail_generated_code(code, variables):
            if code == "pass":
                return ""
            raise CodeInterpreterError("protocol corrupt")

        mock = MockInterpreter(execute_fn=fail_generated_code)
        rlm = RLM("query -> answer", max_iters=1)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Try code", "code": "print('test')"},
        ])
        rlm.extract = make_mock_predictor([{"answer": "hallucinated"}])

        with pytest.raises(CodeInterpreterError, match="protocol corrupt"):
            await rlm.aforward(mock, query="test")


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
        assert "USE llm_query FREQUENTLY FOR SMALL SEMANTIC EXPLORATIONS" in instructions
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

    def test_basic_execution(self, pooled_interpreter):
        """Test basic code execution."""
        interp = pooled_interpreter
        result = interp.execute("print(1 + 1)")
        assert "2" in result

    def test_variable_injection(self, pooled_interpreter):
        """Test variable injection."""
        interp = pooled_interpreter
        result = interp.execute(
            "print(x + y)",
            variables={"x": 10, "y": 5}
        )
        assert "15" in result

    def test_variable_injection_with_none_values(self, pooled_interpreter):
        """Test variable injection with None values in dicts/lists (JSON null -> Python None)."""
        interp = pooled_interpreter
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

    def test_tool_call_kwargs(self, configure_pooled_interpreter):
        """Test tool call with keyword arguments."""
        def echo(message: str = "") -> str:
            return f"Echo: {message}"

        interp = configure_pooled_interpreter(tools={"echo": echo})
        result = interp.execute('print(echo(message="hello"))')
        assert "Echo: hello" in result

    def test_tool_call_positional(self, configure_pooled_interpreter):
        """Test tool call with positional arguments."""
        def greet(name: str) -> str:
            return f"Hello: {name}"

        interp = configure_pooled_interpreter(tools={"greet": greet})
        result = interp.execute('print(greet("world"))')
        assert "Hello: world" in result

    def test_multiple_tools(self, configure_pooled_interpreter):
        """Test multiple tools."""
        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        def multiply(a: int = 0, b: int = 0) -> str:
            return str(a * b)

        interp = configure_pooled_interpreter(tools={"add": add, "multiply": multiply})
        result = interp.execute("""
sum_result = add(a=3, b=4)
prod_result = multiply(a=3, b=4)
print(f"Sum: {sum_result}, Product: {prod_result}")
""")
        assert "Sum: 7" in result
        assert "Product: 12" in result

    def test_tool_returns_list(self, configure_pooled_interpreter):
        """Test tool that returns a list (like llm_query_batched)."""
        def batch_process(items: list | None = None) -> list:
            items = items or []
            return [f"processed_{item}" for item in items]

        interp = configure_pooled_interpreter(tools={"batch_process": batch_process})
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

    def test_tool_returns_dict(self, configure_pooled_interpreter):
        """Test tool that returns a dict."""
        def get_info() -> dict:
            return {"name": "test", "count": 42}

        interp = configure_pooled_interpreter(tools={"get_info": get_info})
        result = interp.execute("""
info = get_info()
print(f"Type: {type(info).__name__}")
print(f"Name: {info['name']}")
print(f"Count: {info['count']}")
""")
        assert "Type: dict" in result
        assert "Name: test" in result
        assert "Count: 42" in result

    def test_state_persists(self, pooled_interpreter):
        """Test that state persists across executions."""
        interp = pooled_interpreter
        interp.execute("x = 10")
        result = interp.execute("print(x + 5)")
        assert "15" in result

    def test_syntax_error(self, pooled_interpreter):
        """Test syntax error handling."""
        interp = pooled_interpreter
        with pytest.raises(SyntaxError):
            interp.execute("def incomplete(")

    def test_runtime_error(self, pooled_interpreter):
        """Test runtime error handling."""
        interp = pooled_interpreter
        with pytest.raises(CodeExecutionError):
            interp.execute("undefined_variable")


@pytest.mark.deno
class TestSandboxSecurity:
    """Integration tests for sandbox security restrictions."""

    def test_no_network_access(self, pooled_interpreter):
        """Test that network access is blocked."""
        interp = pooled_interpreter
        with pytest.raises(CodeInterpreterError) as exc_info:
            interp.execute("""
from pyodide.http import pyfetch
import asyncio
asyncio.get_event_loop().run_until_complete(pyfetch("https://example.com"))
""")
        assert "net access" in str(exc_info.value).lower() or "allow-net" in str(exc_info.value).lower()

    def test_imports_work(self, pooled_interpreter):
        """Test that standard library imports work."""
        interp = pooled_interpreter
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
    async def test_aforward_rejects_undeclared_inputs_before_interpreter_execution(self):
        factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "done"})])
        rlm = RLM("context -> answer", interpreter_factory=factory)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("done")'},
        ])

        with pytest.raises(ValueError, match="Unexpected inputs not declared in the signature: \\['SUBMIT'\\]"):
            await rlm.acall(context="some context", SUBMIT="shadowed value")

        assert factory.instances == []

    @pytest.mark.asyncio
    async def test_aforward_basic(self):
        """Test aforward() returns Prediction with expected output (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
        ])

        result = await rlm.aforward(mock, query="What is the answer?")
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_aforward_int_output_mock(self):
        """Test aforward() returns int when signature expects int (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"count": 42})])
        rlm = RLM("query -> count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return count", "code": "SUBMIT(42)"},
        ])

        result = await rlm.aforward(mock, query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_iteration_mock(self):
        """Test aforward() handles multiple iterations before SUBMIT (MockInterpreter)."""
        mock = MockInterpreter(responses=[
            "explored data",
            FinalOutput({"answer": "done"}),
        ])
        rlm = RLM("query -> answer", max_iters=5)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Explore first", "code": "print('exploring')"},
            {"reasoning": "Now finish", "code": 'SUBMIT("done")'},
        ])

        result = await rlm.aforward(mock, query="test")
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
        rlm = RLM(f"query -> {output_field}: {output_type}", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return value", "code": code},
        ])

        result = rlm.forward(mock, query="test")
        assert getattr(result, output_field) == expected

    def test_type_error_retries(self):
        """Test RLM retries when type validation fails (MockInterpreter)."""
        mock = MockInterpreter(responses=[
            FinalOutput({"answer": "maybe"}),  # Invalid for Literal
            FinalOutput({"answer": "yes"}),    # Valid
        ])
        rlm = RLM("query -> answer: Literal['yes', 'no']", max_iters=5)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Try maybe", "code": 'SUBMIT("maybe")'},
            {"reasoning": "Try yes", "code": 'SUBMIT("yes")'},
        ])

        result = rlm.forward(mock, query="is it yes?")
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
    def test_type_coercion(self, output_field, output_type, code, expected, expected_type, pooled_interpreter):
        """Test RLM type coercion for various types with PythonInterpreter."""
        rlm = RLM(f"query -> {output_field}: {output_type}", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return value", "code": code},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
        assert getattr(result, output_field) == expected
        assert isinstance(getattr(result, output_field), expected_type)

    def test_submit_extracts_typed_value(self, pooled_interpreter):
        """Test RLM SUBMIT correctly extracts typed value."""
        rlm = RLM("query -> count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Compute and return", "code": "result = 42\nSUBMIT(result)"},
        ])

        result = rlm.forward(pooled_interpreter, query="count items")
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

    def test_multi_output_final_kwargs(self, pooled_interpreter):
        """SUBMIT(field1=val1, field2=val2) with keyword args."""
        rlm = RLM("query -> name: str, count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return both outputs", "code": 'SUBMIT(name="alice", count=5)'},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
        assert result.name == "alice"
        assert result.count == 5
        assert isinstance(result.count, int)

    def test_multi_output_final_positional(self, pooled_interpreter):
        """SUBMIT(val1, val2) with positional args mapped to field order."""
        rlm = RLM("query -> name: str, count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return both outputs positionally", "code": 'SUBMIT("bob", 10)'},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
        assert result.name == "bob"
        assert result.count == 10

    def test_multi_output_three_fields(self, pooled_interpreter):
        """Signature with 3+ output fields of different types."""
        rlm = RLM("query -> name: str, age: int, active: bool", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return all three", "code": 'SUBMIT(name="carol", age=30, active=True)'},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
        assert result.name == "carol"
        assert result.age == 30
        assert result.active is True

    def test_multi_output_final_missing_field_errors(self, pooled_interpreter):
        """SUBMIT() with missing field should return error in output."""
        rlm = RLM("query -> name: str, count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Missing count field", "code": 'SUBMIT(name="alice")'},
            {"reasoning": "Now provide both", "code": 'SUBMIT(name="alice", count=5)'},
        ])

        # RLM should retry after getting error for missing field
        result = rlm.forward(pooled_interpreter, query="test")
        assert result.name == "alice"
        assert result.count == 5

    def test_multi_output_submit_vars(self, pooled_interpreter):
        """SUBMIT can pass variables directly for multiple outputs."""
        rlm = RLM("query -> name: str, count: int", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Use SUBMIT", "code": 'n = "dave"\nc = 15\nSUBMIT(n, c)'},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
        assert result.name == "dave"
        assert result.count == 15

    def test_multi_output_type_coercion(self, pooled_interpreter):
        """Each output field is coerced to its declared type."""
        rlm = RLM("query -> count: int, ratio: float, flag: bool", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Return mixed types", "code": "SUBMIT(count=42, ratio=3.14, flag=True)"},
        ])

        result = rlm.forward(pooled_interpreter, query="test")
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

    def test_simple_computation_e2e(self, pooled_interpreter):
        """Test full RLM pipeline: DummyLM -> RLM -> PythonInterpreter -> result."""
        with dummy_lm_context([
            {"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"},
        ]):
            rlm = RLM("query -> answer: int", max_iters=3)
            result = rlm.forward(pooled_interpreter, query="What is 2 + 3?")

            assert result.answer == 5
            assert isinstance(result.answer, int)

    def test_multi_turn_computation_e2e(self, pooled_interpreter):
        """Test RLM with multiple turns before SUBMIT."""
        with dummy_lm_context([
            {"reasoning": "First explore the data", "code": "x = 10\nprint(f'x = {x}')"},
            {"reasoning": "Now compute and return", "code": "y = x * 2\nSUBMIT(y)"},
        ]):
            rlm = RLM("query -> answer: int", max_iters=5)
            result = rlm.forward(pooled_interpreter, query="Double ten")

            assert result.answer == 20
            assert len(result.trajectory) == 2

    def test_with_input_variables_e2e(self, pooled_interpreter):
        """Test RLM with input variables passed to sandbox."""
        with dummy_lm_context([
            {"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"},
        ]):
            rlm = RLM("numbers: list[int] -> total: int", max_iters=3)
            result = rlm.forward(pooled_interpreter, numbers=[1, 2, 3, 4, 5])

            assert result.total == 15

    def test_auto_batches_scalar_queries_e2e(self, pooled_interpreter):
        class EchoLM:
            def __call__(self, prompt):
                return [prompt.upper()]

        code = 'answers = [llm_query(f"value:{item}") for item in items]\nSUBMIT("|".join(answers))'
        rlm = RLM("items: list[str] -> answer: str", max_iters=1, sub_lm=EchoLM())
        rlm.generate_action = make_mock_predictor([{"reasoning": "Query every item", "code": code}])

        result = rlm.forward(pooled_interpreter, items=["a", "b"])

        assert result.answer == "VALUE:A|VALUE:B"
        assert result.trajectory[0]["code"] == code

    def test_auto_batches_query_postprocessing_e2e(self, pooled_interpreter):
        class JsonLM:
            def __call__(self, prompt):
                return ['{"count": 0}' if "skip" in prompt else '{"count": 1}']

        code = """import json
total = 0
for index in range(len(items) + 1):
    chunk = items[index:index + 1]
    if len(chunk) == 0:
        break
    item = chunk[0]
    prompt = f"classify:{item}"
    print(f"querying {item}")
    result = llm_query(prompt)
    parsed = json.loads(result)
    if parsed["count"] == 0:
        continue
    total += parsed["count"]
SUBMIT(total)
"""
        rlm = RLM("items: list[str] -> answer: int", max_iters=1, sub_lm=JsonLM())
        rlm.generate_action = make_mock_predictor([{"reasoning": "Query every item", "code": code}])

        result = rlm.forward(pooled_interpreter, items=["a", "skip", "c"])

        assert result.answer == 2
        assert result.trajectory[0]["code"] == code.strip()

    def test_auto_batch_replays_ordered_failure_and_cleans_temporaries_e2e(self, pooled_interpreter):
        calls = []

        class FailingLM:
            def __call__(self, prompt):
                calls.append(prompt)
                if prompt == "p:b":
                    raise dspy.LMTransportError("LM failed")
                return [f"R:{prompt}"]

        code = """answers = []
result = "before"
try:
    for item in items:
        prompt = f"p:{item}"
        result = llm_query(prompt)
        answers.append(result)
except RuntimeError as error:
    caught = str(error)
SUBMIT(answers, item, prompt, result, caught)"""
        compiled, count = RLM._compile_llm_query_loops(code)
        assert count == 1
        assert "__dspy_replay_llm_query" in compiled

        rlm = RLM(
            "items: list[str] -> answers: list[str], item: str, prompt: str, result: str, caught: str",
            max_iters=1,
            sub_lm=FailingLM(),
        )
        rlm.generate_action = make_mock_predictor([{"reasoning": "Query every item", "code": code}])

        output = rlm.forward(pooled_interpreter, items=["a", "b", "c"])

        assert set(calls) == {"p:a", "p:b", "p:c"}
        assert output.answers == ["R:p:a"]
        assert (output.item, output.prompt, output.result) == ("b", "p:b", "R:p:a")
        assert output.caught == "LMTransportError: LM failed"
        assert output.trajectory[0]["code"] == code

    @pytest.mark.parametrize(
        "code",
        [
            # Luna: chunk formatting, query, parse, append.
            """import json
counts = []
for start in range(0, len(items), 1):
    chunk = items[start:start + 1]
    numbered = "\\n".join(f"{offset + 1}. {text}" for offset, text in enumerate(chunk))
    prompt = f"Classify each item:\\n{numbered}"
    raw = llm_query(prompt)
    parsed = json.loads(raw)
    counts.append(parsed["count"])
SUBMIT(sum(counts))""",
            # Haiku: progress output, response cleanup, guarded parsing.
            """import json
total = 0
for index, item in enumerate(items):
    print(f"Query {index + 1}/{len(items)}")
    prompt = "Count this item: {}".format(item)
    raw = llm_query(prompt)
    cleaned = raw.strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        parsed = {"count": 0}
    if parsed["count"] < 0:
        print("invalid")
    total += parsed["count"]
SUBMIT(total)""",
            # DeepSeek: call counter, parse failure continue, nested aggregation.
            """import json
calls = 0
total = 0
for start in range(0, len(items), 1):
    chunk = items[start:start + 1]
    prompt = f"Return a count for: {' | '.join(chunk)}"
    raw = llm_query(prompt)
    calls += 1
    try:
        parsed = json.loads(raw)
    except ValueError:
        continue
    if "count" not in parsed:
        continue
    for key, value in parsed.items():
        total += value
SUBMIT(total)""",
            # Kimi: tuple loop target, JSON fallback, nested filtering.
            """import json
batch_counts = []
for batch_number, batch in enumerate(items):
    prompt = f"Analyze batch {batch_number}: {batch}"
    raw = llm_query(prompt)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"count": 0}
    for key, value in parsed.items():
        if key == "count":
            batch_counts.append(value)
SUBMIT(sum(batch_counts))""",
        ],
        ids=["luna", "haiku", "deepseek", "kimi"],
    )
    def test_auto_batches_observed_model_loop_shapes_e2e(self, pooled_interpreter, code):
        class JsonLM:
            def __call__(self, prompt):
                return ['{"count": 1}']

        compiled, count = RLM._compile_llm_query_loops(code)
        assert count == 1
        assert "llm_query_batched" in compiled

        rlm = RLM("items: list[str] -> answer: int", max_iters=1, sub_lm=JsonLM())
        rlm.generate_action = make_mock_predictor([{"reasoning": "Classify all items", "code": code}])

        result = rlm.forward(pooled_interpreter, items=["a", "b", "c"])

        assert result.answer == 3
        assert result.trajectory[0]["code"] == code

    def test_auto_batch_executes_subqueries_concurrently_e2e(self, pooled_interpreter):
        import threading

        barrier = threading.Barrier(3)

        class BarrierLM:
            def __call__(self, prompt):
                barrier.wait(timeout=3)
                return [prompt.upper()]

        code = """answers = []
for start in range(0, len(items) + 2, 2):
    batch = items[start:start + 2]
    if len(batch) == 0:
        break
    lines = []
    for index, item in enumerate(batch):
        lines.append(f"{index + 1}:{item}")
    prompt = "|".join(lines)
    answer = llm_query(prompt)
    answers.append(answer)
SUBMIT("|".join(answers))"""
        rlm = RLM("items: list[str] -> answer: str", max_iters=1, sub_lm=BarrierLM())
        rlm.generate_action = make_mock_predictor([{"reasoning": "Query concurrently", "code": code}])

        result = rlm.forward(pooled_interpreter, items=["a", "b", "c", "d", "e"])

        assert result.answer == "1:A|2:B|1:C|2:D|1:E"
        assert barrier.n_waiting == 0

    def test_with_tool_e2e(self, pooled_interpreter):
        """Test RLM calling a host-side tool through the sandbox."""
        def lookup(key: str) -> str:
            return {"apple": "red", "banana": "yellow"}.get(key, "unknown")

        with dummy_lm_context([
            {"reasoning": "Look up the color of apple", "code": 'color = lookup(key="apple")\nSUBMIT(color)'},
        ]):
            rlm = RLM("fruit -> color: str", max_iters=3, tools=[lookup])
            result = rlm.forward(pooled_interpreter, fruit="apple")

            assert result.color == "red"

    def test_dspy_tool_execution_semantics_e2e(self, pooled_interpreter):
        import inspect

        from pydantic import BaseModel

        import dspy
        from dspy.utils.callback import BaseCallback

        class Payload(BaseModel):
            value: int

        received = []
        callback_events = []

        async def score(payload: Payload, factor: int = 2):
            received.append((payload, factor))
            return payload.value * factor

        class Recorder(BaseCallback):
            def on_tool_start(self, call_id, instance, inputs):
                callback_events.append(("start", instance))

            def on_tool_end(self, call_id, outputs, exception):
                callback_events.append(("end", outputs, exception))

        tool = Tool(score, name="score_payload")
        rlm = RLM("query -> answer: int", max_iters=1, tools=[tool])
        execution_tool = rlm._prepare_execution_tools()["score_payload"]

        assert execution_tool.__name__ == score.__name__
        assert inspect.signature(execution_tool) == inspect.signature(score)

        with dummy_lm_context([
            {
                "reasoning": "Call the tool",
                "code": 'result = score_payload({"value": 3})\nSUBMIT(result)',
            },
        ]):
            with dspy.context(callbacks=[Recorder()]):
                result = rlm.forward(pooled_interpreter, query="test")

        assert result.answer == 6
        assert len(received) == 1
        assert isinstance(received[0][0], Payload)
        assert received[0][0].value == 3
        assert received[0][1] == 2
        assert callback_events == [("start", tool), ("end", 6, None)]

    @pytest.mark.asyncio
    async def test_aforward_simple_computation_e2e(self):
        """Test aforward() full pipeline: DummyLM -> RLM -> PythonInterpreter -> result."""
        with dummy_lm_context([
            {"reasoning": "I need to compute 2 + 3", "code": "result = 2 + 3\nSUBMIT(result)"},
        ]):
            rlm = RLM("query -> answer: int", max_iters=3)
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
            rlm = RLM("query -> answer: int", max_iters=5)
            result = await rlm.aforward(query="Double ten")

            assert result.answer == 20
            assert len(result.trajectory) == 2

    @pytest.mark.asyncio
    async def test_aforward_with_input_variables_e2e(self):
        """Test aforward() with input variables passed to sandbox."""
        with dummy_lm_context([
            {"reasoning": "Sum the numbers in the list", "code": "SUBMIT(sum(numbers))"},
        ]):
            rlm = RLM("numbers: list[int] -> total: int", max_iters=3)
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

        rlm = RLM("context, query -> answer", max_iters=5)
        result = rlm(
            context={"numbers": [1, 2, 3, 4, 5]},
            query="What is the sum of the numbers?"
        )
        assert "15" in result.answer

    def test_with_llm_query(self):
        """Test RLM using the llm_query tool."""
        import dspy
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        rlm = RLM("context, query -> answer", max_iters=5)
        result = rlm(
            context="The quick brown fox jumps over the lazy dog.",
            query="Use llm_query to describe what animal is mentioned as lazy."
        )
        assert "dog" in result.answer.lower()


# ============================================================================
# Unit Tests: SandboxSerializable integration with RLM
# ============================================================================


class _StubSerializable(SandboxSerializable):
    """Stub serializable used to exercise RLM integration."""

    def __init__(self, data: str = "stub_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return f"StubData({self.data})"


class _BinarySerializable(SandboxSerializable):
    """Serializable that emits non-UTF8 bytes to exercise binary payload path."""

    def sandbox_setup(self) -> str:
        return ""

    def to_sandbox(self) -> bytes:
        return b"\xff\xfe\xfd"

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return "BinaryPayload"


class TestBuildVariablesWithSerializable:
    """Tests for _build_variables with SandboxSerializable inputs."""

    def test_serializable_uses_build_repl_variable(self):
        """SandboxSerializable subclasses route through build_repl_variable."""
        rlm = RLM("data, query -> answer")
        stub = _StubSerializable("my_data")
        variables = rlm._build_variables(data=stub, query="test query")

        data_var = next(v for v in variables if v.name == "data")
        query_var = next(v for v in variables if v.name == "query")

        assert "StubData(my_data)" in data_var.preview
        assert "test query" in query_var.preview

        # sandbox_setup imports should be surfaced in the description.
        assert "import json" in data_var.desc

    def test_regular_values_unchanged(self):
        """Non-SandboxSerializable values should use default REPLVariable creation."""
        rlm = RLM("context -> answer")
        variables = rlm._build_variables(context="plain text")
        assert len(variables) == 1
        assert variables[0].name == "context"
        assert "plain text" in variables[0].preview


class TestPrepareSerializableVars:
    """Tests for _prepare_serializable_vars with MockInterpreter."""

    def test_separates_serializable_from_regular(self):
        """Serializable values are injected; regular values are returned."""
        mock = MockInterpreter(responses=["", FinalOutput({"answer": "42"})])
        rlm = RLM("data, query -> answer", max_iters=3)

        stub = _StubSerializable("payload")

        # Manually call _prepare_serializable_vars
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"data": stub, "query": "hello"}, mock)

        # Regular args should only contain non-serializable values
        assert "query" in regular
        assert regular["query"] == "hello"
        assert "data" not in regular

        # MockInterpreter should have received an execute call for the setup
        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "import json" in code
        assert "_raw_data" in variables

    def test_no_serializable_returns_all(self):
        """When no SandboxSerializable values exist, all args are returned."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM("query -> answer", max_iters=3)

        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        regular = rlm._prepare_serializable_vars({"query": "hello"}, mock)

        assert regular == {"query": "hello"}
        assert mock.call_count == 0

    def test_binary_payload_uses_base64_transport(self):
        """Non-UTF8 bytes should be transported via base64 and decoded in sandbox code."""
        mock = MockInterpreter(responses=[""])
        rlm = RLM("data, query -> answer")

        payload = _BinarySerializable()
        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": payload, "query": "hello"}, mock)

        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        assert "_raw_data = base64.b64decode(_raw_data_base64)" in code
        assert variables["_raw_data_base64"] == base64.b64encode(b"\xff\xfe\xfd").decode("ascii")

    def test_large_payload_not_inlined_in_code(self):
        """Large payloads should ride in the variables kwarg, not the code string.

        Inlining a multi-MB payload into the code text would balloon every
        subsequent prompt and could blow past sandbox limits. The transport
        contract is: code stays small, payload travels as a named variable.
        """
        mock = MockInterpreter(responses=[""])
        rlm = RLM("data, query -> answer")

        large_text = "x" * (2 * 1024 * 1024)  # 2 MB UTF-8 payload

        class _LargeText(SandboxSerializable):
            def sandbox_setup(self) -> str:
                return ""

            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"LargeText({len(large_text)} chars)"

        rlm._inject_execution_context(mock, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": _LargeText(), "query": "hi"}, mock)

        assert mock.call_count == 1
        code, variables = mock.call_history[0]
        # Payload must be in variables, not the code string.
        assert variables["_raw_data"] == large_text
        assert large_text not in code
        assert len(code) < 1000

    def test_forward_with_serializable(self):
        """Full forward() pass with a SandboxSerializable input."""
        mock = MockInterpreter(responses=[
            "",  # setup execution for _prepare_serializable_vars
            FinalOutput({"answer": "done"}),
        ])
        rlm = RLM("data, query -> answer", max_iters=3)
        rlm.generate_action = make_mock_predictor([
            {"reasoning": "Done", "code": 'SUBMIT("done")'},
        ])

        stub = _StubSerializable("test_payload")
        result = rlm.forward(mock, data=stub, query="test")
        assert result.answer == "done"

        # First call should be the serializable setup, second should be the iteration
        assert mock.call_count == 2


@pytest.mark.deno
class TestLargeSerializableRoundTrip:
    """End-to-end test that large SandboxSerializable payloads survive the sandbox."""

    def test_large_payload_round_trips_through_real_sandbox(self, pooled_interpreter):
        """A multi-MB payload should be reconstructable inside the real interpreter."""
        large_text = "abc123" * (200 * 1024)  # ~1.2 MB UTF-8

        class _LargeText(SandboxSerializable):
            def sandbox_setup(self) -> str:
                return ""

            def to_sandbox(self) -> bytes:
                return large_text.encode("utf-8")

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = {data_expr}"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"LargeText({len(large_text)} chars)"

        interp = pooled_interpreter
        rlm = RLM("data -> answer")
        rlm._inject_execution_context(interp, rlm._prepare_execution_tools())
        rlm._prepare_serializable_vars({"data": _LargeText()}, interp)
        result = interp.execute("print(len(data)); print(data[:6])")

        assert str(len(large_text)) in result
        assert "abc123" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
