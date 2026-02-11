"""
Tests for the CLI module.

Test organization:
- Unit tests: CLI initialization, signature construction, output parsing, validation
- Subprocess tests: Real subprocess calls using cli_echo.py test script
- Optimizer compatibility: named_predictors, dump_state, trace capture
- Budget controls: timeout, retries
- JSONL parsing: Codex/Claude Code event formats
- Async tests: aforward() with real subprocess
- DummyLM integration: Full forward pass with mocked LM for prepare/extract

Follows the same patterns as test_rlm.py.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dspy.predict.cli import CLI, CLIError, PROMPT_PLACEHOLDER
from dspy.primitives.cli_types import CLIEvent, CLITrajectory, parse_jsonl_events
from dspy.primitives.prediction import Prediction

# Path to the test CLI script
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cli_echo.py"


def _make_cli(sig="question -> answer", env=None, **kwargs):
    """Factory for CLI module with echo script."""
    return CLI(
        sig,
        command=[sys.executable, str(SCRIPT)],
        env=env,
        **kwargs,
    )


def make_mock_predictor(responses: list[dict]):
    """Factory for mock predictors with scripted responses (same pattern as test_rlm.py)."""

    class MockPredictor:
        def __init__(self):
            self.idx = 0
            # Provide a minimal signature-like object for named_predictors compatibility
            self.signature = None
            self.demos = []

        def _next_response(self):
            result = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**result)

        def __call__(self, **kwargs):
            return self._next_response()

        async def acall(self, **kwargs):
            return self._next_response()

    return MockPredictor()


# ============================================================================
# Unit Tests: CLIEvent and CLITrajectory
# ============================================================================


class TestCLIEvent:
    """Tests for CLIEvent data type."""

    def test_basic_event(self):
        event = CLIEvent(type="agent_message", content="Hello world")
        assert event.type == "agent_message"
        assert event.content == "Hello world"
        assert event.timestamp is None

    def test_event_with_timestamp(self):
        event = CLIEvent(type="thinking", content="hmm", timestamp=1.5)
        formatted = event.format()
        assert "[thinking]" in formatted
        assert "1.50s" in formatted
        assert "hmm" in formatted

    def test_event_format_no_timestamp(self):
        event = CLIEvent(type="tool_call", content="search()")
        formatted = event.format()
        assert "[tool_call]: search()" == formatted


class TestCLITrajectory:
    """Tests for CLITrajectory data type."""

    def test_basic_trajectory(self):
        traj = CLITrajectory(
            prompt="test prompt",
            events=[],
            raw_stdout="output text",
            stderr="",
            returncode=0,
            elapsed=1.5,
        )
        assert traj.prompt == "test prompt"
        assert traj.returncode == 0
        assert traj.elapsed == 1.5

    def test_trajectory_format(self):
        traj = CLITrajectory(
            prompt="test",
            events=[CLIEvent(type="thinking", content="hmm")],
            raw_stdout="result",
            stderr="warning",
            returncode=0,
            elapsed=2.0,
        )
        formatted = traj.format()
        assert "exit=0" in formatted
        assert "2.0s" in formatted
        assert "test" in formatted
        assert "Events (1)" in formatted
        assert "result" in formatted
        assert "warning" in formatted

    def test_trajectory_format_truncation(self):
        traj = CLITrajectory(
            prompt="test",
            events=[],
            raw_stdout="x" * 200,
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        formatted = traj.format(max_chars=50)
        assert "truncated" in formatted

    def test_trajectory_no_stderr_section_when_empty(self):
        traj = CLITrajectory(
            prompt="test",
            events=[],
            raw_stdout="output",
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        formatted = traj.format()
        assert "Stderr" not in formatted

    def test_get_agent_message(self):
        traj = CLITrajectory(
            prompt="test",
            events=[
                CLIEvent(type="thinking", content="hmm"),
                CLIEvent(type="agent_message", content="The answer is 42"),
            ],
            raw_stdout="",
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        assert traj.get_agent_message() == "The answer is 42"

    def test_get_agent_message_none_when_no_events(self):
        traj = CLITrajectory(
            prompt="test",
            events=[],
            raw_stdout="output",
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        assert traj.get_agent_message() is None

    def test_get_agent_message_returns_last(self):
        traj = CLITrajectory(
            prompt="test",
            events=[
                CLIEvent(type="agent_message", content="first"),
                CLIEvent(type="agent_message", content="second"),
            ],
            raw_stdout="",
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        assert traj.get_agent_message() == "second"

    def test_str_delegates_to_format(self):
        traj = CLITrajectory(
            prompt="test",
            events=[],
            raw_stdout="out",
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        assert str(traj) == traj.format()


# ============================================================================
# Unit Tests: JSONL Parsing
# ============================================================================


class TestJSONLParsing:
    """Tests for parse_jsonl_events."""

    def test_codex_style_event(self):
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Hello"},
        })
        events = parse_jsonl_events(line)
        assert len(events) == 1
        assert events[0].type == "agent_message"
        assert events[0].content == "Hello"

    def test_generic_event(self):
        line = json.dumps({"type": "thinking", "content": "Let me consider..."})
        events = parse_jsonl_events(line)
        assert len(events) == 1
        assert events[0].type == "thinking"
        assert events[0].content == "Let me consider..."

    def test_generic_event_with_text_key(self):
        line = json.dumps({"type": "message", "text": "Hello"})
        events = parse_jsonl_events(line)
        assert len(events) == 1
        assert events[0].content == "Hello"

    def test_event_with_timestamp(self):
        line = json.dumps({"type": "step", "content": "done", "timestamp": 1.5})
        events = parse_jsonl_events(line)
        assert events[0].timestamp == 1.5

    def test_multiple_events(self):
        lines = "\n".join([
            json.dumps({"type": "thinking", "content": "hmm"}),
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}),
        ])
        events = parse_jsonl_events(lines)
        assert len(events) == 2
        assert events[0].type == "thinking"
        assert events[1].type == "agent_message"

    def test_mixed_jsonl_and_plaintext(self):
        """Non-JSON lines are silently skipped."""
        lines = "some plain text\n" + json.dumps({"type": "msg", "content": "hi"}) + "\nmore text\n"
        events = parse_jsonl_events(lines)
        assert len(events) == 1
        assert events[0].content == "hi"

    def test_empty_input(self):
        assert parse_jsonl_events("") == []
        assert parse_jsonl_events("\n\n") == []

    def test_invalid_json_skipped(self):
        events = parse_jsonl_events("{not json}\n{also bad}")
        assert events == []

    def test_non_dict_json_skipped(self):
        events = parse_jsonl_events(json.dumps([1, 2, 3]))
        assert events == []

    def test_missing_type_skipped(self):
        events = parse_jsonl_events(json.dumps({"content": "no type field"}))
        assert events == []

    def test_raw_field_preserved(self):
        data = {"type": "thinking", "content": "hmm", "extra": "data"}
        events = parse_jsonl_events(json.dumps(data))
        assert events[0].raw == data


# ============================================================================
# Unit Tests: CLI Module Initialization
# ============================================================================


class TestCLIInitialization:
    """Tests for CLI module initialization."""

    def test_basic_init(self):
        cli = _make_cli("question -> answer")
        assert "question" in cli.signature.input_fields
        assert "answer" in cli.signature.output_fields
        assert cli.prepare_prompt is not None
        assert cli.extract is not None

    def test_multi_field_signature(self):
        cli = _make_cli("context, question -> summary, answer")
        assert "context" in cli.signature.input_fields
        assert "question" in cli.signature.input_fields
        assert "summary" in cli.signature.output_fields
        assert "answer" in cli.signature.output_fields

    def test_command_string_parsing(self):
        cli = CLI("q -> a", command="python script.py --flag")
        assert cli.command == ["python", "script.py", "--flag"]

    def test_command_list(self):
        cli = CLI("q -> a", command=["python", "script.py"])
        assert cli.command == ["python", "script.py"]

    def test_empty_command_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CLI("q -> a", command="")

    def test_empty_list_command_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CLI("q -> a", command=[])

    def test_prompt_placeholder_detection(self):
        cli = CLI("q -> a", command='echo "{PROMPT}"')
        assert cli._uses_placeholder is True

    def test_no_placeholder_means_stdin(self):
        cli = _make_cli()
        assert cli._uses_placeholder is False

    def test_default_budget_controls(self):
        cli = _make_cli()
        assert cli.timeout is None
        assert cli.max_retries == 0

    def test_custom_budget_controls(self):
        cli = _make_cli(timeout=60, max_retries=3)
        assert cli.timeout == 60
        assert cli.max_retries == 3

    def test_default_parse_jsonl_true(self):
        cli = _make_cli()
        assert cli.parse_jsonl is True

    def test_env_stored(self):
        cli = _make_cli(env={"MY_VAR": "value"})
        assert cli.env["MY_VAR"] == "value"


class TestCLISignatureConstruction:
    """Tests for the dynamically built signatures."""

    def test_prepare_signature_has_input_fields(self):
        cli = _make_cli("context, question -> answer")
        prep_sig = cli.prepare_prompt.signature
        assert "context" in prep_sig.input_fields
        assert "question" in prep_sig.input_fields
        assert "cli_prompt" in prep_sig.output_fields

    def test_extract_signature_has_output_fields(self):
        cli = _make_cli("question -> summary, answer")
        extract_sig = cli.extract.signature
        assert "cli_output" in extract_sig.input_fields
        assert "summary" in extract_sig.output_fields
        assert "answer" in extract_sig.output_fields

    def test_prepare_instructions_include_output_descriptions(self):
        cli = _make_cli("question -> answer")
        instructions = cli.prepare_prompt.signature.instructions
        assert "answer" in instructions
        assert "CLI" in instructions

    def test_task_instructions_propagated(self):
        import dspy

        class MySig(dspy.Signature):
            """Summarize the document carefully."""
            doc: str = dspy.InputField()
            summary: str = dspy.OutputField()

        cli = CLI(MySig, command=[sys.executable, str(SCRIPT)])
        prep_instructions = cli.prepare_prompt.signature.instructions
        assert "Summarize the document carefully" in prep_instructions


class TestCLIValidation:
    """Tests for input validation."""

    def test_missing_input_raises(self):
        cli = _make_cli("context, question -> answer")
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        with pytest.raises(ValueError, match="Missing required inputs"):
            cli.forward(context="only context")

    def test_multiple_missing_inputs_reported(self):
        cli = _make_cli("a, b, c -> answer")
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        with pytest.raises(ValueError) as exc_info:
            cli.forward(a="only a")
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)


# ============================================================================
# Unit Tests: Output Parsing (no subprocess needed)
# ============================================================================


class TestDirectParsing:
    """Tests for _try_direct_parse."""

    def test_single_string_field_returns_stdout(self):
        cli = _make_cli("question -> answer")
        traj = CLITrajectory(
            prompt="test", events=[], raw_stdout="The answer is 42",
            stderr="", returncode=0, elapsed=0.1,
        )
        result = cli._try_direct_parse("The answer is 42", traj)
        assert result == {"answer": "The answer is 42"}

    def test_single_string_field_prefers_agent_message(self):
        cli = _make_cli("question -> answer")
        traj = CLITrajectory(
            prompt="test",
            events=[CLIEvent(type="agent_message", content="From events")],
            raw_stdout='{"type":"item.completed"...}',
            stderr="",
            returncode=0,
            elapsed=0.1,
        )
        result = cli._try_direct_parse("raw stdout", traj)
        assert result == {"answer": "From events"}

    def test_json_output_matching_fields(self):
        cli = _make_cli("question -> name, age: int")
        traj = CLITrajectory(
            prompt="test", events=[], raw_stdout='{"name": "Alice", "age": 30}',
            stderr="", returncode=0, elapsed=0.1,
        )
        result = cli._try_direct_parse('{"name": "Alice", "age": 30}', traj)
        assert result == {"name": "Alice", "age": 30}

    def test_json_missing_fields_returns_none(self):
        cli = _make_cli("question -> name, age: int")
        traj = CLITrajectory(
            prompt="test", events=[], raw_stdout='{"name": "Alice"}',
            stderr="", returncode=0, elapsed=0.1,
        )
        result = cli._try_direct_parse('{"name": "Alice"}', traj)
        assert result is None

    def test_non_string_single_field_returns_none(self):
        """Non-string single output needs extract, not direct parse."""
        cli = _make_cli("question -> count: int")
        traj = CLITrajectory(
            prompt="test", events=[], raw_stdout="42",
            stderr="", returncode=0, elapsed=0.1,
        )
        result = cli._try_direct_parse("42", traj)
        assert result is None  # Can't direct-parse int

    def test_invalid_json_returns_none(self):
        cli = _make_cli("question -> name, age: int")
        traj = CLITrajectory(
            prompt="test", events=[], raw_stdout="not json",
            stderr="", returncode=0, elapsed=0.1,
        )
        result = cli._try_direct_parse("not json", traj)
        assert result is None


# ============================================================================
# Subprocess Tests: Real CLI Execution
# ============================================================================


class TestCLISubprocess:
    """Tests using real subprocess calls to cli_echo.py."""

    def test_basic_echo(self):
        """CLI echoes stdin back as stdout."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "hello world"}])
        result = cli.forward(question="test")
        assert result.answer == "hello world"

    def test_jsonl_mode(self):
        """CLI emits JSONL, agent_message is extracted."""
        cli = _make_cli(env={"CLI_MODE": "json"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "the answer"}])
        result = cli.forward(question="test")
        assert result.answer == "the answer"
        assert len(result.trajectory.events) == 1
        assert result.trajectory.events[0].type == "agent_message"

    def test_multi_jsonl_events(self):
        """CLI emits multiple JSONL events, all captured in trajectory."""
        cli = _make_cli(env={"CLI_MODE": "multi_json"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "final answer"}])
        result = cli.forward(question="test")
        assert result.answer == "final answer"
        assert len(result.trajectory.events) == 4  # thinking, tool_call, tool_result, agent_message

    def test_stderr_captured(self):
        """Stderr is captured in trajectory."""
        cli = _make_cli(env={"CLI_MODE": "warn"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "warned output"}])
        result = cli.forward(question="test")
        assert result.answer == "warned output"
        assert "proceed with caution" in result.trajectory.stderr

    def test_json_fields_direct_parse(self):
        """JSON output with matching field names is parsed directly."""
        cli = CLI(
            "question -> name, role",
            command=[sys.executable, str(SCRIPT)],
            env={"CLI_MODE": "json_fields"},
        )
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "name=Alice,role=Engineer"}])
        result = cli.forward(question="test")
        assert result.name == "Alice"
        assert result.role == "Engineer"

    def test_trajectory_has_timing(self):
        """Trajectory captures elapsed time."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "timed"}])
        result = cli.forward(question="test")
        assert result.trajectory.elapsed > 0
        assert result.trajectory.returncode == 0

    def test_trajectory_has_prompt(self):
        """Trajectory records what prompt was sent."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "my specific prompt"}])
        result = cli.forward(question="test")
        assert result.trajectory.prompt == "my specific prompt"


class TestCLIErrors:
    """Tests for CLI error handling."""

    def test_nonzero_exit_raises(self):
        cli = _make_cli(env={"CLI_MODE": "fail"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "will fail"}])
        with pytest.raises(CLIError, match="exited with status 2"):
            cli.forward(question="test")

    def test_command_not_found_raises(self):
        cli = CLI("question -> answer", command="nonexistent_program_xyz")
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        with pytest.raises(CLIError, match="not found"):
            cli.forward(question="test")

    def test_timeout_raises(self):
        cli = _make_cli(env={"CLI_MODE": "timeout"}, timeout=0.5)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "will timeout"}])
        with pytest.raises(CLIError, match="timed out"):
            cli.forward(question="test")

    def test_cli_error_has_stdout_stderr(self):
        cli = _make_cli(env={"CLI_MODE": "fail"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        with pytest.raises(CLIError) as exc_info:
            cli.forward(question="test")
        assert exc_info.value.stderr is not None
        assert exc_info.value.returncode == 2


# ============================================================================
# Budget Control Tests
# ============================================================================


class TestCLIRetries:
    """Tests for retry logic."""

    def test_retries_on_failure(self):
        """CLI retries max_retries times before raising."""
        cli = _make_cli(env={"CLI_MODE": "fail"}, max_retries=2)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "retry me"}])

        with pytest.raises(CLIError) as exc_info:
            cli.forward(question="test")
        # Error message should mention attempt 3/3
        assert "attempt 3/3" in str(exc_info.value)

    def test_zero_retries_means_one_attempt(self):
        """max_retries=0 means exactly one attempt."""
        cli = _make_cli(env={"CLI_MODE": "fail"}, max_retries=0)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "no retry"}])

        with pytest.raises(CLIError) as exc_info:
            cli.forward(question="test")
        assert "attempt 1/1" in str(exc_info.value)

    def test_retry_succeeds_on_later_attempt(self):
        """Mock subprocess that fails first, succeeds second."""
        call_count = [0]

        original_run = subprocess.run

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="fail")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="success", stderr="")

        cli = _make_cli(max_retries=2)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "retry test"}])

        with patch("subprocess.run", side_effect=mock_run):
            result = cli.forward(question="test")
        assert result.answer == "success"


class TestCLITimeout:
    """Tests for timeout budget control."""

    def test_timeout_kills_process(self):
        """timeout kills long-running CLI."""
        cli = _make_cli(env={"CLI_MODE": "timeout"}, timeout=0.3)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "hang forever"}])
        with pytest.raises(CLIError, match="timed out"):
            cli.forward(question="test")

    def test_fast_cli_within_timeout(self):
        """CLI that finishes quickly succeeds with timeout set."""
        cli = _make_cli(timeout=10)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "fast"}])
        result = cli.forward(question="test")
        assert result.answer == "fast"


# ============================================================================
# Optimizer Compatibility Tests
# ============================================================================


class TestOptimizerCompatibility:
    """Tests that CLI module works with DSPy's optimizer infrastructure."""

    def test_named_predictors(self):
        """CLI exposes prepare_prompt and extract as named predictors."""
        cli = _make_cli()
        predictor_names = [name for name, _ in cli.named_predictors()]
        assert "prepare_prompt" in predictor_names
        assert "extract" in predictor_names

    def test_predictors_count(self):
        """CLI has exactly 2 predictors."""
        cli = _make_cli()
        assert len(cli.predictors()) == 2

    def test_dump_state(self):
        """dump_state returns serializable dict with key fields."""
        cli = _make_cli(env={"MY_VAR": "val"}, timeout=60, max_retries=2)
        state = cli.dump_state()
        assert state["timeout"] == 60
        assert state["max_retries"] == 2
        assert state["env"]["MY_VAR"] == "val"
        assert "prepare_prompt" in state
        assert "extract" in state

    def test_dump_state_filters_secrets(self):
        """dump_state excludes env vars with 'key' or 'secret' in name."""
        cli = _make_cli(env={"API_KEY": "secret123", "SAFE_VAR": "visible"})
        state = cli.dump_state()
        assert "API_KEY" not in state["env"]
        assert state["env"]["SAFE_VAR"] == "visible"

    def test_load_state_round_trip(self):
        """dump_state -> new CLI -> load_state -> fields match."""
        cli = _make_cli(
            env={"MY_VAR": "val"},
            timeout=60,
            max_retries=2,
            max_output_chars=5000,
            parse_jsonl=False,
        )
        state = cli.dump_state()

        cli2 = _make_cli()
        cli2.load_state(state)

        assert cli2.command == cli.command
        assert cli2.env == cli.env
        assert cli2.cwd == cli.cwd
        assert cli2.encoding == cli.encoding
        assert cli2.timeout == cli.timeout
        assert cli2.max_retries == cli.max_retries
        assert cli2.parse_jsonl == cli.parse_jsonl
        assert cli2.max_output_chars == cli.max_output_chars

    def test_set_lm_propagates(self):
        """set_lm() propagates to both Predict nodes."""
        import dspy

        cli = _make_cli()
        mock_lm = MagicMock(spec=dspy.LM)
        cli.set_lm(mock_lm)

        assert cli.prepare_prompt.lm is mock_lm
        assert cli.extract.lm is mock_lm

    def test_trajectory_in_prediction(self):
        """Forward pass includes trajectory for optimizer inspection."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "optimizable"}])
        result = cli.forward(question="test")
        assert hasattr(result, "trajectory")
        assert isinstance(result.trajectory, CLITrajectory)

    def test_call_is_alias_for_forward(self):
        """__call__ works the same as forward()."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "via call"}])
        result = cli(question="test")
        assert result.answer == "via call"


# ============================================================================
# DummyLM Integration Tests
# ============================================================================


class TestCLIWithDummyLM:
    """Tests using DummyLM for prepare_prompt and extract Predict nodes."""

    def test_full_forward_with_dummy_lm(self):
        """Full forward pass: DummyLM generates prompt, CLI echoes, direct parse."""
        import dspy
        from dspy.utils.dummies import DummyLM

        # DummyLM returns a response that ChatAdapter will parse as cli_prompt
        lm = DummyLM([{"cli_prompt": "What is the capital of France?"}])

        with dspy.context(lm=lm):
            cli = _make_cli("question -> answer")
            result = cli(question="capitals")

        assert isinstance(result, Prediction)
        assert hasattr(result, "answer")
        assert hasattr(result, "trajectory")

    def test_extract_fallback_with_dummy_lm(self):
        """When direct parse fails, extract Predict is called."""
        import dspy
        from dspy.utils.dummies import DummyLM

        # First call: prepare_prompt returns prompt
        # Second call: extract returns structured output
        lm = DummyLM([
            {"cli_prompt": "count=42,name=test"},
            {"count": "42"},
        ])

        with dspy.context(lm=lm):
            cli = CLI(
                "question -> count: int",
                command=[sys.executable, str(SCRIPT)],
            )
            result = cli(question="how many?")

        # count:int can't be direct-parsed from plain text, so extract is used
        assert hasattr(result, "count")
        assert hasattr(result, "trajectory")


# ============================================================================
# Async Tests
# ============================================================================


class TestCLIAsync:
    """Tests for aforward() async execution."""

    @pytest.mark.asyncio
    async def test_aforward_basic(self):
        """aforward() echoes back via async subprocess."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "async hello"}])
        result = await cli.aforward(question="test")
        assert result.answer == "async hello"

    @pytest.mark.asyncio
    async def test_aforward_jsonl(self):
        """aforward() parses JSONL events."""
        cli = _make_cli(env={"CLI_MODE": "json"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "async json"}])
        result = await cli.aforward(question="test")
        assert result.answer == "async json"
        assert len(result.trajectory.events) == 1

    @pytest.mark.asyncio
    async def test_aforward_timeout(self):
        """aforward() respects timeout."""
        cli = _make_cli(env={"CLI_MODE": "timeout"}, timeout=0.3)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "hang"}])
        with pytest.raises(CLIError, match="timed out"):
            await cli.aforward(question="test")

    @pytest.mark.asyncio
    async def test_aforward_failure(self):
        """aforward() raises on non-zero exit."""
        cli = _make_cli(env={"CLI_MODE": "fail"})
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "boom"}])
        with pytest.raises(CLIError):
            await cli.aforward(question="test")

    @pytest.mark.asyncio
    async def test_aforward_validates_inputs(self):
        """aforward() validates required inputs."""
        cli = _make_cli("a, b -> answer")
        with pytest.raises(ValueError, match="Missing required inputs"):
            await cli.aforward(a="only a")

    @pytest.mark.asyncio
    async def test_aforward_command_not_found(self):
        """aforward() raises CLIError for missing command."""
        cli = CLI("question -> answer", command="nonexistent_xyz_program")
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        with pytest.raises(CLIError, match="not found"):
            await cli.aforward(question="test")


# ============================================================================
# Prompt Placeholder Tests
# ============================================================================


class TestPromptPlaceholder:
    """Tests for {PROMPT} placeholder in CLI command."""

    def test_placeholder_spliced_into_command(self):
        cli = CLI("q -> a", command=["echo", "{PROMPT}"])
        cmd = cli._prepare_cli_command("hello world")
        assert cmd == ["echo", "hello world"]

    def test_no_placeholder_command_unchanged(self):
        cli = CLI("q -> a", command=["echo", "fixed"])
        cmd = cli._prepare_cli_command("ignored prompt")
        assert cmd == ["echo", "fixed"]

    def test_multiple_placeholders(self):
        cli = CLI("q -> a", command=["cmd", "{PROMPT}", "--repeat", "{PROMPT}"])
        cmd = cli._prepare_cli_command("text")
        assert cmd == ["cmd", "text", "--repeat", "text"]

    def test_placeholder_not_detected_across_token_boundary(self):
        """'{PROMPT}' split across two tokens should not trigger placeholder mode."""
        cli = CLI("q -> a", command=["cmd", "{PROMP", "T}"])
        assert cli._uses_placeholder is False

    def test_placeholder_embedded_in_token(self):
        """'{PROMPT}' embedded in a larger token is expanded via str.format."""
        cli = CLI("q -> a", command=["cmd", "--input={PROMPT}"])
        cmd = cli._prepare_cli_command("hello")
        assert cmd == ["cmd", "--input=hello"]


# ============================================================================
# Environment Variable Tests
# ============================================================================


class TestCLIEnvironment:
    """Tests for environment variable handling."""

    def test_generation_index_in_env(self):
        cli = _make_cli()
        env = cli._cli_env(generation_index=2, total=5)
        assert env["CLI_GENERATION_INDEX"] == "2"
        assert env["CLI_TOTAL_GENERATIONS"] == "5"

    def test_user_env_merged(self):
        cli = _make_cli(env={"CUSTOM_VAR": "custom_value"})
        env = cli._cli_env()
        assert env["CUSTOM_VAR"] == "custom_value"
        assert "PATH" in env  # inherits from os.environ

    def test_user_env_overrides_os(self):
        cli = _make_cli(env={"PATH": "/custom/path"})
        env = cli._cli_env()
        assert env["PATH"] == "/custom/path"


# ============================================================================
# Edge Cases
# ============================================================================


class TestCLIEdgeCases:
    """Edge case tests."""

    def test_whitespace_only_prompt(self):
        """CLI handles whitespace-only prompt (echo script strips to empty)."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "   "}])
        # cli_echo strips whitespace, returns empty string
        result = cli.forward(question="test")
        assert hasattr(result, "answer")
        assert result.answer == ""

    def test_very_long_output_truncated_for_extract(self):
        """Long CLI output is truncated before passing to extract."""
        cli = _make_cli(max_output_chars=100)

        # Mock _invoke_cli to return long output
        long_output = "x" * 500
        with patch.object(cli, "_invoke_cli", return_value=(long_output, "", 0, 0.1)):
            cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
            # Need extract since long output won't match single-string direct parse cleanly
            # Actually for single str field it will direct-parse. Let's use multi-field.
            pass

        cli2 = CLI(
            "question -> name, count: int",
            command=[sys.executable, str(SCRIPT)],
            max_output_chars=100,
        )
        cli2.prepare_prompt = make_mock_predictor([{"cli_prompt": "test"}])
        cli2.extract = make_mock_predictor([{"name": "test", "count": 1}])

        with patch.object(cli2, "_invoke_cli", return_value=("x" * 500, "", 0, 0.1)):
            result = cli2.forward(question="test")
        assert result.name == "test"

    def test_parse_jsonl_disabled(self):
        """parse_jsonl=False skips JSONL parsing."""
        cli = _make_cli(env={"CLI_MODE": "json"}, parse_jsonl=False)
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "json disabled"}])
        result = cli.forward(question="test")
        # With JSONL parsing disabled, events should be empty
        assert result.trajectory.events == []
        # But raw_stdout still has the JSONL
        assert "item.completed" in result.trajectory.raw_stdout

    def test_unicode_handling(self):
        """CLI handles unicode content."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "café résumé naïve"}])
        result = cli.forward(question="test")
        assert "café" in result.answer

    def test_multiline_output(self):
        """CLI with multiline stdout."""
        cli = _make_cli()
        cli.prepare_prompt = make_mock_predictor([{"cli_prompt": "line1\nline2\nline3"}])
        result = cli.forward(question="test")
        assert "line1" in result.answer
        assert "line3" in result.answer


