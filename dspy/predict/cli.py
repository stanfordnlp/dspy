"""
CLI module for DSPy.

Wraps arbitrary CLI programs (coding agents, scripts, etc.) as optimizable
DSPy modules. The CLI is an execution environment — like RLM's PythonInterpreter —
while optimizable Predict nodes control what gets sent to it and how output
is parsed.

Architecture follows dspy.RLM: the module exposes `prepare_prompt` and `extract`
as named predictors that optimizers (MIPRO, GEPA, BootstrapTrace) can tune.

Reference: https://github.com/stanfordnlp/dspy/issues/9034
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from typing import TYPE_CHECKING, Any, Sequence

import dspy
from dspy.primitives.cli_types import CLITrajectory, parse_jsonl_events
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# =============================================================================
# Errors
# =============================================================================


class CLIError(RuntimeError):
    """Raised when the CLI subprocess fails."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "", returncode: int = -1):
        details = [message]
        if stdout.strip():
            details.append(f"stdout:\n{stdout.strip()}")
        if stderr.strip():
            details.append(f"stderr:\n{stderr.strip()}")
        super().__init__("\n".join(details))
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# =============================================================================
# Signature Templates
# =============================================================================

PREPARE_INSTRUCTIONS_TEMPLATE = """You are preparing a prompt to send to a CLI agent.
Given the task inputs, produce a clear, detailed prompt that the CLI can act on.

The CLI agent will receive your prompt and return a text response.
Be specific about what output format you expect."""

EXTRACT_INSTRUCTIONS_TEMPLATE = """Based on the CLI agent's output, extract the final structured outputs.

Review the CLI output carefully and provide the requested fields.
If the output doesn't contain enough information, do your best to extract what's available."""


# =============================================================================
# CLI Module
# =============================================================================


PROMPT_PLACEHOLDER = "{prompt}"


# =============================================================================
# Agent Presets
# =============================================================================

# Each preset defines the CLI flags for non-interactive, auto-approve ("yolo")
# and print-mode operation. This is the CLI equivalent of what litellm does
# for API providers — you just name the agent and the module knows the flags.
#
# Keys:
#   command:      Base command tokens
#   print_flags:  Flags for non-interactive one-shot mode (print & exit)
#   yolo_flags:   Flags to auto-approve all tool use (no human in the loop)
#   output_flags: Flags to control output format
#   model_flag:   Flag to select model (e.g., --model)
#   session_flags: Flags to disable session persistence
#   parse_jsonl:  Whether the output format produces JSONL events

AGENT_PRESETS: dict[str, dict[str, Any]] = {
    "claude": {
        "command": ["claude"],
        "print_flags": ["-p"],
        "yolo_flags": ["--dangerously-skip-permissions"],
        "output_flags": ["--output-format", "text"],
        "model_flag": "--model",
        "session_flags": ["--no-session-persistence"],
        "parse_jsonl": False,

    },
    "claude-json": {
        "command": ["claude"],
        "print_flags": ["-p", "--verbose"],
        "yolo_flags": ["--dangerously-skip-permissions"],
        "output_flags": ["--output-format", "stream-json"],
        "model_flag": "--model",
        "session_flags": ["--no-session-persistence"],
        "parse_jsonl": True,

    },
    "codex": {
        "command": ["codex", "exec"],
        "print_flags": [],  # exec is already non-interactive
        "yolo_flags": ["--dangerously-bypass-approvals-and-sandbox"],
        "output_flags": [],
        "model_flag": "--model",
        "session_flags": [],
        "parse_jsonl": False,

    },
    "codex-json": {
        "command": ["codex", "exec"],
        "print_flags": [],
        "yolo_flags": ["--dangerously-bypass-approvals-and-sandbox"],
        "output_flags": ["--json"],
        "model_flag": "--model",
        "session_flags": [],
        "parse_jsonl": True,

    },
    "gemini": {
        "command": ["gemini"],
        "print_flags": [],  # uses -p flag for prompt, not "print mode"
        "yolo_flags": ["--yolo"],
        "output_flags": ["-o", "text"],
        "model_flag": "--model",
        "session_flags": [],
        "parse_jsonl": False,
        "stdin": True,  # stdin works, -p is for inline prompt
    },
    "gemini-json": {
        "command": ["gemini"],
        "print_flags": [],
        "yolo_flags": ["--yolo"],
        "output_flags": ["-o", "json"],
        "model_flag": "--model",
        "session_flags": [],
        "parse_jsonl": True,

    },
    "pi": {
        "command": ["pi"],
        "print_flags": ["-p"],
        "yolo_flags": [],  # pi doesn't need approval bypass
        "output_flags": [],
        "model_flag": "--model",
        "session_flags": ["--no-session"],
        "parse_jsonl": False,

    },
    "pi-json": {
        "command": ["pi"],
        "print_flags": ["-p"],
        "yolo_flags": [],
        "output_flags": ["--mode", "json"],
        "model_flag": "--model",
        "session_flags": ["--no-session"],
        "parse_jsonl": True,

    },
}


def _build_agent_command(
    agent: str,
    *,
    model: str | None = None,
    yolo: bool = False,
    extra_flags: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Build a CLI command from an agent preset.

    Args:
        agent: Agent name (e.g., "claude", "codex-json", "pi").
        model: Model to use (e.g., "sonnet", "o3"). None = agent default.
        yolo: Whether to auto-approve all tool use.
        extra_flags: Additional CLI flags to append.

    Returns:
        Tuple of (command_list, parse_jsonl).

    Raises:
        ValueError: If agent name is not recognized.
    """
    if agent not in AGENT_PRESETS:
        available = ", ".join(sorted(AGENT_PRESETS.keys()))
        raise ValueError(f"Unknown agent {agent!r}. Available: {available}")

    preset = AGENT_PRESETS[agent]
    cmd = list(preset["command"])
    cmd.extend(preset["print_flags"])

    if yolo:
        cmd.extend(preset["yolo_flags"])

    cmd.extend(preset["output_flags"])
    cmd.extend(preset["session_flags"])

    if model is not None:
        cmd.extend([preset["model_flag"], model])

    if extra_flags:
        cmd.extend(extra_flags)

    return cmd, preset["parse_jsonl"]


@experimental
class CLI(Module):
    """CLI module — wraps any stdin/stdout CLI as an optimizable DSPy module.

    Like dspy.RLM wraps a sandboxed Python REPL, dspy.CLI wraps a CLI subprocess.
    The module exposes two optimizable Predict nodes:
    - `prepare_prompt`: Converts signature inputs into a prompt for the CLI
    - `extract`: Parses CLI stdout into structured signature outputs

    Optimizers discover these via `named_predictors()` and can tune their
    instructions, demos, and few-shot examples.

    Use ``from_agent()`` for built-in presets that handle each CLI's flags
    for non-interactive, auto-approve ("yolo"), and print mode:

    Example:
        ```python
        # Using agent presets (recommended):
        cli = dspy.CLI.from_agent("claude", "question -> answer")
        cli = dspy.CLI.from_agent("codex", "task -> result", model="o3")
        cli = dspy.CLI.from_agent("gemini", "question -> answer")
        cli = dspy.CLI.from_agent("pi", "question -> answer", model="gemini-2.5-flash")

        # Or with a raw command:
        cli = dspy.CLI("question -> answer", cli_command="my-cli --flag")

        result = cli(question="What is the capital of France?")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        cli_command: Sequence[str] | str,
        *,
        # Subprocess config
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        encoding: str = "utf-8",
        # Budget controls
        max_time: float | None = None,
        max_retries: int = 0,
        # Output parsing
        parse_jsonl: bool = True,
        max_output_chars: int = 100_000,
        # Execution
        verbose: bool = False,
    ):
        """
        Args:
            signature: Defines inputs and outputs. String like "question -> answer"
                      or a Signature class.
            cli_command: CLI command to execute. String (shell-split) or list.
                        Use "{prompt}" placeholder to splice prompt into args,
                        otherwise prompt is piped via stdin.
            env: Extra environment variables for the subprocess.
            cwd: Working directory for the subprocess.
            encoding: Text encoding for subprocess I/O.
            max_time: Maximum wall-clock seconds per CLI execution. None = no limit.
            max_retries: Number of retries on non-zero exit code.
            parse_jsonl: Whether to attempt JSONL event parsing on stdout.
            max_output_chars: Maximum chars to include in trajectory for Predict inputs.
            verbose: Whether to log detailed execution info.
        """
        super().__init__()
        self.signature = ensure_signature(signature)

        # Parse CLI command
        if isinstance(cli_command, str):
            cli_command = shlex.split(cli_command)
        if not cli_command:
            raise ValueError("cli_command cannot be empty")
        self.cli_command = list(cli_command)
        self._uses_placeholder = PROMPT_PLACEHOLDER in " ".join(self.cli_command)

        # Subprocess config
        self.env = dict(env or {})
        self.cwd = cwd
        self.encoding = encoding

        # Budget controls
        self.max_time = max_time
        self.max_retries = max_retries

        # Output parsing
        self.parse_jsonl = parse_jsonl
        self.max_output_chars = max_output_chars
        self.verbose = verbose

        # Build optimizable Predict nodes
        prepare_sig, extract_sig = self._build_signatures()
        self.prepare_prompt = dspy.Predict(prepare_sig)
        self.extract = dspy.Predict(extract_sig)

    @classmethod
    def from_agent(
        cls,
        agent: str,
        signature: type[Signature] | str,
        *,
        model: str | None = None,
        yolo: bool = False,
        extra_flags: list[str] | None = None,
        **kwargs,
    ) -> CLI:
        """Create a CLI module from a named agent preset.

        This is the recommended way to create CLI modules for known coding agents.
        Each preset knows the right flags for non-interactive, auto-approve, and
        session-less operation — the CLI equivalent of what litellm does for API
        providers.

        Available agents: claude, claude-json, codex, codex-json, gemini, gemini-json, pi, pi-json

        Args:
            agent: Agent name (e.g., "claude", "codex", "pi", "gemini").
                   Append "-json" for JSONL structured output (e.g., "codex-json").
            signature: DSPy signature (e.g., "question -> answer").
            model: Model to use (e.g., "sonnet", "o3"). None = agent default.
            yolo: Auto-approve all tool use (default False). Set True to skip
                  human approval for tool use.
            extra_flags: Additional CLI flags to append to the command.
            **kwargs: Additional keyword arguments passed to CLI.__init__
                     (e.g., max_time, max_retries, cwd, env).

        Returns:
            A configured CLI module.

        Example:
            ```python
            # Claude Code with Sonnet
            cli = dspy.CLI.from_agent("claude", "task -> result", model="sonnet")

            # Codex with JSONL events for rich trajectory
            cli = dspy.CLI.from_agent("codex-json", "task -> result", model="o3")

            # Gemini CLI
            cli = dspy.CLI.from_agent("gemini", "question -> answer")

            # Pi with a specific model
            cli = dspy.CLI.from_agent("pi", "question -> answer", model="gemini-2.5-flash")

            # With budget controls
            cli = dspy.CLI.from_agent("claude", "task -> result", max_time=120, max_retries=2)
            ```
        """
        command, preset_parse_jsonl = _build_agent_command(
            agent,
            model=model,
            yolo=yolo,
            extra_flags=extra_flags,
        )

        # Let preset control parse_jsonl unless user explicitly overrides
        if "parse_jsonl" not in kwargs:
            kwargs["parse_jsonl"] = preset_parse_jsonl

        return cls(
            signature=signature,
            cli_command=command,
            **kwargs,
        )

    # =========================================================================
    # Signature Construction
    # =========================================================================

    def _build_signatures(self) -> tuple:
        """Build the prepare_prompt and extract signatures.

        prepare_prompt: takes signature inputs → produces cli_prompt
        extract: takes cli_output → produces signature outputs
        """
        task_instructions = self.signature.instructions or ""
        input_names = list(self.signature.input_fields.keys())
        output_names = list(self.signature.output_fields.keys())

        # Format output field descriptions for the prepare prompt
        output_descs = []
        for name, field in self.signature.output_fields.items():
            desc = field.json_schema_extra.get("desc", "") if field.json_schema_extra else ""
            annotation = getattr(field, "annotation", str)
            type_name = getattr(annotation, "__name__", str(annotation))
            output_descs.append(f"- `{name}` ({type_name}): {desc}" if desc else f"- `{name}` ({type_name})")

        output_fields_str = "\n".join(output_descs)

        # Prepare prompt signature
        prepare_instructions = task_instructions
        if prepare_instructions:
            prepare_instructions += "\n\n"
        prepare_instructions += PREPARE_INSTRUCTIONS_TEMPLATE
        prepare_instructions += f"\n\nThe CLI should produce these outputs:\n{output_fields_str}"

        prepare_sig = dspy.Signature({}, prepare_instructions)
        for name, field in self.signature.input_fields.items():
            desc = field.json_schema_extra.get("desc", "") if field.json_schema_extra else ""
            prepare_sig = prepare_sig.append(
                name,
                dspy.InputField(desc=desc or f"Input: {name}"),
                type_=field.annotation,
            )
        prepare_sig = prepare_sig.append(
            "cli_prompt",
            dspy.OutputField(desc="The prompt to send to the CLI agent. Be specific and detailed."),
            type_=str,
        )

        # Extract signature
        extract_instructions = ""
        if task_instructions:
            extract_instructions = f"Original task: {task_instructions}\n\n"
        extract_instructions += EXTRACT_INSTRUCTIONS_TEMPLATE

        extract_sig = dspy.Signature({}, extract_instructions)
        extract_sig = extract_sig.append(
            "cli_output",
            dspy.InputField(desc="The raw output from the CLI agent"),
            type_=str,
        )
        for name, field in self.signature.output_fields.items():
            desc = field.json_schema_extra.get("desc", "") if field.json_schema_extra else ""
            extract_sig = extract_sig.append(
                name,
                dspy.OutputField(desc=desc or f"Output: {name}"),
                type_=field.annotation,
            )

        return prepare_sig, extract_sig

    # =========================================================================
    # CLI Invocation
    # =========================================================================

    def _prepare_cli_command(self, prompt_text: str) -> list[str]:
        """Build the actual command list, splicing in prompt if placeholder is used."""
        if not self._uses_placeholder:
            return list(self.cli_command)
        return [
            prompt_text if token == PROMPT_PLACEHOLDER else token
            for token in self.cli_command
        ]

    def _cli_env(self, generation_index: int = 0, total: int = 1) -> dict[str, str]:
        """Build environment for the subprocess."""
        env = os.environ.copy()
        env.update(self.env)
        env["CLI_GENERATION_INDEX"] = str(generation_index)
        env["CLI_TOTAL_GENERATIONS"] = str(total)
        return env

    def _invoke_cli(
        self,
        prompt_text: str,
        *,
        generation_index: int = 0,
        total: int = 1,
    ) -> tuple[str, str, int, float]:
        """Execute the CLI subprocess synchronously.

        Returns:
            Tuple of (stdout, stderr, returncode, elapsed_seconds)

        Raises:
            CLIError: On subprocess failure after all retries exhausted.
        """
        command = self._prepare_cli_command(prompt_text)
        env = self._cli_env(generation_index, total)
        should_pipe = not self._uses_placeholder

        last_error: CLIError | None = None

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                run_kwargs: dict[str, Any] = dict(
                    capture_output=True,
                    text=True,
                    encoding=self.encoding,
                    cwd=self.cwd,
                    env=env,
                    timeout=self.max_time,
                    check=False,
                )
                if should_pipe:
                    run_kwargs["input"] = prompt_text

                completed = subprocess.run(command, **run_kwargs)
                elapsed = time.monotonic() - start

                if completed.returncode == 0:
                    return completed.stdout, completed.stderr or "", 0, elapsed

                last_error = CLIError(
                    f"CLI exited with status {completed.returncode} (attempt {attempt + 1}/{self.max_retries + 1})",
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                    returncode=completed.returncode,
                )

                if self.verbose:
                    logger.warning(f"CLI attempt {attempt + 1} failed: {last_error}")

            except FileNotFoundError as exc:
                raise CLIError(
                    f"CLI command not found: {shlex.join(self.cli_command)}",
                    returncode=-1,
                ) from exc
            except subprocess.TimeoutExpired as exc:
                elapsed = time.monotonic() - start
                raise CLIError(
                    f"CLI timed out after {self.max_time}s",
                    stdout=str(exc.stdout or ""),
                    stderr=str(exc.stderr or ""),
                    returncode=-1,
                ) from exc

        # All retries exhausted
        assert last_error is not None
        raise last_error

    async def _invoke_cli_async(
        self,
        prompt_text: str,
        *,
        generation_index: int = 0,
        total: int = 1,
    ) -> tuple[str, str, int, float]:
        """Execute the CLI subprocess asynchronously.

        Returns:
            Tuple of (stdout, stderr, returncode, elapsed_seconds)

        Raises:
            CLIError: On subprocess failure after all retries exhausted.
        """
        command = self._prepare_cli_command(prompt_text)
        env = self._cli_env(generation_index, total)
        should_pipe = not self._uses_placeholder

        last_error: CLIError | None = None

        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE if should_pipe else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.cwd,
                    env=env,
                )

                input_bytes = prompt_text.encode(self.encoding) if should_pipe else None

                if self.max_time is not None:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(input_bytes),
                        timeout=self.max_time,
                    )
                else:
                    stdout_bytes, stderr_bytes = await process.communicate(input_bytes)

                elapsed = time.monotonic() - start
                stdout = stdout_bytes.decode(self.encoding, errors="replace")
                stderr = stderr_bytes.decode(self.encoding, errors="replace")

                if process.returncode == 0:
                    return stdout, stderr, 0, elapsed

                last_error = CLIError(
                    f"CLI exited with status {process.returncode} (attempt {attempt + 1}/{self.max_retries + 1})",
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode or -1,
                )

            except FileNotFoundError as exc:
                raise CLIError(
                    f"CLI command not found: {shlex.join(self.cli_command)}",
                    returncode=-1,
                ) from exc
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise CLIError(
                    f"CLI timed out after {self.max_time}s",
                    returncode=-1,
                ) from exc

        assert last_error is not None
        raise last_error

    # =========================================================================
    # Output Parsing
    # =========================================================================

    def _build_trajectory(
        self,
        prompt_text: str,
        stdout: str,
        stderr: str,
        returncode: int,
        elapsed: float,
    ) -> CLITrajectory:
        """Build a CLITrajectory from subprocess results."""
        events = parse_jsonl_events(stdout) if self.parse_jsonl else []
        return CLITrajectory(
            prompt=prompt_text,
            events=events,
            raw_stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed=elapsed,
        )

    def _try_direct_parse(self, stdout: str, trajectory: CLITrajectory) -> dict[str, Any] | None:
        """Try to parse CLI output without an LM call.

        Returns parsed outputs dict if successful, None otherwise.

        Parse strategies (in order):
        1. If structured events have an agent_message → use that as source text
        2. Single string output field → return stdout directly
        3. JSON output matching field names → parse directly
        """
        output_fields = self.signature.output_fields
        output_names = list(output_fields.keys())

        # Get the best text to parse from
        source_text = trajectory.get_agent_message() or stdout.strip()

        # Strategy 1: Single string output → return directly
        if len(output_names) == 1:
            name = output_names[0]
            annotation = getattr(output_fields[name], "annotation", str)
            if annotation is str:
                return {name: source_text}

        # Strategy 2: Try JSON parse matching output fields
        try:
            data = json.loads(source_text)
            if isinstance(data, dict) and all(name in data for name in output_names):
                return {name: data[name] for name in output_names}
        except (json.JSONDecodeError, TypeError):
            pass

        return None

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(self, **input_args) -> Prediction:
        """Execute CLI module to produce outputs from the given inputs.

        Steps:
        1. prepare_prompt (optimizable Predict) → cli_prompt
        2. Invoke CLI subprocess
        3. Try direct parse (no LM cost)
        4. Fall back to extract (optimizable Predict)

        Args:
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output fields from the signature, plus 'trajectory'.

        Raises:
            ValueError: If required input fields are missing.
            CLIError: If CLI fails after all retries.
        """
        self._validate_inputs(input_args)
        output_names = list(self.signature.output_fields.keys())

        # Step 1: Prepare prompt (optimizable)
        prep = self.prepare_prompt(**input_args)
        prompt_text = prep.cli_prompt

        if self.verbose:
            logger.info(f"CLI prompt:\n{prompt_text}")

        # Step 2: Invoke CLI
        stdout, stderr, returncode, elapsed = self._invoke_cli(prompt_text)

        if self.verbose:
            logger.info(f"CLI completed in {elapsed:.1f}s (exit={returncode})")

        # Step 3: Build trajectory
        trajectory = self._build_trajectory(prompt_text, stdout, stderr, returncode, elapsed)

        # Step 4: Try direct parse (cheap path)
        parsed = self._try_direct_parse(stdout, trajectory)
        if parsed is not None:
            return Prediction(
                trajectory=trajectory,
                **parsed,
            )

        # Step 5: Extract via LM (optimizable fallback)
        cli_output_text = stdout.strip()
        if len(cli_output_text) > self.max_output_chars:
            cli_output_text = cli_output_text[:self.max_output_chars] + "\n... (truncated)"

        extract_pred = self.extract(cli_output=cli_output_text)
        return Prediction(
            trajectory=trajectory,
            **{name: getattr(extract_pred, name) for name in output_names},
        )

    async def aforward(self, **input_args) -> Prediction:
        """Async version of forward().

        Args:
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output fields from the signature, plus 'trajectory'.
        """
        self._validate_inputs(input_args)
        output_names = list(self.signature.output_fields.keys())

        # Step 1: Prepare prompt (optimizable)
        prep = await self.prepare_prompt.acall(**input_args)
        prompt_text = prep.cli_prompt

        # Step 2: Invoke CLI
        stdout, stderr, returncode, elapsed = await self._invoke_cli_async(prompt_text)

        # Step 3: Build trajectory
        trajectory = self._build_trajectory(prompt_text, stdout, stderr, returncode, elapsed)

        # Step 4: Try direct parse
        parsed = self._try_direct_parse(stdout, trajectory)
        if parsed is not None:
            return Prediction(trajectory=trajectory, **parsed)

        # Step 5: Extract fallback
        cli_output_text = stdout.strip()
        if len(cli_output_text) > self.max_output_chars:
            cli_output_text = cli_output_text[:self.max_output_chars] + "\n... (truncated)"

        extract_pred = await self.extract.acall(cli_output=cli_output_text)
        return Prediction(
            trajectory=trajectory,
            **{name: getattr(extract_pred, name) for name in output_names},
        )

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        """Raise ValueError if required input fields are missing."""
        missing = set(self.signature.input_fields.keys()) - set(input_args.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    # =========================================================================
    # Serialization
    # =========================================================================

    def dump_state(self) -> dict[str, Any]:
        """Serialize module state for saving/loading."""
        return {
            "signature": str(self.signature),
            "cli_command": list(self.cli_command),
            "env": {k: v for k, v in self.env.items() if "key" not in k.lower() and "secret" not in k.lower()},
            "cwd": self.cwd,
            "encoding": self.encoding,
            "max_time": self.max_time,
            "max_retries": self.max_retries,
            "parse_jsonl": self.parse_jsonl,
            "max_output_chars": self.max_output_chars,
            "prepare_prompt": self.prepare_prompt.dump_state(),
            "extract": self.extract.dump_state(),
        }

    def load_state(self, state: dict) -> "CLI":
        """Load saved state of a CLI module."""
        self.cli_command = state.get("cli_command", self.cli_command)
        self.env = state.get("env", self.env)
        self.cwd = state.get("cwd", self.cwd)
        self.encoding = state.get("encoding", self.encoding)
        self.max_time = state.get("max_time", self.max_time)
        self.max_retries = state.get("max_retries", self.max_retries)
        self.parse_jsonl = state.get("parse_jsonl", self.parse_jsonl)
        self.max_output_chars = state.get("max_output_chars", self.max_output_chars)
        self._uses_placeholder = PROMPT_PLACEHOLDER in " ".join(self.cli_command)
        if "prepare_prompt" in state:
            self.prepare_prompt.load_state(state["prepare_prompt"])
        if "extract" in state:
            self.extract.load_state(state["extract"])
        return self
