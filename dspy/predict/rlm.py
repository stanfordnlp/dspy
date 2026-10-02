"""
Recursive Language Model (RLM) module for DSPy.

RLMs are an inference strategy where LLMs treat long contexts as part of an external
environment rather than feeding them directly to the model. The LLM writes Python code
to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import base64
import contextvars
import functools
import inspect
import keyword
import logging
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic

import dspy
from dspy.adapters.types.decision import Choice, Noul, Score
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import (
    SIMPLE_TYPES,
    CodeExecutionError,
    CodeInterpreter,
    FinalOutput,
    _validate_interpreter,
    _validate_interpreter_factory,
    resolve_interpreter_factory,
)
from dspy.primitives.facade import CALL_TOOL, CONSTRUCT_TOOL, FacadeInvocation, is_reserved_sandbox_name
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.exceptions import format_error_for_lm

if TYPE_CHECKING:

    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# TODO: Optimize this prompt across a diverse benchmark

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.
{interpreter_rules}
Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""

# Appended to the interpreter rules whenever RLM installs the sandbox dspy facade (every factory-made interpreter).
SUB_AGENT_INSTRUCTIONS = """
Sub-agents (dspy):
You may `import dspy` and build sub-agents in the REPL for subtasks that need structured inputs/outputs.
- `dspy.Predict("question -> answer")(question=...)` or `dspy.ChainOfThought(...)` - single-step sub-agents.
- `dspy.ReActV2("question -> answer", tools=[...])(question=...)` - a multi-step tool-using sub-agent.
  Only the provided tools listed above may be passed; functions you define in the REPL cannot cross to the host.
- `dspy.RLM("context, query -> answer")(context=..., query=...)` - a recursive sub-agent with its own
  REPL on the same interpreter backend as yours; it cannot be given another `interpreter_factory`.
  This is the heaviest option: reserve it for deep subtasks whose input is itself too large or
  structured to prompt directly, and prefer Predict/ChainOfThought/ReActV2 for everything else.
Prefer `llm_query` for simple one-shot prompts; use sub-agents for structured, tool-using, or
recursive subtasks.
"""
_PYTHON_FENCE_LANGS = {"python", "py", "python3", "py3", ""}


class _LLMCallBudget:
    """Per-forward ceiling on sub-LLM calls, shared by every host path generated code can reach."""

    def __init__(self, limit: int):
        self._limit = limit
        self._count = 0
        self._lock = threading.Lock()

    def reserve(self, n: int = 1) -> None:
        with self._lock:
            if self._count + n > self._limit:
                raise RuntimeError(
                    f"LLM call limit exceeded: {self._count} + {n} > {self._limit}. "
                    f"Use Python code for aggregation instead of making more LLM calls."
                )
            self._count += n


def _strip_code_fences(code: str) -> str:
    """Extract Python code from markdown fences, or return as-is if no fences."""
    code = code.strip()
    if "```" not in code:
        return code

    # Strip outer decorative fence pairs (e.g. ```\n```python\n...\n```\n```)
    lines = code.splitlines()
    while len(lines) >= 2 and lines[0].strip() == "```" and lines[-1].strip() == "```":
        lines.pop(0)
        lines.pop()
    code = "\n".join(lines).strip()
    if "```" not in code:
        return code

    # Find the first opening fence (skip any text before it)
    fence_start = code.find("```")
    lang_line, separator, remainder = code[fence_start + 3:].partition("\n")
    if not separator:
        return code

    # Accept python-labeled fences or bare ``` fences; reject explicit non-Python tags
    lang = (lang_line.strip().split(maxsplit=1)[0] if lang_line.strip() else "").lower()
    if lang not in _PYTHON_FENCE_LANGS:
        raise SyntaxError(f"Expected Python code but got ```{lang} fence. Write Python code, not {lang}.")

    # Find closing fence
    block_end = remainder.find("```")
    if block_end == -1:
        return remainder.strip()

    return remainder[:block_end].strip()


@experimental
class RLM(Module):
    """Recursive Language Model module.

    Uses a sandboxed REPL to let the LLM programmatically explore large contexts
    through code execution. The LLM writes Python code to examine data, call
    sub-LLMs for semantic analysis, and build up answers iteratively.

    ``interpreter_factory`` defaults to ``PythonInterpreter`` (Deno/Pyodide/WASM), and
    ``dspy.configure(interpreter_factory=...)`` replaces that default. Either route
    accepts an adapter for a remote sandbox.
    RLM updates the interpreter's mutable ``tools`` dictionary with
    invocation-scoped tools before execution. Pass a zero-argument factory via
    ``interpreter_factory=`` at call time to override the runtime for one invocation.
    RLM shuts down every interpreter it creates.

    Examples:
        ```python
        # Basic usage
        rlm = dspy.RLM("context, query -> output", max_iters=10)
        result = rlm(context="...very long text...", query="What is the magic number?")
        print(result.output)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iters: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    ):
        """
        Args:
            signature: Defines inputs and outputs. String like "context, query -> answer"
                      or a Signature class.
            max_iters: Maximum REPL interaction iterations.
            max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched and sub-agent LM calls) per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log detailed execution info.
            tools: List of tool functions or dspy.Tool objects callable from interpreter code.
                  Built-in tools: llm_query(prompt), llm_query_batched(prompts).
            sub_lm: LM for llm_query/llm_query_batched and sub-agents. Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates
                the returned interpreter's mutable ``tools`` dictionary before execution. The callable may expose
                an ``execution_instructions`` string describing its runtime for the action prompt. RLM installs the
                sandbox dspy facade into every interpreter it creates, so the runtime must be able to host it
                (see ``CodeInterpreter``). RLM applies the
                active factory's instructions to each action call, so ``dspy.context`` can switch runtimes
                without changing the shared predictor signature. Defaults to ``dspy.PythonInterpreter``;
                ``dspy.configure(interpreter_factory=...)`` replaces the default.
        """
        super().__init__()
        _validate_interpreter_factory(interpreter_factory)
        self.signature = ensure_signature(signature)
        if any(
            kind.extract_custom_type_from_annotation(field.rebuild_annotation())
            for field in self.signature.output_fields.values()
            for kind in (Noul, Choice, Score)
        ):
            warnings.warn(
                "RLM support for Noul, Choice, and Score outputs is not implemented consistently: "
                "decision evidence decoding is not guaranteed, including for nested output types. "
                "Use Predict with top-level decision outputs instead.",
                UserWarning,
                stacklevel=2,
            )
        self.max_iters = max_iters
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter_factory = interpreter_factory
        self._user_tools = self._normalize_tools(tools)
        self._validate_namespace(self._user_tools)

        # Build the action and extract signatures
        action_sig, extract_sig = self._build_signatures()
        self._action_signature = action_sig
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    # =========================================================================
    # Tool Creation and Validation
    # =========================================================================

    # Names owned by RLM rather than the user-provided signature or tools.
    _RESERVED_SANDBOX_NAMES = frozenset({"llm_query", "llm_query_batched", "SUBMIT", "print"})
    _RESERVED_RESULT_NAMES = frozenset({"trajectory", "final_reasoning"})

    def _normalize_tools(self, tools: list[Callable] | None) -> dict[str, Tool]:
        """Normalize tools list to a dict of Tool objects keyed by name."""
        if not tools:
            return {}

        if isinstance(tools, dict):
            raise TypeError(
                "tools must be a list, not a dict. "
                "Change tools={'name': func} to tools=[func] "
                "(tool names are inferred from function names, or use dspy.Tool(func, name='custom_name'))"
            )

        def to_tool(func: Callable | Tool) -> Tool:
            if isinstance(func, Tool):
                return func
            if not callable(func):
                raise TypeError(f"Tool {func!r} must be callable, got {type(func).__name__}")
            return Tool(func)

        normalized = {}
        for value in tools:
            tool = to_tool(value)
            if tool.name in normalized:
                raise ValueError(f"Duplicate tool name '{tool.name}'")
            normalized[tool.name] = tool
        return normalized

    def _validate_namespace(self, tools: dict[str, Tool]) -> None:
        """Validate names owned by the RLM call, result, sandbox APIs, and the sandbox dspy facade."""
        def is_reserved(name: str) -> bool:
            return name in self._RESERVED_SANDBOX_NAMES or is_reserved_sandbox_name(name)

        for name in tools:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier and not a keyword")
            if is_reserved(name):
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")

        input_names = set(self.signature.input_fields)
        if "interpreter_factory" in input_names:
            raise ValueError("'interpreter_factory' is reserved for RLM runtime configuration, not a signature input.")
        reserved_inputs = sorted(name for name in input_names if is_reserved(name))
        if reserved_inputs:
            raise ValueError(f"Input fields conflict with built-in sandbox functions: {reserved_inputs}")

        tool_inputs = sorted(input_names & tools.keys())
        if tool_inputs:
            raise ValueError(f"Input fields conflict with user tools: {tool_inputs}")

        reserved_outputs = sorted(set(self.signature.output_fields) & self._RESERVED_RESULT_NAMES)
        if reserved_outputs:
            raise ValueError(f"Output fields conflict with RLM result metadata: {reserved_outputs}")

    def _format_tool_docs(self, tools: dict[str, Tool]) -> str:
        """Format user-provided tools for inclusion in instructions."""
        if not tools:
            return ""

        lines = ["\nAdditional tools available (use these instead of standard library equivalents):"]
        for tool in tools.values():
            # Build signature string from Tool's args
            params = []
            for arg_name, arg_schema in (tool.args or {}).items():
                arg_type = arg_schema.get("type", "Any")
                params.append(f"{arg_name}: {arg_type}")
            params_str = ", ".join(params)
            sig_str = f"{tool.name}({params_str})"

            # Get description with newlines escaped
            desc = (tool.desc or "No description").replace("\n", "  ")
            lines.append(f"- `{sig_str}` - {desc}")

        return "\n".join(lines)

    def _make_llm_tools(self, budget: _LLMCallBudget | None = None, max_workers: int = 8) -> dict[str, Callable]:
        """Create llm_query and llm_query_batched tools drawing on ``budget`` (fresh by default)."""
        budget = budget if budget is not None else _LLMCallBudget(self.max_llm_calls)
        lm = self.sub_lm

        def _query_lm(prompt: str) -> str:
            target_lm = lm if lm is not None else dspy.settings.lm
            if target_lm is None:
                raise dspy.LMNotConfiguredError(
                    "No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM."
                )
            response = target_lm(prompt)
            if isinstance(response, dspy.lm15.Response):
                text = response.text
            elif isinstance(response, list) and response:
                first_output = response[0]
                text = first_output.get("text") if isinstance(first_output, dict) else first_output
            else:
                raise TypeError(
                    "Sub-LM must return dspy.lm15.Response or a non-empty list of text outputs, "
                    f"got {type(response).__name__}."
                )

            if not isinstance(text, str):
                raise TypeError(f"Sub-LM response must contain text, got {type(text).__name__}.")
            return text

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt cannot be empty")
            budget.reserve(1)
            return _query_lm(prompt)

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query prompts concurrently, isolating LM failures while propagating contract errors."""
            if not prompts:
                return []
            budget.reserve(len(prompts))

            results: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(contextvars.copy_context().run, _query_lm, prompt): index
                    for index, prompt in enumerate(prompts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except dspy.LMError as e:
                        results[idx] = f"[ERROR] {format_error_for_lm(e)}"
            return [results[i] for i in range(len(prompts))]

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    @property
    def tools(self) -> dict[str, Tool]:
        """User-provided tools (excludes internal llm_query/llm_query_batched)."""
        return dict(self._user_tools)

    # =========================================================================
    # Signature Building
    # =========================================================================

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """Build the action and extract signatures from templates."""
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)

        # Simple names for SUBMIT() examples
        final_output_names = ", ".join(self.signature.output_fields.keys())

        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )

        # Include original signature instructions (docstring) if present
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""

        # Format tool documentation for user-provided tools
        tool_docs = self._format_tool_docs(self._user_tools)

        # Seed with the factory active at construction; each action call refreshes this
        # fragment, so a later dspy.context() can select a different runtime.
        factory = resolve_interpreter_factory(self._interpreter_factory)
        execution_instructions = self._get_execution_instructions(factory)
        self._initial_execution_instructions = execution_instructions
        self._initial_sub_agent_rules = SUB_AGENT_INSTRUCTIONS
        self._initial_interpreter_rules = self._format_interpreter_rules(
            execution_instructions, self._initial_sub_agent_rules
        )
        interpreter_rules = self._initial_interpreter_rules

        action_sig = (
            dspy.Signature({}, task_instructions + ACTION_INSTRUCTIONS_TEMPLATE.format(
                inputs=inputs_str, final_output_names=final_output_names, output_fields=output_fields,
                max_llm_calls=self.max_llm_calls, interpreter_rules=interpreter_rules,
            ) + tool_docs)
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=REPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iters"), type_=str)
            .append("reasoning", dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."), type_=str)
            .append("code", dspy.OutputField(desc="Python code to execute. Use markdown code block format: ```python\\n<code>\\n```"), type_=str)
        )

        # Extract signature: includes the original signature's output fields and task instructions.
        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

        # Prepend original task instructions to extract instructions so the LLM knows what task to extract for
        extended_task_instructions = ""
        if task_instructions:
            extended_task_instructions = "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
        full_extract_instructions = extended_task_instructions + extract_instructions

        extract_sig = dspy.Signature(
            {**self.signature.output_fields},
            full_extract_instructions,
        )
        extract_sig = extract_sig.prepend("repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=REPLHistory)
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str)

        return action_sig, extract_sig

    @staticmethod
    def _format_interpreter_rules(execution_instructions: str, sub_agent_rules: str) -> str:
        rules = f"\nExecution environment:\n{execution_instructions}\n" if execution_instructions else ""
        return rules + sub_agent_rules

    @staticmethod
    def _get_execution_instructions(factory: Callable[[], CodeInterpreter]) -> str:
        execution_instructions = getattr(factory, "execution_instructions", "")
        if not isinstance(execution_instructions, str):
            raise TypeError("interpreter_factory.execution_instructions must be a string")
        return execution_instructions

    def _action_signature_for_current_factory(
        self, factory: Callable[[], CodeInterpreter], sub_agent_rules: str
    ) -> type[Signature]:
        """Return an action signature whose runtime guidance matches the active interpreter.

        ``sub_agent_rules`` is the sub-agent guidance for this invocation's interpreter (see ``_setup_facade``).
        """
        execution_instructions = self._get_execution_instructions(factory)
        # getattr, because generate_action may have been replaced by a predictor that
        # carries no signature of its own.
        current_signature = getattr(self.generate_action, "signature", self._action_signature)

        # Same runtime as at construction: no derived signature per iteration, and an
        # optimizer's revised instructions stay untouched.
        if (
            execution_instructions == self._initial_execution_instructions
            and sub_agent_rules == self._initial_sub_agent_rules
        ):
            return current_signature

        # Replace only DSPy's own fragment, so an override cannot stack on stale guidance.
        current_instructions = current_signature.instructions
        current_rules = self._format_interpreter_rules(execution_instructions, sub_agent_rules)
        if self._initial_interpreter_rules and self._initial_interpreter_rules in current_instructions:
            instructions = current_instructions.replace(self._initial_interpreter_rules, current_rules, 1)
        elif current_rules:
            # An optimizer dropped that fragment, so append the active guidance once instead.
            instructions = current_instructions + current_rules
        else:
            instructions = current_instructions
        return current_signature.with_instructions(instructions)

    # =========================================================================
    # Input/Output Processing
    # =========================================================================

    def _get_output_fields_info(self) -> list[dict]:
        """Get output field info for sandbox registration."""
        fields = []
        for name, field in self.signature.output_fields.items():
            annotation = getattr(field, "annotation", str)
            field_info = {"name": name}
            # Only include type for simple types that work in function signatures
            # Complex types like Literal, Union, etc. are not included
            if annotation in SIMPLE_TYPES:
                field_info["type"] = annotation.__name__
            fields.append(field_info)
        return fields

    def _build_variables(self, **input_args: Any) -> list[REPLVariable]:
        """Build REPLVariable list from input arguments with field metadata."""
        variables = []
        for name, value in input_args.items():
            field_info = self.signature.input_fields.get(name)
            if isinstance(value, SandboxSerializable):
                var = build_repl_variable(value, name, field_info=field_info)
            else:
                var = REPLVariable.from_value(name, value, field_info=field_info)
            variables.append(var)
        return variables

    def _format_output(self, output: str) -> str:
        if not output:
            return "(no output - did you forget to print?)"
        return output

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        """Validate call-time arguments against the signature's input namespace."""
        input_names = set(self.signature.input_fields)
        unexpected = set(input_args) - input_names
        if unexpected:
            raise ValueError(f"Unexpected inputs not declared in the signature: {sorted(unexpected)}")

        missing = input_names - set(input_args)
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    def _prepare_serializable_vars(
        self, input_args: dict[str, Any], repl: CodeInterpreter,
    ) -> dict[str, Any]:
        """Inject SandboxSerializable values into the interpreter.

        For each SandboxSerializable value in input_args, serializes it and
        executes setup + assignment code in the interpreter. Returns the
        remaining non-serializable args (for per-iteration use).
        """
        repl.start()
        regular_args = {}
        for name, value in input_args.items():
            if not isinstance(value, SandboxSerializable):
                regular_args[name] = value
                continue

            payload = value.to_sandbox()
            setup = value.sandbox_setup()
            raw_var_name = f"_raw_{name}"
            assignment = value.sandbox_assignment(name, raw_var_name)
            code_lines = []
            payload_vars: dict[str, str] = {}
            if isinstance(payload, bytes):
                try:
                    payload_vars[raw_var_name] = payload.decode("utf-8")
                except UnicodeDecodeError:
                    encoded_var_name = f"{raw_var_name}_base64"
                    payload_vars[encoded_var_name] = base64.b64encode(payload).decode("ascii")
                    code_lines.extend([
                        "import base64",
                        f"{raw_var_name} = base64.b64decode({encoded_var_name})",
                    ])
            else:
                payload_vars[raw_var_name] = str(payload)

            if setup:
                code_lines.append(setup)
            code_lines.append(assignment)
            repl.execute("\n".join(code_lines), variables=payload_vars)

        return regular_args

    def _setup_facade(
        self, repl: CodeInterpreter, budget: _LLMCallBudget, interpreter_factory: Callable[[], CodeInterpreter]
    ) -> str:
        """Install the sandbox dspy facade into this invocation's interpreter; return its sub-agent guidance.

        Nested code-executing sub-agents get their interpreters from ``interpreter_factory``, the factory this
        invocation runs on. The facade resolves ``sub_lm`` (else ``dspy.settings.lm``) per call, like
        ``llm_query``, and charges ``max_llm_calls``.
        """
        invocation = FacadeInvocation(
            self._user_tools, interpreter_factory, None, lm=self.sub_lm, reserve=budget.reserve
        )
        try:
            invocation.install(repl)
        except CodeExecutionError as e:
            # The runtime cannot host the shim (e.g. it runs code in the host's memory): no sub-agents this call.
            for name in (CONSTRUCT_TOOL, CALL_TOOL):
                repl.tools.pop(name, None)
            logger.warning("RLM sub-agents are unavailable on %s: %s", type(repl).__name__, e)
            return ""
        return SUB_AGENT_INSTRUCTIONS

    # =========================================================================
    # CodeInterpreter Lifecycle
    # =========================================================================

    def _make_interpreter_tool(self, tool: Tool) -> Callable:
        """Preserve function metadata while routing execution through Tool."""
        if inspect.iscoroutinefunction(tool.func) or inspect.iscoroutinefunction(getattr(tool.func, "__call__", None)):
            async def invoke(**kwargs):
                return await tool.acall(**kwargs)
        else:
            def invoke(**kwargs):
                return tool(**kwargs)

        functools.update_wrapper(invoke, tool.func)
        invoke.__signature__ = inspect.signature(tool.func)
        return invoke

    def _prepare_execution_tools(self, budget: _LLMCallBudget | None = None) -> dict[str, Callable]:
        """Create the LLM tools on ``budget`` (fresh by default) and merge with user-provided tools."""
        budget = budget if budget is not None else _LLMCallBudget(self.max_llm_calls)
        execution_tools = self._make_llm_tools(budget)
        execution_tools.update({name: self._make_interpreter_tool(tool) for name, tool in self._user_tools.items()})
        return execution_tools

    def _inject_execution_context(self, interpreter: CodeInterpreter, execution_tools: dict[str, Callable]) -> None:
        """Inject execution tools and output fields into an interpreter.

        This ensures llm_query, llm_query_batched, and typed FINAL signatures are available,
        even for user-provided interpreters. Each forward() call gets fresh tools with a
        fresh call counter, so we must inject on every execution.
        """
        interpreter.tools.update(execution_tools)
        if hasattr(interpreter, "output_fields"):
            interpreter.output_fields = self._get_output_fields_info()
        # Reset registration flag to force re-registration with fresh tools
        if hasattr(interpreter, "_tools_registered"):
            interpreter._tools_registered = False

    @contextmanager
    def _interpreter_context(
        self,
        execution_tools: dict[str, Callable],
        factory: Callable[[], CodeInterpreter],
    ) -> Iterator[CodeInterpreter]:
        """Create and close one interpreter for this invocation."""
        _validate_interpreter_factory(factory)
        interpreter = factory()
        _validate_interpreter(interpreter)
        try:
            self._inject_execution_context(interpreter, execution_tools)
            yield interpreter
        finally:
            interpreter.shutdown()

    # =========================================================================
    # Execution Core
    # =========================================================================

    def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        """Use extract module to get final output when max iterations reached."""
        logger.warning("RLM reached max iterations, using extract to get final output")

        variables_info = [variable.format() for variable in variables]
        extract_pred = self.extract(
            variables_info=variables_info,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def _process_final_output(
        self,
        result: FinalOutput,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate and parse FinalOutput. Returns (parsed_outputs, None) or (None, error)."""
        raw_output = result.output

        # Validate raw_output is a dict
        if not isinstance(raw_output, dict):
            return None, f"[Error] FINAL returned {type(raw_output).__name__}, expected dict with fields: {output_field_names}"

        # Validate all required output fields are present
        missing = set(output_field_names) - set(raw_output.keys())
        if missing:
            return None, f"[Error] Missing output fields: {sorted(missing)}. Use SUBMIT({', '.join(output_field_names)})"

        # Parse and validate each output field
        parsed_outputs = {}
        type_errors = []
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            try:
                parsed_outputs[name] = parse_value(raw_output[name], annotation)
            except (ValueError, pydantic.ValidationError) as e:
                type_errors.append(
                    f"{name}: expected {annotation.__name__ if hasattr(annotation, '__name__') else annotation}, "
                    f"got {type(raw_output[name]).__name__}: {e}"
                )

        if type_errors:
            return None, "[Type Error] " + "; ".join(type_errors)

        return parsed_outputs, None

    def _process_execution_result(
        self,
        pred: Prediction,
        code: str,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Process interpreter result, returning Prediction if final, else updated history.

        This shared helper reduces duplication between sync and async execution paths.

        Args:
            pred: The prediction containing reasoning and code attributes
            code: Code to record in history (already stripped when possible)
            result: Result from interpreter.execute() - FinalOutput, list, str, or error string
            history: Current REPL history
            output_field_names: List of expected output field names

        Returns:
            Prediction if FINAL was called successfully, else updated REPLHistory
        """
        # Handle error strings from caught exceptions
        if isinstance(result, str) and result.startswith("[Error]"):
            output = self._format_output(result)
            return history.append(reasoning=pred.reasoning, code=code, output=output)

        # Handle FINAL output
        if isinstance(result, FinalOutput):
            parsed_outputs, error = self._process_final_output(result, output_field_names)

            if error:
                return history.append(reasoning=pred.reasoning, code=code, output=error)

            final_history = history.append(
                reasoning=pred.reasoning, code=code, output=f"FINAL: {parsed_outputs}"
            )
            return Prediction(
                **parsed_outputs,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning=pred.reasoning,
            )

        # Format non-final result as output
        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""

        output = self._format_output(output)
        if self.verbose:
            logger.info(REPLEntry.format_output(output, self.max_output_chars))
        return history.append(reasoning=pred.reasoning, code=code, output=output)

    def _execute_code(
        self,
        repl: CodeInterpreter,
        code: str,
        input_args: dict[str, Any],
    ) -> Any:
        """Execute code in the interpreter, returning the result or an error string."""
        try:
            return repl.execute(code, variables=dict(input_args))
        except (CodeExecutionError, SyntaxError) as e:
            return f"[Error] {format_error_for_lm(e)}"

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
        interpreter_factory: Callable[[], CodeInterpreter],
        sub_agent_rules: str,
    ) -> Prediction | REPLHistory:
        """Execute one iteration. Returns Prediction if done, else updated REPLHistory."""
        variables_info = [variable.format() for variable in variables]
        # A per-call signature, not a mutation of generate_action.signature, keeps a
        # dspy.context override local to this invocation.
        action = self.generate_action(
            signature=self._action_signature_for_current_factory(interpreter_factory, sub_agent_rules),
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iters}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        try:
            code = _strip_code_fences(action.code)
        except SyntaxError as e:
            code = action.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(action, code, result, history, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(action, code, result, history, output_field_names)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def forward(self, *, interpreter_factory: Callable[[], CodeInterpreter] | None = None, **input_args) -> Prediction:
        """Execute RLM to produce outputs from the given inputs.

        Args:
            interpreter_factory: Optional zero-argument factory, passed by keyword. Overrides the constructor
                and configured factories for this invocation. Must return a fresh interpreter; RLM injects tools
                and output metadata and shuts it down on exit, including failures. Sub-agents built in the
                sandbox run their own code on this factory too.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)
        if interpreter_factory is None:
            interpreter_factory = resolve_interpreter_factory(self._interpreter_factory)

        output_field_names = list(self.signature.output_fields.keys())
        budget = _LLMCallBudget(self.max_llm_calls)
        execution_tools = self._prepare_execution_tools(budget)
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter_factory) as repl:
            sub_agent_rules = self._setup_facade(repl, budget, interpreter_factory)
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history: REPLHistory = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names, interpreter_factory,
                    sub_agent_rules,
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return self._extract_fallback(variables, history, output_field_names)

    async def _aextract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        """Async version: Use extract module when max iterations reached."""
        logger.warning("RLM reached max iterations, using extract to get final output")

        variables_info = [variable.format() for variable in variables]
        extract_pred = await self.extract.acall(
            variables_info=variables_info,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final output",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    async def _aexecute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
        interpreter_factory: Callable[[], CodeInterpreter],
        sub_agent_rules: str,
    ) -> Prediction | REPLHistory:
        """Async version: Execute one iteration."""
        variables_info = [variable.format() for variable in variables]
        pred = await self.generate_action.acall(
            signature=self._action_signature_for_current_factory(interpreter_factory, sub_agent_rules),
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iters}\n"
                f"Reasoning: {pred.reasoning}\nCode:\n{pred.code}"
            )

        try:
            code = _strip_code_fences(pred.code)
        except SyntaxError as e:
            code = pred.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(pred, code, result, history, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(pred, code, result, history, output_field_names)

    async def aforward(self, *, interpreter_factory: Callable[[], CodeInterpreter] | None = None, **input_args) -> Prediction:
        """Async version of forward(). Execute RLM to produce outputs.

        Args:
            interpreter_factory: Optional zero-argument factory, passed by keyword. Overrides the constructor
                and configured factories for this invocation. Must return a fresh interpreter; RLM injects tools
                and output metadata and shuts it down on exit, including failures. Sub-agents built in the
                sandbox run their own code on this factory too.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)
        if interpreter_factory is None:
            interpreter_factory = resolve_interpreter_factory(self._interpreter_factory)

        output_field_names = list(self.signature.output_fields.keys())
        budget = _LLMCallBudget(self.max_llm_calls)
        execution_tools = self._prepare_execution_tools(budget)
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter_factory) as repl:
            sub_agent_rules = self._setup_facade(repl, budget, interpreter_factory)
            regular_args = self._prepare_serializable_vars(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names, interpreter_factory,
                    sub_agent_rules,
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
