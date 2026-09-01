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
import json
import keyword
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic

import dspy
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.dsp.utils.settings import main_thread_config
from dspy.primitives.code_interpreter import (
    SIMPLE_TYPES,
    SUB_DSPY_FACTORY_NAME,
    CodeExecutionError,
    CodeInterpreter,
    CodeInterpreterError,
    FinalOutput,
    InterpreterCapability,
    _create_interpreter,
    _validate_interpreter,
    _validate_interpreter_factory,
    interpreter_capabilities,
)
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

# Appended to the interpreter rules when the interpreter declares the sub-dspy capability.
SUB_DSPY_INSTRUCTIONS = f"""
Sub-agents (dspy):
This environment can run dspy itself. You may `import dspy` and build sub-agents in the REPL for
subtasks that need structured inputs/outputs; a default LM is preconfigured when available.
- `dspy.Predict("question -> answer")(question=...)` or `dspy.ChainOfThought(...)` - single-step sub-agents.
- `dspy.ReActV2("question -> answer", tools=[my_tool])(question=...)` - a multi-step tool-using
  sub-agent; pass plain functions you define in the REPL as tools.
- `dspy.RLM("context, query -> answer", interpreter_factory={SUB_DSPY_FACTORY_NAME})(context=..., query=...)` -
  a recursive sub-agent with its own REPL; always pass the provided `{SUB_DSPY_FACTORY_NAME}`.
  This is the heaviest option: reserve it for deep subtasks whose input is itself too large or
  structured to prompt directly, and prefer Predict/ChainOfThought/ReActV2 for everything else.
Prefer `llm_query` for simple one-shot prompts; use sub-agents for structured, tool-using, or
recursive subtasks. Never call `dspy.configure(...)`: the LM configuration is already provided.
"""

# Appended to the interpreter rules when the interpreter declares the facade-dspy capability.
FACADE_DSPY_INSTRUCTIONS = """
Sub-agents (dspy):
A dspy facade is available in the REPL. Build sub-agents for subtasks that need structured
inputs/outputs; their LM calls run on the host.
- `dspy.Predict("question -> answer")(question=...)` or `dspy.ChainOfThought(...)` - single-step sub-agents.
- `dspy.ReActV2("question -> answer", tools=[...])` / `dspy.RLM("context, query -> answer")` - tool-using
  and recursive sub-agents. Only the provided tools listed above may be passed to them; functions
  you define in the REPL cannot cross to the host.
Prefer `llm_query` for simple one-shot prompts; use sub-agents for structured, tool-using, or
recursive subtasks. dspy.RLM is the heaviest option: reserve it for deep subtasks.
"""

# Sandbox-side variables for the setup and execution code below.
_SUB_DSPY_LM_STATE_VAR = "__dspy_sub_lm_state"
_SUB_DSPY_CODE_VAR = "__dspy_code"

# Validates the SUB_DSPY contract at invocation start.
SUB_DSPY_SETUP_CODE = f"""import dspy
if not callable(globals().get("{SUB_DSPY_FACTORY_NAME}")):
    raise RuntimeError(
        "This interpreter declares InterpreterCapability.SUB_DSPY but does not provide "
        "{SUB_DSPY_FACTORY_NAME} in its execution namespace."
    )
"""

# Runs one generated code block under a scoped override with the sub-agent LM (an explicit
# sub_lm or the host's default).
SUB_DSPY_EXEC_CODE = f"""import dspy as __dspy
with __dspy.context(lm=__dspy.BaseLM.load_state({_SUB_DSPY_LM_STATE_VAR})):
    exec({_SUB_DSPY_CODE_VAR}, globals())
"""


def _flex_bridge():
    """Deferred import: flex loads after rlm in dspy's import graph."""
    from dspy.predict.flex import bridge

    return bridge


class _FacadeRuntime:
    """Backs flex's ``_Invocation`` for RLM's dspy facade.

    Predictors are built with the RLM's tools and interpreter factory, run on the host with
    ``sub_lm`` when set, and get their own predictor-call budget of ``max_llm_calls``.
    """

    def __init__(self, rlm: RLM) -> None:
        self._rlm = rlm
        self._max_predictor_calls = rlm.max_llm_calls

    def _build_predictor(self, kind: str, signature: Any, kwargs: dict[str, Any] | None) -> Any:
        bridge = _flex_bridge()
        cls = getattr(dspy, kind)
        extra = {key: self._decode_tools(value) for key, value in (kwargs or {}).items()}
        if "interpreter_factory" not in extra and bridge._accepts_interpreter_factory(cls):
            extra["interpreter_factory"] = self._rlm._interpreter_factory
        predictor = cls(bridge._resolve_signature(signature), **extra)
        if self._rlm.sub_lm is not None:
            predictor.set_lm(self._rlm.sub_lm)
        return predictor

    def _decode_tools(self, value: Any) -> Any:
        if isinstance(value, dict) and _flex_bridge().TOOL_MARKER in value:
            name = value[_flex_bridge().TOOL_MARKER]
            tool = self._rlm._user_tools.get(name)
            if tool is None:
                raise CodeInterpreterError(
                    f"Sandboxed code passed tool {name!r} to a sub-agent, but it was not provided "
                    "to RLM(tools=...); functions defined in the REPL cannot cross to the host."
                )
            return tool
        if isinstance(value, list):
            return [self._decode_tools(item) for item in value]
        if isinstance(value, dict):
            return {key: self._decode_tools(item) for key, item in value.items()}
        return value


_PYTHON_FENCE_LANGS = {"python", "py", "python3", "py3", ""}


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

    The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but
    ``interpreter_factory`` can create another CodeInterpreter implementation,
    such as an adapter for a remote sandbox. RLM updates the interpreter's
    mutable ``tools`` dictionary with invocation-scoped tools before execution.
    A caller-owned interpreter may be reused sequentially with the same RLM
    instance, but must not be shared by overlapping invocations.

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
            max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log detailed execution info.
            tools: List of tool functions or dspy.Tool objects callable from interpreter code.
                  Built-in tools: llm_query(prompt), llm_query_batched(prompts).
            sub_lm: LM for llm_query/llm_query_batched and, on a sub-dspy interpreter, for
                   in-sandbox sub-agents (applied per code block; when given explicitly it
                   must be a plain, serializable dspy.LM). Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates
                the returned interpreter's mutable ``tools`` dictionary before execution. The callable may expose
                an ``execution_instructions`` string describing its runtime for the action prompt, and a
                ``capabilities`` declaration.
        """
        super().__init__()
        _validate_interpreter_factory(interpreter_factory)
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter_factory = interpreter_factory
        capabilities = interpreter_capabilities(interpreter_factory)
        self._sub_dspy = InterpreterCapability.SUB_DSPY in capabilities
        self._facade_dspy = not self._sub_dspy and InterpreterCapability.FACADE_DSPY in capabilities
        if self._sub_dspy and sub_lm is not None:
            self._serialized_sub_lm_state()
        self._user_tools = self._normalize_tools(tools)
        self._validate_namespace(self._user_tools)

        # Build the action and extract signatures
        action_sig, extract_sig = self._build_signatures()
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
        """Validate names owned by the RLM result and sandbox APIs."""
        reserved_sandbox_names = self._RESERVED_SANDBOX_NAMES
        if self._sub_dspy:
            # In a sub-dspy sandbox, the `dspy` module and the environment-provided
            # nested-interpreter factory are part of the execution namespace.
            reserved_sandbox_names = reserved_sandbox_names | {"dspy", SUB_DSPY_FACTORY_NAME}
        if self._facade_dspy:
            bridge = _flex_bridge()
            reserved_sandbox_names = reserved_sandbox_names | {"dspy", bridge.CONSTRUCT_TOOL, bridge.CALL_TOOL}
        for name in tools:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier and not a keyword")
            if name in reserved_sandbox_names:
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")

        input_names = set(self.signature.input_fields)
        reserved_inputs = sorted(input_names & reserved_sandbox_names)
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

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        """Create llm_query and llm_query_batched tools with a fresh call counter."""
        state = {"call_count": 0}
        lock = threading.Lock()
        lm = self.sub_lm

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        f"Use Python code for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _query_lm(prompt: str) -> str:
            target_lm = lm if lm is not None else dspy.settings.lm
            if target_lm is None:
                raise dspy.LMNotConfiguredError(
                    "No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM."
                )
            response = target_lm(prompt)
            if isinstance(response, dspy.LMResponse):
                text = response.text
            elif isinstance(response, list) and response:
                first_output = response[0]
                text = first_output.get("text") if isinstance(first_output, dict) else first_output
            else:
                raise TypeError(
                    "Sub-LM must return dspy.LMResponse or a non-empty list of text outputs, "
                    f"got {type(response).__name__}."
                )

            if not isinstance(text, str):
                raise TypeError(f"Sub-LM response must contain text, got {type(text).__name__}.")
            return text

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query prompts concurrently, isolating LM failures while propagating contract errors."""
            if not prompts:
                return []
            _check_and_increment(len(prompts))

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

        execution_instructions = getattr(self._interpreter_factory, "execution_instructions", "")
        if not isinstance(execution_instructions, str):
            raise TypeError("interpreter_factory.execution_instructions must be a string")
        interpreter_rules = f"\nExecution environment:\n{execution_instructions}\n" if execution_instructions else ""
        if self._sub_dspy:
            interpreter_rules += SUB_DSPY_INSTRUCTIONS
        elif self._facade_dspy:
            interpreter_rules += FACADE_DSPY_INSTRUCTIONS

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
        if "interpreter" in input_args and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument when calling the module."
            )
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

    def _serialized_sub_lm_state(self) -> dict[str, Any] | None:
        """JSON-able reconstruction state for the sub-agent LM.

        Returns None when no host LM crosses the boundary (the environment's own LM applies);
        raises when an explicit sub_lm cannot cross, since silently substituting the ambient
        LM would violate the caller's model choice.
        """
        lm = self.sub_lm if self.sub_lm is not None else dspy.settings.lm
        state = None
        if type(lm) is dspy.LM:
            state = lm.dump_state()
            try:
                json.dumps(state)
            except (TypeError, ValueError):
                state = None
        if state is None and self.sub_lm is not None:
            raise ValueError(
                "sub_lm must be a plain dspy.LM with JSON-serializable state to cross into a sub-dspy "
                "interpreter; for custom LMs, configure the LM in the interpreter's environment instead."
            )
        return state

    def _setup_dspy_support(self, repl: CodeInterpreter) -> None:
        """Validate the SUB_DSPY contract or install the dspy facade before generated code runs."""
        if self._sub_dspy:
            repl.execute(SUB_DSPY_SETUP_CODE)
        elif self._facade_dspy:
            repl.execute(_flex_bridge().SHIM_SETUP)

    @contextmanager
    def _host_settings_guard(self) -> Iterator[None]:
        """Keep the host's global dspy settings structure intact across generated-code execution.

        Restores top-level settings that generated code replaced (e.g. via ``dspy.configure``)
        and the contents of the callbacks/stream_listeners registries it mutated in place;
        restoration mutates the original containers so aliases such as active context snapshots
        heal too. ``trace`` is left alone because dspy itself appends to it during a forward,
        and everything is compared by identity so user ``__eq__`` never runs here. Internal
        state of user objects reachable from settings is the interpreter's isolation boundary,
        not RLM's: run untrusted code in a worker-process backend.
        """
        if not self._sub_dspy:
            yield
            return
        originals = dict(main_thread_config)
        registries = {
            name: list(value)
            for name in ("callbacks", "stream_listeners")
            if isinstance(value := main_thread_config.get(name), list)
        }
        try:
            yield
        finally:
            replaced = {
                key
                for key in originals.keys() | main_thread_config.keys()
                if main_thread_config.get(key) is not originals.get(key)
            }
            mutated = {
                name
                for name, items in registries.items()
                if len(originals[name]) != len(items)
                or any(current is not item for current, item in zip(originals[name], items, strict=False))
            }
            if replaced or mutated:
                logger.warning(
                    "Generated code changed global dspy settings %s; restoring them.", sorted(replaced | mutated)
                )
                main_thread_config.clear()
                main_thread_config.update(originals)
                for name in mutated:
                    originals[name][:] = registries[name]

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

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Create fresh LLM tools (and facade bridge tools) and merge with user-provided tools."""
        execution_tools = self._make_llm_tools()
        if self._facade_dspy:
            bridge = _flex_bridge()
            invocation = bridge._Invocation(_FacadeRuntime(self), {})
            execution_tools[bridge.CONSTRUCT_TOOL] = invocation.construct
            execution_tools[bridge.CALL_TOOL] = invocation.call
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
        interpreter: CodeInterpreter | None,
    ) -> Iterator[CodeInterpreter]:
        """Yield a caller-owned interpreter or manage a factory-created one."""
        with self._host_settings_guard():
            if interpreter is not None:
                _validate_interpreter(interpreter)
                self._inject_execution_context(interpreter, execution_tools)
                yield interpreter
                return

            interpreter = _create_interpreter(self._interpreter_factory)
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
        variables = dict(input_args)
        if self._sub_dspy and (state := self._serialized_sub_lm_state()) is not None:
            variables[_SUB_DSPY_CODE_VAR] = code
            variables[_SUB_DSPY_LM_STATE_VAR] = state
            code = SUB_DSPY_EXEC_CODE
        try:
            return repl.execute(code, variables=variables)
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
    ) -> Prediction | REPLHistory:
        """Execute one iteration. Returns Prediction if done, else updated REPLHistory."""
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
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

    def forward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Execute RLM to produce outputs from the given inputs.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
                output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
                this RLM instance.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            self._setup_dspy_support(repl)
            history: REPLHistory = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names
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
    ) -> Prediction | REPLHistory:
        """Async version: Execute one iteration."""
        variables_info = [variable.format() for variable in variables]
        pred = await self.generate_action.acall(
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

    async def aforward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction:
        """Async version of forward(). Execute RLM to produce outputs.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. RLM injects invocation tools and
                output metadata into it but does not shut it down. Reuse is supported only for sequential calls to
                this RLM instance.
            **input_args: Input values matching the signature's input fields.

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
            CodeInterpreterError: If interpreter setup, process, or protocol fails
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools, interpreter) as repl:
            regular_args = self._prepare_serializable_vars(input_args, repl)
            self._setup_dspy_support(repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, regular_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
