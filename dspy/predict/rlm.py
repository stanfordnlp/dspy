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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic

import dspy
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import (
    SIMPLE_TYPES,
    CodeExecutionError,
    CodeInterpreter,
    FinalOutput,
    _create_interpreter,
    _validate_interpreter,
    _validate_interpreter_factory,
)
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental

if TYPE_CHECKING:

    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# TODO: Optimize this prompt across a diverse benchmark

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""

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
            sub_lm: LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns. RLM updates
                the returned interpreter's mutable ``tools`` dictionary before execution.
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
        for name in tools:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier and not a keyword")
            if name in self._RESERVED_SANDBOX_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")

        input_names = set(self.signature.input_fields)
        reserved_inputs = sorted(input_names & self._RESERVED_SANDBOX_NAMES)
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
                        results[idx] = f"[ERROR] {e}"
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

        action_sig = (
            dspy.Signature({}, task_instructions + ACTION_INSTRUCTIONS_TEMPLATE.format(
                inputs=inputs_str, final_output_names=final_output_names, output_fields=output_fields,
                max_llm_calls=self.max_llm_calls,
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
        """Apply declared defaults and validate inputs against the signature."""
        if "interpreter" in input_args and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument when calling the module."
            )
        input_names = set(self.signature.input_fields)
        unexpected = set(input_args) - input_names
        if unexpected:
            raise ValueError(f"Unexpected inputs not declared in the signature: {sorted(unexpected)}")

        for name, field in self.signature.input_fields.items():
            if name not in input_args and not field.is_required():
                input_args[name] = field.default_factory() if field.default_factory is not None else field.default

        missing = input_names - set(input_args)
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    def _initialize_inputs(
        self, input_args: dict[str, Any], repl: CodeInterpreter,
    ) -> None:
        """Initialize the interpreter namespace once for this RLM run."""
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

        if regular_args:
            repl.execute("pass", variables=regular_args)

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
        """Create fresh LLM tools and merge with user-provided tools."""
        execution_tools = self._make_llm_tools()
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
        reasoning: str,
        code: str,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Process interpreter result, returning Prediction if final, else updated history.

        This shared helper reduces duplication between sync and async execution paths.

        Args:
            reasoning: Validated reasoning for the executed action
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
            return history.append(reasoning=reasoning, code=code, output=output)

        # Handle FINAL output
        if isinstance(result, FinalOutput):
            parsed_outputs, error = self._process_final_output(result, output_field_names)

            if error:
                return history.append(reasoning=reasoning, code=code, output=error)

            final_history = history.append(
                reasoning=reasoning, code=code, output=f"FINAL: {parsed_outputs}"
            )
            return Prediction(
                **parsed_outputs,
                trajectory=[e.model_dump() for e in final_history],
                final_reasoning=reasoning,
            )

        # Format non-final result as output
        if isinstance(result, list):
            output = "\n".join(map(str, result))
        else:
            output = str(result) if result else ""

        output = self._format_output(output)
        if self.verbose:
            logger.info(REPLEntry.format_output(output, self.max_output_chars))
        return history.append(reasoning=reasoning, code=code, output=output)

    def _execute_code(
        self,
        repl: CodeInterpreter,
        code: str,
    ) -> Any:
        """Execute code in the interpreter, returning the result or an error string."""
        try:
            return repl.execute(code)
        except (CodeExecutionError, SyntaxError) as e:
            return f"[Error] {e}"

    def _execute_action(
        self,
        repl: CodeInterpreter,
        action: Prediction,
        history: REPLHistory,
        iteration: int,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Validate and execute one generated action."""
        reasoning = action.get("reasoning")
        raw_code = action.get("code")

        for name, value in (("reasoning", reasoning), ("code", raw_code)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"RLM action field `{name}` must be str, got {type(value).__name__}")

        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iters}\n"
                f"Reasoning: {reasoning}\nCode:\n{raw_code}"
            )

        errors = []
        if reasoning is None or not reasoning.strip():
            errors.append("`reasoning` must be a non-empty string.")

        code = raw_code or ""
        parse_error = None
        if raw_code is None or not raw_code.strip():
            code = ""
            errors.append("`code` must contain non-empty Python code.")
        else:
            try:
                code = _strip_code_fences(raw_code)
            except SyntaxError as error:
                parse_error = str(error)
            else:
                if not code:
                    errors.append("`code` must contain non-empty Python code.")

        if errors or parse_error:
            if errors:
                if parse_error:
                    errors.append(parse_error)
                error_output = f"[Error] Malformed RLM action: {' '.join(errors)}"
            else:
                error_output = f"[Error] {parse_error}"
            output = self._format_output(error_output)
            return history.append(
                reasoning=reasoning if reasoning and reasoning.strip() else "",
                code=code,
                output=output,
            )

        result = self._execute_code(repl, code)
        return self._process_execution_result(reasoning, code, result, history, output_field_names)

    def _execute_iteration(
        self,
        repl: CodeInterpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Execute one iteration. Returns Prediction if done, else updated REPLHistory."""
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        return self._execute_action(repl, action, history, iteration, output_field_names)

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
            self._initialize_inputs(input_args, repl)
            history: REPLHistory = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, output_field_names
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
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Async version: Execute one iteration."""
        variables_info = [variable.format() for variable in variables]
        pred = await self.generate_action.acall(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        return self._execute_action(repl, pred, history, iteration, output_field_names)

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
            self._initialize_inputs(input_args, repl)
            history = REPLHistory(max_output_chars=self.max_output_chars)

            for iteration in range(self.max_iters):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
