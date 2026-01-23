"""
Recursive Language Model (RLM) module for DSPy.

RLMs are an inference strategy where LLMs treat long contexts as part of an external
environment rather than feeding them directly to the model. The LLM writes Python code
to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic

import dspy
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.code_interpreter import SIMPLE_TYPES, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.repl_types import REPLHistory, REPLVariable
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

# Pattern to match markdown code fences: ```python\n...\n``` or ```\n...\n```
_CODE_FENCE_PATTERN = re.compile(r"^```(?:python|py)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    match = _CODE_FENCE_PATTERN.match(code)
    if match:
        return match.group(1)
    return code


@experimental
class RLM(Module):
    """Recursive Language Model module.

    Uses a sandboxed REPL to let the LLM programmatically explore large contexts
    through code execution. The LLM writes Python code to examine data, call
    sub-LLMs for semantic analysis, and build up answers iteratively.

    The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but you
    can provide any CodeInterpreter implementation (e.g., MockInterpreter, or write a custom one using E2B or Modal).

    Note: RLM instances are not thread-safe when using a custom interpreter.
    Create separate RLM instances for concurrent use, or use the default
    PythonInterpreter which creates a fresh instance per forward() call.

    Example:
        ```python
        # Basic usage
        rlm = dspy.RLM("context, query -> output", max_iterations=10)
        result = rlm(context="...very long text...", query="What is the magic number?")
        print(result.output)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        verbose: bool = False,
        tools: dict[str, Callable[..., str]] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter: CodeInterpreter | None = None,
    ):
        """
        Args:
            signature: Defines inputs and outputs. String like "context, query -> answer"
                      or a Signature class.
            max_iterations: Maximum REPL interaction iterations.
            max_llm_calls: Maximum sub-LLM calls (llm_query/llm_query_batched) per execution.
            max_output_chars: Maximum characters to include from REPL output.
            verbose: Whether to log detailed execution info.
            tools: Additional tool functions callable from interpreter code.
                  Built-in tools: llm_query(prompt), llm_query_batched(prompts).
            sub_lm: LM for llm_query/llm_query_batched. Defaults to dspy.settings.lm.
                   Allows using a different (e.g., cheaper) model for sub-queries.
            interpreter: CodeInterpreter implementation to use. Defaults to PythonInterpreter.
        """
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.max_output_chars = max_output_chars
        self.verbose = verbose
        self.sub_lm = sub_lm
        self._interpreter = interpreter
        self._user_tools = tools or {}
        self._validate_tools(self._user_tools)

        # Build the action and extract signatures
        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    # =========================================================================
    # Tool Creation and Validation
    # =========================================================================

    # Reserved tool names that conflict with built-in sandbox functions
    _RESERVED_TOOL_NAMES = frozenset({"llm_query", "llm_query_batched", "SUBMIT", "print"})

    def _validate_tools(self, tools: dict[str, Callable]) -> None:
        """Validate user-provided tools have valid names and are callable."""
        for name, func in tools.items():
            if not name.isidentifier():
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier")
            if name in self._RESERVED_TOOL_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")
            if not callable(func):
                raise TypeError(f"Tool '{name}' must be callable, got {type(func).__name__}")

    def _format_tool_docs(self, tools: dict[str, Callable]) -> str:
        """Format user-provided tools for inclusion in instructions."""
        if not tools:
            return ""

        lines = ["\nAdditional tools available (use these instead of standard library equivalents):"]
        for name, func in tools.items():
            # Get function signature with return type
            try:
                sig = inspect.signature(func)
                params = []
                for p in sig.parameters.values():
                    if p.annotation != inspect.Parameter.empty:
                        type_name = getattr(p.annotation, "__name__", str(p.annotation))
                        params.append(f"{p.name}: {type_name}")
                    else:
                        params.append(p.name)
                params_str = ", ".join(params)

                # Get return type
                if sig.return_annotation != inspect.Parameter.empty:
                    ret_type = getattr(sig.return_annotation, "__name__", str(sig.return_annotation))
                    sig_str = f"{name}({params_str}) -> {ret_type}"
                else:
                    sig_str = f"{name}({params_str})"
            except (ValueError, TypeError):
                sig_str = f"{name}(...)"

            # Get first line of docstring
            doc = func.__doc__.strip().split("\n")[0] if func.__doc__ else "No description"
            lines.append(f"- `{sig_str}` - {doc}")

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
                raise RuntimeError("No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM.")
            response = target_lm(prompt)
            if isinstance(response, list) and response:
                item = response[0]
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
                return item
            return str(response)

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query the LLM with multiple prompts concurrently."""
            if not prompts:
                return []
            _check_and_increment(len(prompts))

            results: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(_query_lm, p): i for i, p in enumerate(prompts)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"[ERROR] {e}"
            return [results[i] for i in range(len(prompts))]

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    @property
    def tools(self) -> dict[str, Callable]:
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
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"), type_=str)
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
            variables.append(REPLVariable.from_value(name, value, field_info=field_info))
        return variables

    def _format_output(self, output: str) -> str:
        """Format and truncate REPL output."""
        if not output:
            return "(no output - did you forget to print?)"
        if len(output) > self.max_output_chars:
            return output[:self.max_output_chars] + "\n... (truncated)"
        return output

    def _validate_inputs(self, input_args: dict[str, Any]) -> None:
        """Raise ValueError if required input fields are missing."""
        missing = set(self.signature.input_fields.keys()) - set(input_args.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {sorted(missing)}")

    # =========================================================================
    # CodeInterpreter Lifecycle
    # =========================================================================

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Create fresh LLM tools and merge with user-provided tools."""
        execution_tools = self._make_llm_tools()
        execution_tools.update(self._user_tools)
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
    def _interpreter_context(self, execution_tools: dict[str, Callable]) -> Iterator[CodeInterpreter]:
        """Yield interpreter, creating PythonInterpreter if none provided at init."""
        if self._interpreter is not None:
            self._inject_execution_context(self._interpreter, execution_tools)
            yield self._interpreter
        else:
            repl = PythonInterpreter(
                tools=execution_tools,
                output_fields=self._get_output_fields_info(),
            )
            try:
                yield repl
            finally:
                repl.shutdown()

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
        pred: Any,
        result: Any,
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Process interpreter result, returning Prediction if final, else updated history.

        This shared helper reduces duplication between sync and async execution paths.

        Args:
            pred: The prediction containing reasoning and code attributes
            result: Result from interpreter.execute() - FinalOutput, list, str, or error string
            history: Current REPL history
            output_field_names: List of expected output field names

        Returns:
            Prediction if FINAL was called successfully, else updated REPLHistory
        """
        # Strip markdown fences from code for history (format() will re-add them)
        code = _strip_code_fences(pred.code)
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
        return history.append(reasoning=pred.reasoning, code=code, output=output)

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
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        try:
            code = _strip_code_fences(action.code)
            result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as e:
            result = f"[Error] {e}"

        return self._process_execution_result(action, result, history, output_field_names)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def forward(self, **input_args) -> Prediction:
        """Execute RLM to produce outputs from the given inputs.

        Args:
            **input_args: Input values matching the signature's input fields

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history: REPLHistory = REPLHistory()

            for iteration in range(self.max_iterations):
                result: Prediction | REPLHistory = self._execute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
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
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {pred.reasoning}\nCode:\n{pred.code}"
            )

        try:
            code = _strip_code_fences(pred.code)
            result = repl.execute(code, variables=dict(input_args))
        except (CodeInterpreterError, SyntaxError) as e:
            result = f"[Error] {e}"

        return self._process_execution_result(pred, result, history, output_field_names)

    async def aforward(self, **input_args) -> Prediction:
        """Async version of forward(). Execute RLM to produce outputs.

        Args:
            **input_args: Input values matching the signature's input fields

        Returns:
            Prediction with output field(s) from the signature and 'trajectory' for debugging

        Raises:
            ValueError: If required input fields are missing
        """
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        with self._interpreter_context(execution_tools) as repl:
            history = REPLHistory()

            for iteration in range(self.max_iterations):
                result = await self._aexecute_iteration(
                    repl, variables, history, iteration, input_args, output_field_names
                )
                if isinstance(result, Prediction):
                    return result
                history = result

            # Max iterations reached - use extract fallback
            return await self._aextract_fallback(variables, history, output_field_names)
