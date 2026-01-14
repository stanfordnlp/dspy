"""
Recursive Language Model (RLM) module for DSPy.

RLMs are an inference strategy where LLMs treat long contexts as part of an external
environment rather than feeding them directly to the model. The LLM writes Python code
to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

Reference: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator

import pydantic
from pydantic import Field

import dspy
from dspy.adapters.utils import parse_value, serialize_for_json, translate_field_type
from dspy.primitives.interpreter import SIMPLE_TYPES, FinalAnswerResult, Interpreter, InterpreterError
from dspy.primitives.local_interpreter import PythonInterpreter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import ensure_signature

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# TODO: Optimize this prompt across a diverse benchmark

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You can access, transform, and analyze inputs interactively in a REPL environment that can recursively query sub-LLMs. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. Variables {inputs} containing your input data. Check the content to understand what you are working with and look through it sufficiently as you answer.
2. `llm_query(prompt)` - query an LLM (handles ~500K chars) for semantic analysis. Use this especially when analyzing semantics of the data.
3. `llm_query_batched(prompts)` - query multiple prompts concurrently, returns list of responses in same order. Much faster than sequential llm_query calls for independent queries.
4. `print()` - view outputs and continue reasoning. Use variables as buffers to build up your final answer.
5. `FINAL({output_names})` - submit your final output directly, or `FINAL_VAR("variable_name")` to submit a variable's value.
6. Standard libraries: re, json, collections, math, etc.

Make sure to explicitly look through the entire input data in the REPL before answering. An example strategy is to first examine the data and figure out a chunking strategy, break it into smart chunks (by size, semantic boundaries, or structural markers), query an LLM per chunk with a particular question and save answers to a buffer, then query an LLM with all the buffers to produce your final answer.

Your sub-LLMs are powerful - they can handle around 500K characters, so don't be afraid to put a lot of context into them. A viable strategy is to feed multiple documents per sub-LLM query. Analyze your input data and see if it fits in just a few sub-LLM calls.

You have a maximum of {max_llm_calls} sub-LLM calls. For tasks with many items (e.g., counting), use Python code with regex/string matching instead of calling llm_query() per item. Reserve LLM calls for semantic understanding.

IMPORTANT: Do not submit a final answer on your first turn - explore the data first. When you are done with the iterative process, you MUST provide a final answer inside a FINAL function, NOT in code. You have two options:
1. Use FINAL({output_names}) to provide the answer directly
2. Use FINAL_VAR("variable_name") to return a variable you created in the REPL as your final output"""

# Pattern to match markdown code fences: ```python\n...\n``` or ```\n...\n```
_CODE_FENCE_PATTERN = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    match = _CODE_FENCE_PATTERN.match(code)
    if match:
        return match.group(1)
    return code


class RLM(Module):
    """Recursive Language Model module.

    Uses a sandboxed REPL to let the LLM programmatically explore large contexts
    through code execution. The LLM writes Python code to examine data, call
    sub-LLMs for semantic analysis, and build up answers iteratively.

    The default interpreter is PythonInterpreter (Deno/Pyodide/WASM), but you
    can provide any Interpreter implementation (e.g., MockInterpreter, or write a custom one using E2B or Modal).

    Example:
        ```python
        # Basic usage
        rlm = dspy.RLM("context, query -> answer", max_iterations=10)
        result = rlm(context="...very long text...", query="What is the magic number?")
        print(result.answer)
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
        interpreter: Interpreter | None = None,
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
            interpreter: Interpreter implementation to use. Defaults to PythonInterpreter.
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

    # Reserved tool names that conflict with built-in sandbox functions
    _RESERVED_TOOL_NAMES = frozenset({"llm_query", "llm_query_batched", "FINAL", "FINAL_VAR", "print"})

    def _validate_tools(self, tools: dict[str, Callable]) -> None:
        """Validate user-provided tools have valid names and are callable."""
        for name, func in tools.items():
            if not name.isidentifier():
                raise ValueError(f"Invalid tool name '{name}': must be a valid Python identifier")
            if name in self._RESERVED_TOOL_NAMES:
                raise ValueError(f"Tool name '{name}' conflicts with built-in sandbox function")
            if not callable(func):
                raise TypeError(f"Tool '{name}' must be callable, got {type(func).__name__}")

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
            return response[0] if isinstance(response, list) and response else str(response)

        def llm_query(prompt: str = "") -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt is required")
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

    def _build_signatures(self) -> tuple[Signature, Signature]:
        """Build the action and extract signatures from templates."""
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)

        # Simple names for FINAL() examples
        output_names = ", ".join(self.signature.output_fields.keys())

        # Full field descriptions with type constraints (reuses adapter logic)
        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )

        # Include original signature instructions (docstring) if present
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""

        action_sig = (
            dspy.Signature({}, task_instructions + ACTION_INSTRUCTIONS_TEMPLATE.format(
                inputs=inputs_str, output_names=output_names, output_fields=output_fields,
                max_llm_calls=self.max_llm_calls,
            ))
            .append("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=list[REPLVariable])
            .append("repl_history", dspy.InputField(desc="Previous REPL code executions and their outputs"), type_=REPLHistory)
            .append("iteration", dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"), type_=str)
            .append("reasoning", dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."), type_=str)
            .append("code", dspy.OutputField(desc="Python code to execute."), type_=str)
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
        extract_sig = extract_sig.prepend("variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=list[REPLVariable])

        return action_sig, extract_sig

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

    def _prepare_execution_tools(self) -> dict[str, Callable]:
        """Create fresh LLM tools and merge with user-provided tools."""
        execution_tools = self._make_llm_tools()
        execution_tools.update(self._user_tools)
        return execution_tools

    def _inject_execution_context(self, interpreter: Interpreter, execution_tools: dict[str, Callable]) -> None:
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
    def _interpreter_context(self, execution_tools: dict[str, Callable]) -> Iterator[Interpreter]:
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

    def _extract_fallback(
        self,
        variables: list[REPLVariable],
        history: REPLHistory,
        output_field_names: list[str],
    ) -> Prediction:
        """Use extract module to get final answer when max iterations reached."""
        logger.warning("RLM reached max iterations, using extract to get final answer")

        extract_pred = self.extract(
            variables_info=variables,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final answer",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    def _process_final_answer(
        self,
        result: FinalAnswerResult,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Validate and parse FinalAnswerResult. Returns (parsed_outputs, None) or (None, error)."""
        raw_answer = result.answer

        # Validate raw_answer is a dict
        if not isinstance(raw_answer, dict):
            return None, f"[Error] FINAL returned {type(raw_answer).__name__}, expected dict with fields: {output_field_names}"

        # Validate all required output fields are present
        missing = set(output_field_names) - set(raw_answer.keys())
        if missing:
            return None, f"[Error] Missing output fields: {sorted(missing)}. Use FINAL({', '.join(output_field_names)})"

        # Parse and validate each output field
        parsed_outputs = {}
        type_errors = []
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            try:
                parsed_outputs[name] = parse_value(raw_answer[name], annotation)
            except (ValueError, pydantic.ValidationError) as e:
                type_errors.append(
                    f"{name}: expected {annotation.__name__ if hasattr(annotation, '__name__') else annotation}, "
                    f"got {type(raw_answer[name]).__name__}: {e}"
                )

        if type_errors:
            return None, "[Type Error] " + "; ".join(type_errors)

        return parsed_outputs, None

    def _execute_iteration(
        self,
        repl: Interpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Execute one iteration. Returns Prediction if done, else updated REPLHistory."""
        action = self.generate_action(
            variables_info=variables,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )
        if self.verbose:
            logger.info(
                f"RLM iteration {iteration + 1}/{self.max_iterations}\n"
                f"Reasoning: {action.reasoning}\nCode:\n{action.code}"
            )

        try:
            # Strip markdown code fences if present
            code = _strip_code_fences(action.code)
            # Always inject input variables - ensures they're available even if sandbox restarts
            result = repl.execute(code, variables=dict(input_args))

            if isinstance(result, FinalAnswerResult):
                parsed_outputs, error = self._process_final_answer(result, output_field_names)

                if error:
                    return history.append(reasoning=action.reasoning, code=action.code, output=error)

                final_history = history.append(
                    reasoning=action.reasoning, code=action.code, output=f"FINAL: {parsed_outputs}"
                )
                return Prediction(
                    **parsed_outputs,
                    trajectory=[e.model_dump() for e in final_history],
                    final_reasoning=action.reasoning,
                )

            # Format non-final result as output
            if isinstance(result, list):
                output = "\n".join(map(str, result))
            else:
                output = str(result) if result else ""

        except (InterpreterError, SyntaxError) as e:
            output = f"[Error] {e}"

        output = self._format_output(output)
        return history.append(reasoning=action.reasoning, code=action.code, output=output)

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
            history = REPLHistory()

            for iteration in range(self.max_iterations):
                result = self._execute_iteration(
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
        logger.warning("RLM reached max iterations, using extract to get final answer")

        extract_pred = await self.extract.acall(
            variables_info=variables,
            repl_history=history,
        )

        return Prediction(
            trajectory=[e.model_dump() for e in history],
            final_reasoning="Extract forced final answer",
            **{name: getattr(extract_pred, name) for name in output_field_names},
        )

    async def _aexecute_iteration(
        self,
        repl: Interpreter,
        variables: list[REPLVariable],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> Prediction | REPLHistory:
        """Async version: Execute one iteration."""
        pred = await self.generate_action.acall(
            variables_info=variables,
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

            if isinstance(result, FinalAnswerResult):
                parsed_outputs, error = self._process_final_answer(result, output_field_names)

                if error:
                    return history.append(reasoning=pred.reasoning, code=pred.code, output=error)

                final_history = history.append(
                    reasoning=pred.reasoning, code=pred.code, output=f"FINAL: {parsed_outputs}"
                )
                return Prediction(
                    **parsed_outputs,
                    trajectory=[e.model_dump() for e in final_history],
                    final_reasoning=pred.reasoning,
                )

            if isinstance(result, list):
                output = "\n".join(map(str, result))
            else:
                output = str(result) if result else ""

        except (InterpreterError, SyntaxError) as e:
            output = f"[Error] {e}"

        output = self._format_output(output)
        return history.append(reasoning=pred.reasoning, code=pred.code, output=output)

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


# =============================================================================
# REPL Types (internal to RLM)
# =============================================================================

class REPLVariable(pydantic.BaseModel):
    """Metadata about a variable available in the REPL environment."""

    name: str
    type_name: str
    desc: str = ""
    constraints: str = ""
    total_length: int
    preview: str

    model_config = pydantic.ConfigDict(frozen=True)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        field_info: FieldInfo | None = None,
        preview_chars: int = 500,
    ) -> REPLVariable:
        """Create REPLVariable from an actual value and optional field info.

        Args:
            name: Variable name
            value: The actual value
            field_info: Optional pydantic FieldInfo with desc/constraints metadata
            preview_chars: Max characters for preview
        """
        jsonable = serialize_for_json(value)
        if isinstance(jsonable, (dict, list)):
            value_str = json.dumps(jsonable, indent=2)
        else:
            value_str = str(jsonable)
        is_truncated = len(value_str) > preview_chars
        preview = value_str[:preview_chars] + ("..." if is_truncated else "")

        # Extract desc and constraints from field_info if provided
        desc = ""
        constraints = ""
        if field_info and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            raw_desc = field_info.json_schema_extra.get("desc", "")
            # Skip placeholder descs like "${name}"
            if raw_desc and not raw_desc.startswith("${"):
                desc = raw_desc
            constraints = field_info.json_schema_extra.get("constraints", "")

        return cls(
            name=name,
            type_name=type(value).__name__,
            desc=desc,
            constraints=constraints,
            total_length=len(value_str),
            preview=preview,
        )

    def format(self) -> str:
        """Format variable metadata for prompt inclusion."""
        lines = [f"Variable: `{self.name}` (access it in your code)"]
        lines.append(f"Type: {self.type_name}")
        if self.desc:
            lines.append(f"Description: {self.desc}")
        if self.constraints:
            lines.append(f"Constraints: {self.constraints}")
        lines.append(f"Total length: {self.total_length:,} characters")
        lines.append(f"Preview:\n```\n{self.preview}\n```")
        return "\n".join(lines)

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()


class REPLEntry(pydantic.BaseModel):
    """A single REPL interaction entry containing reasoning, code, and output."""

    reasoning: str = ""
    code: str
    output: str

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self, index: int, max_output_chars: int = 5000) -> str:
        """Format this entry for inclusion in prompts."""
        output = self.output
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + f"\n... (truncated to {max_output_chars}/{len(self.output):,} chars)"
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return f"=== Step {index + 1} ===\n{reasoning_line}Code:\n```python\n{self.code}\n```\nOutput ({len(self.output):,} chars):\n{output}"


class REPLHistory(pydantic.BaseModel):
    """Container for REPL interaction history.

    Immutable: append() returns a new instance with the entry added.
    """

    entries: list[REPLEntry] = Field(default_factory=list)

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self, max_output_chars: int = 5000) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(entry.format(index=i, max_output_chars=max_output_chars) for i, entry in enumerate(self.entries))

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append(self, *, reasoning: str = "", code: str, output: str) -> REPLHistory:
        """Return a new REPLHistory with the entry appended."""
        new_entry = REPLEntry(reasoning=reasoning, code=code, output=output)
        return REPLHistory(entries=list(self.entries) + [new_entry])

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[REPLEntry]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
