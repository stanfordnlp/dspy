import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any

import dspy
from dspy.primitives.python_interpreter import InterpreterError, PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature


@dataclass
class PIPSResult:
    """Container for the outcome of a single PIPS interaction."""

    mode: str
    probability: float
    answer: str | None = None
    rationale: str | None = None
    symbols_json: str | None = None
    symbols: dict[str, Any] | None = None
    program: str | None = None
    execution_result: Any | None = None
    review_verdict: bool | None = None
    review_feedback: str | None = None
    attempts: int = 0


class ModeSelection(dspy.Signature):
    """Decide whether to solve the task with CoT or program synthesis."""

    question = dspy.InputField(
        desc="The problem to solve."
    )
    probability = dspy.OutputField(
        desc=textwrap.dedent(
            """
            A single floating point number between 0 and 1 (inclusive)
            indicating the probability that solving the problem by writing and
            executing code is more likely to reliably result in the correct
            answer than using textual chain-of-thought without code execution. 0
            means pure chain-of-thought reasoning without a code interpreter is
            clearly better and 1 means using code execution is clearly better.
            """
        ).strip()
    )


class ProgramSynthesis(dspy.Signature):
    """Generate a Python program that solves the task when given the extracted symbols."""

    question = dspy.InputField()
    trajectory = dspy.InputField()
    symbols = dspy.OutputField(
        desc=textwrap.dedent(
            """
            A JSON object with immutable problem-specific data, parameters, and
            any intermediate constants needed by the solver. The object must be
            valid JSON.
            """
        ).strip()
    )
    code = dspy.OutputField(
        desc=textwrap.dedent(
            """
            A single Python function `def solve(symbols):`
            which consumes the provided symbols dictionary and returns only the
            final output which answers the question. This must not include any
            `if __name__ == "__main__":` blocks.
            """
        ).strip()
    )


class CodeReview(dspy.Signature):
    """Evaluate the synthesized program for correctness and potential issues."""

    question = dspy.InputField()
    symbols = dspy.InputField()
    code = dspy.InputField()
    execution_result = dspy.InputField(
        desc="The output of running solve(symbols) or the runtime error message."
    )
    feedback = dspy.OutputField(
        desc="Explanation of any issues identified in the code. Focus on if it executes, if it hardcodes the answer, performs all the computation within comments and then returns a hardcoded result, or if there are any obvious bugs impacting the answer correctness."
    )
    passed = dspy.OutputField(
        desc=(
            "A boolean flag representing if the given symbols and code are "
            "acceptable or not. True means the program is acceptable (it "
            "executes, does not have a hardcoded answer, and does not have "
            "obvious bugs impacting answer correctness) while False means the "
            "code is incorrect or appears to hard-code the answer."
        ),
        type_=bool,
    )


class PIPS(dspy.Module):
    """Per-instance program synthesis agent that mirrors CodeAct's module interface."""

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        signature: str | type[Signature],
        max_iters: int = 5,
        interpreter: PythonInterpreter | None = None,
    ) -> None:
        super().__init__()
        self.signature = ensure_signature(signature)
        self.input_field_names = list(self.signature.input_fields.keys())
        if not self.input_field_names:
            raise ValueError("PIPS requires at least one input field.")

        self.threshold = self.DEFAULT_THRESHOLD
        self.max_code_attempts = max(1, max_iters)
        self.mode_selector = dspy.ChainOfThought(
            ModeSelection.with_instructions(self._build_switch_instructions(self.signature)))
        self.cot_solver = dspy.ChainOfThought(
            self.signature.with_instructions(self._build_solver_instructions(self.signature, synthesis=False)))
        self.code_generator = dspy.Predict(
            ProgramSynthesis.with_instructions(self._build_solver_instructions(self.signature, synthesis=True)))
        self.code_reviewer = dspy.Predict(
            CodeReview.with_instructions(self._build_review_instructions(self.signature)))
        self._interpreter_factory = self._build_interpreter_factory(interpreter)

    def _build_switch_instructions(self, signature: Signature) -> str:
        base = []
        if signature.instructions:
            base.append(signature.instructions.strip())

        input_names = list(signature.input_fields.keys())
        output_names = list(signature.output_fields.keys())
        inputs = ", ".join(f"`{name}`" for name in input_names) or "the provided inputs"
        outputs = ", ".join(f"`{name}`" for name in output_names) or "the requested outputs"

        base.append(
            textwrap.dedent(
                f"""
                You are the Per-Instance Program Synthesis (PIPS) agent. You will receive the fields {inputs} as input and you must decide whether
                deliberate textual reasoning or executable code is the best approach to producing {outputs}.
                """
            ).strip()
        )
        return "\n\n".join(filter(None, base)).strip()

    def _build_review_instructions(self, signature: Signature) -> str:
        base = []
        if signature.instructions:
            base.append(signature.instructions.strip())

        input_names = list(signature.input_fields.keys()) + ["symbols", "code", "execution_result"]
        output_names = ["feedback", "passed"]
        inputs = ", ".join(f"`{name}`" for name in input_names) or "the provided inputs"
        outputs = ", ".join(f"`{name}`" for name in output_names) or "the requested outputs"

        base.append(
            textwrap.dedent(
                f"""
                You are a code reviewer. You will receive the fields {inputs} as
                input and you will review the symbols, code, and execution
                result to produce {outputs}. Make sure the code does not
                directly hardcode the answer (meaning the answer is actually
                computed using code from the symbols rather than being directly
                returned) and that there are not any code bugs which will impact
                the correctness of the final returned answer. If the code does
                not have any of these issues, then the passed field should be
                True, otherwise describe the issues and set the passed field to
                False. If none of the issues impact answer correctness for the
                particular input, then the passed field should be True.
                """
            ).strip()
        )
        return "\n\n".join(filter(None, base)).strip()

    def _build_solver_instructions(self, signature: Signature, synthesis=True) -> str:
        base = []
        if signature.instructions:
            base.append(signature.instructions.strip())

        input_names = list(signature.input_fields.keys()) + ["trajectory"]
        output_names = list(signature.output_fields.keys())
        inputs = ", ".join(f"`{name}`" for name in input_names) or "the provided inputs"
        outputs = ", ".join(f"`{name}`" for name in output_names) or "the requested outputs"

        if synthesis:
            base.append(
                textwrap.dedent(
                    f"""
                    You are the Per-Instance Program Synthesis (PIPS) agent. You
                    will receive the fields {inputs} as input and you must solve
                    the task to produce {outputs} by producing a JSON symbols
                    extracted from the input and a Python code block which
                    produces {outputs} when executed with the symbols as input.
                    To solve the task, if the trajectory is empty, you must
                    start by producing JSON symbols which contain any relevant
                    information from the input and a Python function called
                    `solve` which when given the symbols as input returns
                    exactly {outputs}. You only have access to the Python
                    Standard Library. Once you produce a program, you will
                    receive the execution result as well as feedback from an
                    external code analysis in the trajectory to help with fixing
                    any issues. You should fix any code or symbols issues
                    resulting in execution or code analysis failures in order to
                    get the correct answer. Be sure that the code does not
                    hardcode the answer since it should be computed from the
                    symbols. Your objective is to solve the task, so keep going
                    until it is fully solved and you are confident in the final
                    output.
                    """
                ).strip()
            )
        else:
            base.append(
                textwrap.dedent(
                    f"""
                    You are the Per-Instance Program Synthesis (PIPS) agent. You
                    will receive the fields {inputs} as input as well as the
                    current trajectory and you must solve the task to produce
                    {outputs}. To solve the task, think step-by-step and then
                    output exactly {outputs}.
                    """
                ).strip()
            )
        return "\n\n".join(filter(None, base)).strip()

    def _build_interpreter_factory(self, interpreter: PythonInterpreter | None):
        if interpreter is None:
            return lambda: PythonInterpreter()

        prototype = interpreter
        def _maybe_list(value):
            if not value:
                return None
            return list(value)

        factory_kwargs = {
            "deno_command": list(getattr(prototype, "deno_command", []) or []) or None,
            "enable_read_paths": _maybe_list(getattr(prototype, "enable_read_paths", []) or []),
            "enable_write_paths": _maybe_list(getattr(prototype, "enable_write_paths", []) or []),
            "enable_env_vars": _maybe_list(getattr(prototype, "enable_env_vars", []) or []),
            "enable_network_access": _maybe_list(getattr(prototype, "enable_network_access", []) or []),
            "sync_files": getattr(prototype, "sync_files", True),
            "timeout": getattr(prototype, "timeout", None),
        }
        prototype.shutdown()

        def factory():
            return PythonInterpreter(**factory_kwargs)

        return factory

    def forward(self, max_iters: int | None = None, **kwargs: Any):  # type: ignore[override]
        attempt_override = max_iters if max_iters is not None else kwargs.pop("max_iters", None)
        attempt_budget = self._normalize_attempt_budget(attempt_override)

        inputs = self._collect_inputs(kwargs)
        mode_raw = self.mode_selector(**inputs)
        probability = self._parse_probability(getattr(mode_raw, "probability", "0.5"))
        if probability < self.threshold:
            cot = self.cot_solver(**inputs)
            cot_result = PIPSResult(
                mode="cot",
                probability=probability,
                answer=self._clean_str(getattr(cot, "answer", "")),
                rationale=self._clean_str(getattr(cot, "reasoning", "")),
            )
            return dspy.Prediction(pips_result=cot_result, **self._project_outputs(cot_result))

        synthesis = self._run_program_synthesis(inputs=inputs, attempt_budget=attempt_budget)
        synthesis.probability = probability
        outputs = self._project_outputs(synthesis)
        outputs["pips_result"] = synthesis
        return dspy.Prediction(**outputs)

    def _project_outputs(self, result: PIPSResult) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for name in self.signature.output_fields.keys():
            outputs[name] = getattr(result, name, None)
        if not outputs:
            outputs["result"] = result.execution_result
        return outputs

    def _collect_inputs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for name in self.input_field_names:
            if name not in kwargs:
                raise ValueError(f"Missing required input field `{name}` for PIPS signature.")
            inputs[name] = kwargs[name]
        return inputs

    def _run_program_synthesis(self, inputs: dict[str, Any], attempt_budget: int) -> PIPSResult:
        feedback: str | None = None
        symbols_json: str | None = None
        symbols: dict[str, Any] | None = None
        program: str | None = None
        execution_result: Any = None
        passed: bool | None = None
        reviewer_feedback: str | None = None

        trajectory = {}
        for attempt in range(1, attempt_budget + 1):
            out = self.code_generator(trajectory=self._format_trajectory(trajectory), **inputs)
            symbols_json = self._clean_str(getattr(out, "symbols", ""))
            symbols, parse_feedback = self._parse_symbols(symbols_json)
            trajectory[f"symbols_{attempt}"] = str(symbols)
            if parse_feedback:
                feedback = parse_feedback
                symbols = None
                trajectory[f"feedback_{attempt}"] = feedback
                continue

            program = self._clean_str(getattr(out, "code", ""))
            trajectory[f"code_{attempt}"] = str(program)
            execution_result, runtime_feedback = self._execute_program(program, symbols)
            trajectory[f"execution_result_{attempt}"] = self._format_for_review(execution_result, runtime_feedback)
            review = self.code_reviewer(
                symbols=symbols_json or "{}",
                code=program,
                execution_result=self._format_for_review(execution_result, runtime_feedback),
                **inputs
            )
            passed = getattr(review, "passed", False)
            reviewer_feedback = self._clean_str(getattr(review, "feedback", ""))
            if runtime_feedback:
                passed = False
                reviewer_feedback = reviewer_feedback or runtime_feedback
            feedback = reviewer_feedback if not passed else None
            trajectory[f"feedback_{attempt}"] = feedback

            if passed:
                return PIPSResult(
                    mode="code",
                    probability=self.threshold,
                    symbols_json=symbols_json,
                    symbols=symbols,
                    program=program,
                    execution_result=execution_result,
                    review_verdict=passed,
                    review_feedback=reviewer_feedback,
                    attempts=attempt,
                    answer=self._stringify_answer(execution_result) if execution_result is not None else None,
                )

        return PIPSResult(
            mode="code",
            probability=self.threshold,
            symbols_json=symbols_json,
            symbols=symbols,
            program=program,
            execution_result=execution_result,
            review_verdict=passed,
            review_feedback=reviewer_feedback,
            attempts=attempt_budget,
            answer=self._stringify_answer(execution_result) if execution_result is not None else None,
        )

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def _parse_probability(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        match = re.search(r"0?\.\d+|1(?:\.0+)?|0", str(value))
        probability = float(match.group()) if match else self.threshold
        return max(0.0, min(1.0, probability))

    def _parse_symbols(self, payload: str | None) -> tuple[dict[str, Any] | None, str | None]:
        if not payload:
            return None, "Symbol extraction returned an empty result. Provide valid JSON."
        try:
            return json.loads(payload), None
        except json.JSONDecodeError as exc:
            message = (
                "Symbols must be valid JSON. "
                f"Encountered JSONDecodeError: {exc.msg} at line {exc.lineno}, column {exc.colno}."
            )
            return None, message

    def _execute_program(
        self, program: str | None, symbols: dict[str, Any] | None
    ) -> tuple[Any, str | None]:
        if not program:
            return None, "No program was generated."
        if "def solve" not in program:
            return None, "Program must define a `solve(symbols)` function."
        if symbols is None:
            return None, "Symbols are unavailable; regenerate them before running code."

        execution_suffix = textwrap.dedent(
            """
            _pips_result = solve(symbols)
            final_answer(_pips_result)
            """
        )
        execution_code = f"{program}\n\n{execution_suffix}"
        interpreter = self._interpreter_factory()
        try:
            result = interpreter(
                execution_code,
                variables={"symbols": symbols},
            )
            return result, None
        except (InterpreterError, SyntaxError) as exc:
            return None, f"{exc.__class__.__name__}: {exc}"
        except Exception as exc:
            return None, f"{exc.__class__.__name__}: {exc}"
        finally:
            interpreter.shutdown()

    def _normalize_attempt_budget(self, per_call_attempts: Any) -> int:
        if per_call_attempts is None:
            return self.max_code_attempts
        try:
            value = int(per_call_attempts)
        except (TypeError, ValueError):
            return self.max_code_attempts
        return max(1, value)

    def _format_for_review(self, result: Any, error: str | None) -> str:
        if error:
            return json.dumps({"error": error})
        try:
            return json.dumps(result)
        except TypeError:
            return json.dumps({"value": self._stringify_answer(result)})

    @staticmethod
    def _clean_str(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _stringify_answer(result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result)
        return str(result)
