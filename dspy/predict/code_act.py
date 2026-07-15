import inspect
import logging
from typing import Callable

import dspy
from dspy.adapters.types.tool import Tool
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct
from dspy.primitives.code_interpreter import CodeInterpreter, _validate_interpreter_factory
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

class CodeAct(ReAct, ProgramOfThought):
    """
    CodeAct is a module that utilizes the Code Interpreter and predefined tools to solve the problem.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        tools: list[Callable],
        max_iters: int = 5,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    ):
        """
        Initializes the CodeAct class with the specified model, temperature, and max tokens.

        Args:
            signature (Union[str, Type[Signature]]): The signature of the module.
            tools (list[Callable]): The tool callables to be used. CodeAct only accepts functions and not callable objects.
            max_iters (int): The maximum number of iterations to generate the answer.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns.
        Examples:
            ```python
            from dspy.predict import CodeAct
            def factorial(n):
                if n == 1:
                    return 1
                return n * factorial(n-1)

            act = CodeAct("n->factorial", tools=[factorial])
            act(n=5) # 120
            ```
        """
        _validate_interpreter_factory(interpreter_factory)
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.history = []

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        if any(
            not inspect.isfunction(tool.func) for tool in tools
        ):
            raise ValueError("CodeAct only accepts functions and not callable objects.")
        tools = {tool.name: tool for tool in tools}

        instructions = self._build_instructions(self.signature, tools)

        codeact_signature = (
            dspy.Signature({**self.signature.input_fields}, "\n".join(instructions))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("generated_code", dspy.OutputField(desc="Python code that when executed, produces output relevant to answering the question"), type_=str)
            .append("finished", dspy.OutputField(desc="a boolean flag to determine if the process is done"), type_=bool)
        )

        extract_signature = dspy.Signature(
            {**self.signature.input_fields, **self.signature.output_fields},
            self.signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools: dict[str, Tool] = tools
        self.codeact = dspy.Predict(codeact_signature)
        self.extractor = dspy.ChainOfThought(extract_signature)
        self._interpreter_factory = interpreter_factory

    def _build_instructions(self, signature, tools):
        instructions = [f"{signature.instructions}\n"] if signature.instructions else []
        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])

        instructions.append(
            f"You are an intelligent agent. For each episode, you will receive the fields {inputs} as input.\n"
            f"Your goal is to generate executable Python code that collects any necessary information for producing {outputs}.\n"
            "For each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.\n"
            "Ensure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.\n"
            f"When all information for producing the outputs ({outputs}) are available to be extracted, mark `finished=True` besides the final Python code.\n"
            "You have access to the Python Standard Library and the following functions:"
        )

        for idx, tool in enumerate(tools.values()):
            instructions.append(f"({idx + 1}) {tool}")

        return instructions

    def forward(self, interpreter: CodeInterpreter | None = None, /, **kwargs):
        """Run the program with a fresh interpreter or a caller-owned override.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. The caller must shut it down.
            **kwargs: Input values matching the signature's input fields.

        Raises:
            CodeInterpreterError: If interpreter setup, process, or protocol fails.
        """
        if "interpreter" in kwargs and "interpreter" not in self.signature.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument when calling the module."
            )
        with self._interpreter_context(interpreter) as interpreter:
            # Define the tool functions in the interpreter
            for tool in self.tools.values():
                interpreter.execute(inspect.getsource(tool.func))

            trajectory = {}
            max_iters = kwargs.pop("max_iters", self.max_iters)
            for idx in range(max_iters):
                try:
                    code_data = self.codeact(trajectory=trajectory, **kwargs)
                except AdapterParseError as err:
                    # Same failure class as dspy.ReAct (#8377): record the parse
                    # failure as an observation and let the model self-correct.
                    logger.warning(f"Failed to parse the LM response for the next step: {err}")
                    trajectory[f"observation_{idx}"] = self._format_parse_failure_observation(err)
                    continue

                output = None
                code, error = self._parse_code(code_data)

                if error:
                    trajectory[f"observation_{idx}"] = f"Failed to parse the generated code: {error}"
                    continue

                trajectory[f"generated_code_{idx}"] = code
                output, error = self._execute_code(code, interpreter)

                if not error:
                    trajectory[f"code_output_{idx}"] = output
                else:
                    trajectory[f"observation_{idx}"] = f"Failed to execute the generated code: {error}"

                if code_data.finished:
                    break

            extract = self._call_extract_with_parse_retry(self.extractor, trajectory, **kwargs)
            return dspy.Prediction(trajectory=trajectory, **extract)

    def truncate_trajectory(self, trajectory):
        """Truncate the oldest CodeAct iteration so the trajectory fits the context window.

        CodeAct steps have a variable number of keys (1 on a parse/execution failure:
        ``observation_i``; 2 on success: ``generated_code_i`` + ``code_output_i``), unlike
        ReAct's fixed 4 keys per step. Popping a fixed ``keys[:4]`` slice (ReAct's behavior)
        would cut across iteration boundaries and desynchronize the trajectory, so instead we
        drop every key belonging to the earliest iteration index.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if not keys:
            raise ValueError(
                "The trajectory is empty, so it cannot be truncated to fit the context window."
            )

        iteration_indices = {int(key.rsplit("_", 1)[-1]) for key in keys}
        if len(iteration_indices) < 2:
            # Only one iteration is present; dropping it would leave no context (same spirit as
            # ReAct's single-tool-call guard).
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but it "
                "cannot be truncated because it only contains a single iteration."
            )

        oldest = min(iteration_indices)
        for key in keys:
            if int(key.rsplit("_", 1)[-1]) == oldest:
                trajectory.pop(key)

        return trajectory
