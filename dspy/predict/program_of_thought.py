import json
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import dspy
from dspy.primitives.code_interpreter import (
    CodeExecutionError,
    CodeInterpreter,
    FinalOutput,
    _create_interpreter,
    _validate_interpreter,
    _validate_interpreter_factory,
)
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)


class ProgramOfThought(Module):
    """
    A DSPy module that runs Python programs to solve a problem.
    This module requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/

    Examples:
    ```
    import dspy

    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    pot = dspy.ProgramOfThought("question -> answer")
    pot(question="what is 1+1?")
    ```
    """

    def __init__(
        self,
        signature: str | type[Signature],
        max_iters: int = 3,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
    ):
        """
        Args:
            signature: The signature of the module.
            max_iters: The maximum number of iterations to retry code generation and execution.
            interpreter_factory: Zero-argument callable that creates an interpreter for each forward pass. The
                callable may be invoked concurrently, and DSPy shuts down each interpreter it returns.
        """
        super().__init__()
        _validate_interpreter_factory(interpreter_factory)
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.input_fields = signature.input_fields
        self.output_fields = signature.output_fields

        self.code_generate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("generate").fields,
                self._generate_instruction("generate"),
            ),
        )
        self.code_regenerate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("regenerate").fields,
                self._generate_instruction("regenerate"),
            ),
        )
        self.generate_output = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("answer").fields,
                self._generate_instruction("answer"),
            ),
        )
        self._interpreter_factory = interpreter_factory

    @contextmanager
    def _interpreter_context(self, interpreter: CodeInterpreter | None) -> Iterator[CodeInterpreter]:
        """Yield a caller-owned interpreter or manage a factory-created one."""
        if interpreter is not None:
            _validate_interpreter(interpreter)
            yield interpreter
            return

        interpreter = _create_interpreter(self._interpreter_factory)
        try:
            yield interpreter
        finally:
            interpreter.shutdown()

    def _generate_signature(self, mode):
        signature_dict = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": dspy.OutputField(
                    desc="python code that answers the question",
                ),
            },
            "regenerate": {
                "previous_code": dspy.InputField(
                    desc="previously-generated python code that errored",
                ),
                "error": dspy.InputField(
                    desc="error message from previously-generated python code",
                ),
                "generated_code": dspy.OutputField(
                    desc="python code that answers the question",
                ),
            },
            "answer": {
                "final_generated_code": dspy.InputField(
                    desc="python code that answers the question",
                ),
                "code_output": dspy.InputField(
                    desc="output of previously-generated python code",
                ),
            }
            | self.signature.output_fields,
        }
        signature_dict.update(fields_for_mode[mode])
        return dspy.Signature(signature_dict)

    def _generate_instruction(self, mode):
        mode_inputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).input_fields],
        )
        mode_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).output_fields],
        )
        final_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self.output_fields],
        )
        if mode == "generate":
            instr = [
                f"You will be given {mode_inputs} and you will respond with {mode_outputs}.",
                f"Generating executable Python code that programmatically computes the correct {mode_outputs}.",
                "After you're done with the computation and think you have the final output, make sure to submit your output by calling the preloaded function `SUBMIT()`.",
                f'You must structure your output in a dict, like {{"field_a": value_a, ...}}, with the correct value mapping for the field(s): {final_outputs}.',
            ]
        elif mode == "regenerate":
            instr = [
                f"You are given {mode_inputs} due to an error in previous code.",
                "Your task is to correct the error and provide the new `generated_code`.",
            ]
        else:  # mode == 'answer'
            instr = [
                f"Given the final code {mode_inputs}, provide the final {mode_outputs}.",
            ]

        return "\n".join(instr)

    def _parse_code(self, code_data):
        code = code_data.get("generated_code", "").split("---", 1)[0].split("\n\n\n", 1)[0]
        code_match = re.search(r"```python[ \n](.*?)[ \n]```?", code, re.DOTALL)
        code_block = code_match.group(1) if code_match else code
        if not code_block:
            return code, "Error: Empty code after parsing."
        if "\n" not in code_block and code_block.count("=") > 1:
            return code, "Error: Code format is not correct."
        lines = code_block.split("\n")
        last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
        if last_line_match and len(lines) > 1:
            code_block += "\n" + last_line_match.group(1)
        return code_block, None

    def _execute_code(self, code: str, interpreter: CodeInterpreter):
        """
        Execute the code using the current interpreter and return the output or error.
        """
        if not code:
            return None, "Error: Empty code before execution."

        try:
            result = interpreter.execute(code)
            if isinstance(result, FinalOutput):
                result = result.output
            # Since it's more complex structure now, just blindly use json to represents all.
            output = json.dumps(result)
            return output, None
        except (CodeExecutionError, SyntaxError) as e:
            return None, str(e)

    def forward(self, interpreter: CodeInterpreter | None = None, /, **kwargs):
        """Run the program with a fresh interpreter or a caller-owned override.

        Args:
            interpreter: Optional caller-owned interpreter, passed positionally. The caller must shut it down.
            **kwargs: Input values matching the signature's input fields.
        """
        if "interpreter" in kwargs and "interpreter" not in self.input_fields:
            raise TypeError(
                "To use a caller-owned interpreter, pass it as the first positional argument to forward(...)."
            )
        with self._interpreter_context(interpreter) as interpreter:
            input_kwargs = {field_name: kwargs[field_name] for field_name in self.input_fields}
            code_data = self.code_generate(**input_kwargs)
            output = None
            code, error = self._parse_code(code_data)
            if not error:
                output, error = self._execute_code(code, interpreter)
            hop = 1
            # Retrying code generation and execution until no error or reach max_iters
            while error is not None:
                logger.error(f"Error in code execution: {error}")
                if hop == self.max_iters:
                    raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {error}")
                input_kwargs.update({"previous_code": code, "error": error})
                code_data = self.code_regenerate(**input_kwargs)
                code, error = self._parse_code(code_data)
                if not error:
                    output, error = self._execute_code(code, interpreter)
                hop += 1
            input_kwargs.update({"final_generated_code": code, "code_output": output})
            return self.generate_output(**input_kwargs)
