import logging
import re
from typing import Union, Type

import dspy
from dspy.signatures.signature import ensure_signature, Signature

from dspy.primitives.program import Module
from dspy.primitives.python_interpreter import PythonInterpreter

logger = logging.getLogger(__name__)

class ProgramOfThought(Module):
    """
    A DSPy module that runs Python programs to solve a problem.
    This module reuires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/

    Example:
    ```
    import dspy

    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    pot = dspy.ProgramOfThought("question -> answer")
    pot(question="what is 1+1?")
    ```
    """

    def __init__(self, signature: Union[str, Type[Signature]], max_iters=3):
        """
        Args:
            signature: The signature of the module.
            max_iters: The maximum number of iterations to retry code generation and execution.
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.input_fields = signature.input_fields
        self.output_fields = signature.output_fields

        assert len(self.output_fields) == 1, "PoT only supports one output field."

        self.output_field_name = next(iter(self.output_fields))
        inputs_ = ", ".join(
            [f"`{field_name}`" for field_name in self.input_fields.keys()],
        )
        outputs_ = f"`{self.output_field_name}`"

        assert len(self.output_fields) == 1, "PoT only supports one output field."

        instr = []
        instr.append(
            f"You will be given {inputs_} and you will respond with {outputs_}.",
        )
        instr.append(
            f"Generating executable Python code that programmatically computes the correct {outputs_}.",
        )
        instr.append(
            f"After you're done with the computation, make sure the last line in your code evaluates to the correct value for {outputs_}.",
        )
        instr = "\n".join(instr)

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
        self.generate_answer = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("answer").fields,
                self._generate_instruction("answer"),
            ),
        )
        # Currently, the interpreter class checks the deno availability at execution time. 
        # We may consider checking it at the initialization time for better instruction.
        self.interpreter = PythonInterpreter()

    def _generate_signature(self, mode):
        signature_dict = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": dspy.OutputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
            },
            "regenerate": {
                "previous_code": dspy.InputField(
                    prefix="Previous Code:",
                    desc="previously-generated python code that errored",
                    format=str,
                ),
                "error": dspy.InputField(
                    prefix="Error:",
                    desc="error message from previously-generated python code",
                ),
                "generated_code": dspy.OutputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
            },
            "answer": {
                "final_generated_code": dspy.InputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
                "code_output": dspy.InputField(
                    prefix="Code Output:",
                    desc="output of previously-generated python code",
                ),
                self.output_field_name: self.signature.fields[self.output_field_name],
            },
        }
        signature_dict.update(fields_for_mode[mode])
        return dspy.Signature(signature_dict)

    def _generate_instruction(self, mode):
        mode_inputs = ", ".join(
            [
                f"`{field_name}`"
                for field_name in self._generate_signature(mode).input_fields
            ],
        )
        mode_outputs = f"`{self.output_field_name}`"
        if mode == "generate":
            instr = [
                f"You will be given {mode_inputs} and you will respond with {mode_outputs}.",
                f"Generating executable Python code that programmatically computes the correct {mode_outputs}.",
                f"After you're done with the computation, make sure the last line in your code evaluates to the correct value for {mode_outputs}.",
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
        code = (
            code_data.get("generated_code", "").split("---", 1)[0].split("\n\n\n", 1)[0]
        )
        code_match = re.search(r"```python[ \n](.*?)[ \n]```?", code, re.DOTALL)
        code_block = (code_match.group(1) if code_match else code).replace("\\n", "\n")
        if not code_block:
            return code, "Error: Empty code after parsing."
        if "\n" not in code_block and code_block.count("=") > 1:
            return code, "Error: Code format is not correct."
        lines = code_block.split("\n")
        last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
        if last_line_match and len(lines) > 1:
            code_block += "\n" + last_line_match.group(1)
        else:
            code_block = re.sub(
                r"([a-zA-Z_]\w* *=.*?)(?=[a-zA-Z_]\w* *=)", r"\1\n", code_block,
            )
            code_block = re.sub(
                r"([a-zA-Z_]\w* *=.*?)([a-zA-Z_]\w*)$", r"\1\n\2", code_block,
            )
        return code_block, None

    def _execute_code(self, code):
        """
        Execute the code using PythonInterpreter and return the output or error.
        """
        if not code:
            return None, "Error: Empty code before execution."
        
        try:
            output = str(self.interpreter.execute(code))
            return output, None
        except Exception as e:
            return None, str(e)

    def forward(self, **kwargs):
        input_kwargs = {
            field_name: kwargs[field_name] for field_name in self.input_fields
        }
        code_data = self.code_generate(**input_kwargs)
        output = None
        code, error = self._parse_code(code_data)
        if not error:
            output, error = self._execute_code(code)
        hop = 1
        # Retying code generation and execution until no error or reach max_iters
        while error is not None:
            logger.error(f"Error in code execution: {error}")
            if hop == self.max_iters:
                self.interpreter.shutdown()
                raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {error}")
            input_kwargs.update({"previous_code": code, "error": error})
            code_data = self.code_regenerate(**input_kwargs)
            code, error = self._parse_code(code_data)
            if not error:
                output, error = self._execute_code(code)
            hop += 1
        input_kwargs.update({"final_generated_code": code, "code_output": output})
        answer_gen_result = self.generate_answer(**input_kwargs)
        self.interpreter.shutdown()
        return answer_gen_result
