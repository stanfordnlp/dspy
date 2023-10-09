import dsp
import dspy
from ..primitives.program import Module
from ..primitives.interpreter import CodePrompt, PythonInterpreter
import re

class ProgramOfThought(Module):
    def __init__(self, signature, max_iters=3):
        super().__init__()

        self.signature = signature = dspy.Predict(signature).signature
        self.max_iters = max_iters

        self.input_fields = signature.input_fields()
        self.output_fields = signature.output_fields()

        inputs_ = ', '.join([f"`{field_name}`" for field_name in self.input_fields.keys()])
        outputs_ = ', '.join([f"`{field_name}`" for field_name in self.output_fields.keys()])

        assert len(self.output_fields) == 1, "PoT only supports one output field."
        
        instr = []
        instr.append(f"You will be given {inputs_} and you will respond with {outputs_}.")
        instr.append(f"Generating executable Python code that programmatically computes the correct {outputs_}.")
        instr.append(f"After you're done with the computation, make sure the last line in your code evaluates to the correct value for {outputs_}.")

        self.code_generate = dspy.ChainOfThought(dsp.Template(instr, **self._generate_signature('generate')))

        self.code_generate = dspy.ChainOfThought(dsp.Template(instr, **self._generate_signature('generate')))
        self.code_regenerate = dspy.ChainOfThought(dsp.Template(instr, **self._generate_signature('regenerate')))
        self.generate_answer = dspy.ChainOfThought(dsp.Template(instr, **self._generate_signature('answer')))

    def _generate_signature(self, mode):
        signature_dict = {}
        if mode == 'generate':
            signature_dict['question'] = self.signature.kwargs["question"]
            signature_dict['code_output'] = dspy.OutputField(prefix="Code:", desc="python code that answers the question")
        elif mode == 'regenerate':
            signature_dict.update(self.input_fields)
            signature_dict['generated_code'] = dspy.OutputField(prefix="Code:", desc="python code that answers the question")
        else:
            signature_dict.update(self.input_fields)
            signature_dict['answer'] = self.signature.kwargs["answer"]
        return signature_dict

    def execute_code(self, code):
        code_match = re.search(r'```python\n(.*?)\n```', code.get('code_output', ''), re.DOTALL)
        if not code_match:
            return None, None, "Error in code matching"
        code_prompt = CodePrompt(code_match.group(1), code_type="python")
        interpreter = PythonInterpreter(action_space={"print": print})
        try:
            output = str(code_prompt.execute(interpreter=interpreter)[0])
            return code_match.group(1), output, None
        except Exception as e:
            return code_match.group(1), None, str(e)

    def forward(self, **kwargs):
        code_data = self.code_generate(question=kwargs["question"])
        code, output, error = self.execute_code(code_data)
        hop = 0
        while hop < self.max_iters and error:
            print('Error in code execution')
            code_data = self.code_regenerate(question=kwargs["question"], previous_code=code, error=error)
            code, output, error = self.execute_code(code_data)
            hop += 1
            if hop == self.max_iters:
                print('Max hops reached. Error persists.')
                return None
        answer_gen_result = self.generate_answer(question=kwargs["question"], final_generated_code=code, code_output=output)
        return answer_gen_result