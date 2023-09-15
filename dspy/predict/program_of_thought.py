import dsp
import dspy
from ..primitives.program import Module
from ..primitives.interpreter import CodePrompt, PythonInterpreter
import re

class ProgramOfThought(Module):
    def __init__(self, signature, max_retries=3):
        previous_code = dspy.InputField(prefix="Previous Code:", desc="previously-generated python code that errored")
        error = dspy.InputField(prefix="Error:", desc="error message from previously-generated python code")
        generated_code = dspy.OutputField(prefix="Code:", desc="python code that answers the question") 
        final_generated_code = dspy.InputField(prefix="Code:", desc="python code that answers the question") 
        code_output = dspy.InputField(prefix="Code Output:", desc="output of previously-generated python code")
        code_generate = {"question": signature.kwargs["question"], "code_output": generated_code}
        code_regenerate = {"question": signature.kwargs["question"],"previous_code": previous_code,"error": error,"generated_code": generated_code}
        generate_answer = {"question": signature.kwargs["question"],"final_generated_code": final_generated_code,"code_output": code_output,"answer": signature.kwargs["answer"]}
        self.code_generate = dspy.ChainOfThought(dsp.Template(signature.instructions, **code_generate))
        self.code_regenerate = dspy.ChainOfThought(dsp.Template(signature.instructions, **code_regenerate))
        self.generate_answer = dspy.ChainOfThought(dsp.Template(signature.instructions, **generate_answer))
        self.max_retries = max_retries

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
        code = self.code_generate(question=kwargs["question"])
        code, output, error = self.execute_code(code)
        retries = 0
        while retries < self.max_retries and error:
            print('Error in code execution')
            code = self.code_regenerate(question=kwargs["question"], previous_code=prev_code, error=error)
            code, output, error = self.execute_code(code)
            retries += 1
            if retries == self.max_retries:
                print('Max retries reached. Error persists.')
                return None
        answer_gen_result = self.generate_answer(question=kwargs["question"], final_generated_code=code, code_output=output)
        return answer_gen_result
        