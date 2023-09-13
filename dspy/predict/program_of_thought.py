import dsp
import dspy
from .predict import Predict
from ..primitives.interpreter import CodePrompt, PythonInterpreter


class ProgramOfThought(Predict):
    def __init__(self, signature, max_retries=3, activated=True, **config):
        super().__init__(signature, **config)
        self.max_retries = max_retries

    def execute_code(self, code):
        code_prompt = CodePrompt(code, code_type="python")
        interpreter = PythonInterpreter(action_space={"print": print})
        output, error = None, None
        try:
            output = str(code_prompt.execute(interpreter=interpreter)[0])
        except Exception as e:
            error = str(e)
        return output, error

    def forward(self, **kwargs):
        question = kwargs['question']
        signature = self.signature
        answer_field = signature.kwargs.get('answer')
        if 'answer' in signature.kwargs:
            del signature.kwargs['answer']
        signature.kwargs['code'] = dspy.OutputField(prefix="Code:", desc="python code that answers the question")
        code_gen_result = dspy.ChainOfThought(signature).forward(question=question)
        code = code_gen_result.get('code', '').replace("```", "").strip().split('Output')[0].strip()
        output, error = self.execute_code(code)
        retries = 0
        while retries < self.max_retries and error:
            print('Error in code execution')
            *keys, second_last_key, last_key = signature.kwargs.keys()
            prev_code = code
            if 'code' in signature.kwargs:
                del signature.kwargs['code']
            prev_code_field = dspy.InputField(prefix="Previous Code:", desc="previously-generated python code that errored")
            error_field = dspy.InputField(prefix="Error:", desc="error message from previously-generated python code")
            code_field = dspy.OutputField(prefix="Code:", desc="python code that answers the question")
            extended_kwargs = {key: signature.kwargs[key] for key in keys}
            extended_kwargs.update({second_last_key: signature.kwargs[second_last_key], 'previous code': prev_code_field, 'error': error_field, 'code': code_field})
            signature = dsp.Template(signature.instructions, **extended_kwargs)
            code_result = dspy.ChainOfThought(signature).forward(question=question, prev_code=prev_code, error=error)
            code = code_result.get('code', '').replace("```", "").strip().replace("python", "").strip().split('Output')[0].strip().lstrip().split('Code:')[1].strip()
            output, error = self.execute_code(code)
            retries += 1
            if retries == self.max_retries:
                print('Max retries reached. Error persists.')
                return None
        if 'previous code' in signature.kwargs:
            del signature.kwargs['previous code']
        if 'error' in signature.kwargs:
            del signature.kwargs['error']
        code_field = dspy.InputField(prefix="Code:", desc="previously-generated python code that was executed without errors")
        output_field = dspy.InputField(prefix="Output:", desc="output of previously-generated python code")
        *keys, _ = signature.kwargs.keys()
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update({'code': code_field, 'output': output_field, 'answer': answer_field})
        signature = dsp.Template(signature.instructions, **extended_kwargs)
        answer_gen_result = dspy.ChainOfThought(signature).forward(question=question, code=code, output=output)
        return answer_gen_result
        