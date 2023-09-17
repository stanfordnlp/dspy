import dsp
import dspy
from ..primitives.program import Module
from .predict import Predict

class ReAct(Module):
    def __init__(self, signature, max_iters=5, num_results=3):
        self.signature = signature
        self.max_iters = max_iters
        self.retrieve = dspy.Retrieve(k=num_results)
        self.predictors = [Predict(dsp.Template(self.signature.instructions, **self._generate_signature(i))) for i in range(1, max_iters + 1)]

    def _generate_signature(self, iters):
        signature_dict = {"question": self.signature.kwargs["question"]}
        for j in range(1, iters + 1):
            signature_dict[f"Thought_{j}"] = dspy.OutputField(prefix=f"Thought {j}:", desc="next steps to take based on last observation in history")
            signature_dict[f"Action_{j}"] = dspy.OutputField(prefix=f"Action {j}:", desc="Search: prefix if querying based on question or thought or Finish: prefix when found answer")
            if j < iters:
                signature_dict[f"Observation_{j}"] = dspy.OutputField(prefix=f"Observation {j}:", desc="observations based on action")
        return signature_dict

    def forward(self, **kwargs):
        output = type('', (), {})()
        for i in range(self.max_iters):
            output = self.predictors[i](question=kwargs["question"], **vars(output))
            if 'Finish:' in getattr(output, f"Action_{i+1}"):
                answer = getattr(output, f"Action_{i+1}").split('Finish: ')[1]
                output = dspy.Prediction() 
                output.answer = answer
                break
            output[f"Observation_{i+1}"] = self.retrieve(getattr(output, f"Action_{i+1}").split(':')[1])
        return output