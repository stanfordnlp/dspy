import dsp
import dspy
from ..primitives.program import Module
from .predict import Predict

class ReAct(Module):
    def __init__(self, signature, max_iters=5, num_results=3, tools=None):
        super().__init__()
        self.signature = signature
        self.input_fields = {k: v for k, v in self.signature.kwargs.items() if isinstance(v, dspy.InputField)}
        self.output_fields = {k: v for k, v in self.signature.kwargs.items() if isinstance(v, dspy.OutputField)}
        self.max_iters = max_iters
        self.tools = tools or {"search": dspy.Retrieve(k=num_results)}
        self.predictors = [Predict(dsp.Template(self.signature.instructions, **self._generate_signature(i))) for i in range(1, max_iters + 1)]

    def _generate_signature(self, iters):
        signature_dict = {}
        for key, val in self.input_fields.items():
            signature_dict[key] = val
        for j in range(1, iters + 1):
            signature_dict[f"Thought_{j}"] = dspy.OutputField(prefix=f"Thought {j}:", desc="next steps to take based on last observation in history")
            signature_dict[f"Action_{j}"] = dspy.OutputField(prefix=f"Action {j}:", desc="always either Search[query] if querying or Finish[answer] when found answer")
            if j < iters:
                signature_dict[f"Observation_{j}"] = dspy.OutputField(prefix=f"Observation {j}:", desc="observations based on action")
        return signature_dict

    def forward(self, **kwargs):
        output = {}
        args= {key: kwargs[key] for key in self.input_fields.keys() if key in kwargs}
        for i in range(self.max_iters):
            args.update(output)
            output = self.predictors[i](**args)
            action_val = output[f"Action_{i+1}"].split('[')[1].split(']')[0]
            if 'Finish[' in output[f"Action_{i+1}"]:
                break
            output[f"Observation_{i+1}"] = self.tools["search"](action_val.split('\n')[0]).passages[0]
        return dspy.Prediction(**{list(self.output_fields.keys())[0]: action_val}) #assumes only 1 output field for now - TODO: handling for multiple output fields