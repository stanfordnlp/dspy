import dspy
from .predict import Module

class ChainOfThoughtWithHint(Module):
    def __init__(self, signature, rationale_type=None, **config):
        self.signature = dspy.ensure_signature(signature)
        self.module = dspy.ChainOfThought(signature, rationale_type=rationale_type, **config)
    
    def forward(self, **kwargs):
        if 'hint' in kwargs and kwargs['hint']:
            hint = f"\n\t\t(secret hint: {kwargs.pop('hint')})"
            original_kwargs = kwargs.copy()
            
            # Convert the first field's value to string and append the hint
            last_key = list(self.signature.input_fields.keys())[-1]
            kwargs[last_key] = str(kwargs[last_key]) + hint

            # Run CoT then update the trace with original kwargs, i.e. without the hint.
            with dspy.context(trace=[]):
                pred = self.module(**kwargs)
                this_trace = dspy.settings.trace[-1]

            dspy.settings.trace.append((this_trace[0], original_kwargs, this_trace[2]))
            return pred
        
        return self.module(**kwargs)
