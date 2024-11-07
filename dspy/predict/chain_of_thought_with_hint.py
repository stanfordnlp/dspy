import dspy

from .predict import Module

# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

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
            pred = self.module(**kwargs)
            this_trace = dspy.settings.trace[-1]
            dspy.settings.trace[-1] = (this_trace[0], original_kwargs, this_trace[2])
            return pred
        
        return self.module(**kwargs)


"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""