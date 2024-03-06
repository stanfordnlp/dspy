import dsp
import dspy

from .predict import Predict

# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

class ChainOfThoughtWithHint(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)
        self.activated = activated
        signature = self.signature

        *keys, last_key = signature.fields.keys()
        rationale_type = rationale_type or dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + last_key + "}. We ...",
        )
        self.extended_signature1 = self.signature.insert(-2, "rationale", rationale_type, type_=str)

        DEFAULT_HINT_TYPE = dspy.OutputField()
        self.extended_signature2 = self.extended_signature1.insert(-2, "hint", DEFAULT_HINT_TYPE, type_=str)
    
    def forward(self, **kwargs):
        signature = self.signature

        if self.activated is True or (self.activated is None and isinstance(dsp.settings.lm, dsp.GPT3)):
            if 'hint' in kwargs and kwargs['hint']:
                signature = self.extended_signature2
            else:
                signature = self.extended_signature1
        
        return super().forward(signature=signature, **kwargs)


"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""