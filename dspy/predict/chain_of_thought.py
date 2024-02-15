import dsp, dspy
from dspy.signatures.signature import ensure_signature

from .predict import Predict, signature_to_template


# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

# TODO: This shouldn't inherit from Predict. It should be a module that has one or two predictors.
# Let's focus on the activated case. It's a predictor with the expanded signature.
# Now, when deactivated, it's a predictor with the original signature.
# When activate is None, though, we need the expanded one but during forward we need to pass the right signature.

"""
class ChainOfThought(dspy.Module):
    def __init__(self, signature):

        input_fields, output_fields = dspy.process_signature(signature)
        output_fields = dict(rationale=dspy.OutputField(prefix="Reasoning: Let's think step by step."), **output_fields)
        self.signature = dspy.Signature(input_fields, output_fields)
        
        self.predict = dspy.Predict(self.signature)
    
    def forward(self, **kwargs):
        return self.predict(**kwargs)

# How this should look like. But with also passing signature=simpler_signature to the predict module *if* deactivated.
"""


class ChainOfThought(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)

        self.activated = activated

        signature = ensure_signature(self.signature)
        *_keys, last_key = signature.output_fields.keys()

        rationale_type = rationale_type or dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + last_key + "}. We ...",
        )

        self.extended_signature = signature.prepend("rationale", rationale_type, type_=str)

    def forward(self, **kwargs):
        new_signature = kwargs.pop("new_signature", None)
        if new_signature is None:
            if self.activated is True or (
                self.activated is None and isinstance(dsp.settings.lm, dsp.GPT3)
            ):
                signature = self.extended_signature
            else:
                signature = self.signature
        else:
            signature = new_signature
            # template = dsp.Template(self.signature.instructions, **new_signature)
        return super().forward(signature=signature, **kwargs)


"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""
