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


    def dump_state(self):
        state = super().dump_state()

        # Cache the signature instructions and the last field's name.
        state["extended_signature_instructions"] = self.extended_signature.instructions

        *_, last_key = self.signature.fields.keys()
        state["extended_signature_prefix"] = self.extended_signature.fields[last_key].json_schema_extra['prefix']

        return state

    def load_state(self, state):
        super().load_state(state)
                    
        # Reconstruct the signature.
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.extended_signature = self.extended_signature.with_instructions(instructions)

        if "extended_signature_prefix" in state:
            prefix = state["extended_signature_prefix"]
            *_, last_key = self.extended_signature.fields.keys()
            self.extended_signature = self.extended_signature.with_updated_fields(last_key, prefix=prefix)

"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""
