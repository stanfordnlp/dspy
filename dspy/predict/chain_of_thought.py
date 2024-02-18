import dsp

from .predict import Predict


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

        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        DEFAULT_RATIONALE_TYPE = dsp.Type(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + last_key + "}. We ...",
        )

        rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE

        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update(
            {"rationale": rationale_type, last_key: signature.kwargs[last_key]}
        )

        self.extended_signature = dsp.Template(
            signature.instructions, **extended_kwargs
        )

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
            signature = dsp.Template(self.signature.instructions, **new_signature)
        return super().forward(signature=signature, **kwargs)


    def dump_state(self):
        state = super().dump_state()

        # Cache the signature instructions and the last field's name.
        state["extended_signature_instructions"] = self.extended_signature.instructions
        state["extended_signature_prefix"] = self.extended_signature.fields[-1].name

        return state

    def load_state(self, state):
        super().load_state(state)
        
        # Reconstruct the signature.
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.extended_signature.instructions = instructions
        
        if "extended_signature_prefix" in state:
            prefix = state["extended_signature_prefix"]
            self.extended_signature.fields[-1] = self.extended_signature.fields[-1]._replace(name=prefix)

"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""
