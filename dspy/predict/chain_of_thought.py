import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature


# TODO: This shouldn't inherit from Predict. It should be a module that has one or two predictors.
# Let's focus on the activated case. It's a predictor with the expanded signature.
# Now, when deactivated, it's a predictor with the original signature.
# When activate is None, though, we need the expanded one but during forward we need to pass the right signature.


class ChainOfThought(Module):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__()

        self.activated = activated

        self.signature = signature = ensure_signature(signature)
        *_keys, last_key = signature.output_fields.keys()

        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${produce the " + last_key + "}. We ..."

        if dspy.settings.experimental:
            desc = "${produce the output fields}. We ..."

        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)

        extended_signature = signature.prepend("rationale", rationale_type, type_=str)
        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature

    def forward(self, **kwargs):
        assert self.activated in [True, False]

        signature = kwargs.pop("new_signature", self._predict.extended_signature if self.activated else self.signature)
        return self._predict(signature=signature, **kwargs)
        # return super().forward(signature=signature, **kwargs)

    @property
    def demos(self):
        return self._predict.demos
    
    @property
    def extended_signature(self):
        return self._predict.extended_signature

"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""
