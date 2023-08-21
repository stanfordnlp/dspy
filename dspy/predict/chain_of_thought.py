import dsp

from .predict import Predict



# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

class ChainOfThought(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)

        self.activated = activated

        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        DEFAULT_RATIONALE_TYPE = dsp.Type(prefix="Reasoning: Let's think step by step in order to",
                                          desc="${produce the " + last_key + "}. We ...")

        rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE
        
        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})
        
        self.extended_signature = dsp.Template(signature.instructions, **extended_kwargs)
    
    def forward(self, **kwargs):
        signature = self.signature

        if self.activated is True or (self.activated is None and isinstance(dsp.settings.lm, dsp.GPT3)):
            signature = self.extended_signature
        
        return super().forward(signature=signature, **kwargs)


"""
TODO: In principle, we can update the field's prefix during forward too to fill any thing based on the input args.

IF the user didn't overwrite our default rationale_type.
"""