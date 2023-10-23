import dsp

from .predict import Predict



# TODO: FIXME: Insert this right before the *first* output field. Also rewrite this to use the new signature system.

class ChainOfThoughtWithHint(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)

        self.activated = activated

        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        DEFAULT_HINT_TYPE = dsp.Type(prefix="Hint:", desc="${hint}")

        DEFAULT_RATIONALE_TYPE = dsp.Type(prefix="Reasoning: Let's think step by step in order to",
                                          desc="${produce the " + last_key + "}. We ...")

        rationale_type = rationale_type or DEFAULT_RATIONALE_TYPE
        
        extended_kwargs1 = {key: signature.kwargs[key] for key in keys}
        extended_kwargs1.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        extended_kwargs2 = {key: signature.kwargs[key] for key in keys}
        extended_kwargs2.update({'hint': DEFAULT_HINT_TYPE, 'rationale': rationale_type, last_key: signature.kwargs[last_key]})
        
        self.extended_signature1 = dsp.Template(signature.instructions, **extended_kwargs1)
        self.extended_signature2 = dsp.Template(signature.instructions, **extended_kwargs2)
    
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