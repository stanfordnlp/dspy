from .predict import Predict
from ..primitives.program import Module

import dsp

class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = Predict(signature).signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}

        for idx in range(M):
            candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
            extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
        
        rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        signature = dsp.Template(signature.instructions, **extended_kwargs)
        self.predict = Predict(signature, temperature=temperature, **config)
        self.last_key = last_key
    
    def forward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.rationale.strip().split('\n')[0].strip()
            answer = c[self.last_key].strip().split('\n')[0].strip()
            attempts.append(f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»")

        assert len(attempts) == self.M, len(attempts)

        kwargs = {**{f'reasoning_attempt_{idx+1}': attempt for idx, attempt in enumerate(attempts)}, **kwargs}
        return self.predict(**kwargs)
