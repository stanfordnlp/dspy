import dspy
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils import settings
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature


class ChainOfThought(Module):
    def __init__(self, signature, rationale_type=None, **config):
        super().__init__()

        signature = ensure_signature(signature)

        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend("reasoning", rationale_type, type_=str)

        self.plain_predict = dspy.Predict(signature, **config)
        self.cot_predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        # Keep same logic with `dspy.Predict`
        lm = kwargs.pop("lm", self.lm) or settings.lm
        assert isinstance(lm, BaseLM), "No LM is loaded."

        # Custom models that subclassing `BaseLM` don't have this parameter
        if getattr(lm, "reasoning_model", False):
            return self.plain_predict(**kwargs)
        return self.cot_predict(**kwargs)
