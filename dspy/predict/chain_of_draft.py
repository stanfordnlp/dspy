import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature


class ChainOfDraft(Module):
    def __init__(self, signature, rationale_type=None, **config):
        super().__init__()

        signature = ensure_signature(signature)

        prefix = "draft: Think step by step, but only keep a minimum draft " \
                 "for each thinking step, with 5 words at most"
        desc = "${draft}"
        rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend("draft", rationale_type, type_=str)
        
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
