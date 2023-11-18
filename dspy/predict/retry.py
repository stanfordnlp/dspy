import dspy
import dsp

class Retry(dspy.Predict):
    def __init__(self, module):
        super().__init__(module.signature)
        self.module = module
        self.original_signature = module.signature.signature
        self.new_signature = self._create_new_signature(self.original_signature)

    def _create_new_signature(self, original_signature):
        extended_signature = {}
        for key, value in original_signature.items():
            extended_signature[key] = value
        extended_signature['traces'] = dspy.InputField(
            prefix="Trace:", 
            desc="Traces from your past attempts", 
            format=dsp.passages2text
        )
        extended_signature['feedback'] = dspy.InputField(
            prefix="Instruction:", 
            desc="Some instructions you must satisfy", 
            format=str
        )
        return extended_signature

    def forward(self, **kwargs):
        return self.module.forward(signature=self.new_signature, **kwargs)