import dspy
import dsp

from .predict import Predict


class Retry(Predict):
    def __init__(self, module):
        super().__init__(module.signature)
        self.module = module
        self.original_signature = module.signature.signature
        self.original_forward = module.forward
        self.new_signature = self._create_new_signature(self.original_signature)

    def _create_new_signature(self, original_signature):
        extended_signature = {}
        input_fields = original_signature.input_fields()
        output_fields = original_signature.output_fields()
        modified_output_fields = {}

        for key, value in output_fields.items():
            modified_output_fields[f"past_{key}"] = dspy.InputField(
                prefix="Past " + value.prefix,
                desc="past output with errors",
                format=value.format,
            )

        extended_signature.update(input_fields)
        extended_signature.update(modified_output_fields)

        extended_signature["feedback"] = dspy.InputField(
            prefix="Instructions:",
            desc="Some instructions you must satisfy",
            format=str,
        )
        extended_signature.update(output_fields)

        return extended_signature

    def forward(self, *args, **kwargs):
        for key, value in kwargs["past_outputs"].items():
            past_key = f"past_{key}"
            if past_key in self.new_signature:
                kwargs[past_key] = value
        del kwargs["past_outputs"]
        kwargs["signature"] = self.new_signature
        demos = kwargs.pop("demos", self.demos if self.demos is not None else [])
        return self.original_forward(demos=demos, **kwargs)
    
    def __call__(self, **kwargs):
        if dspy.settings.backtrack_to == self.module:
            for key, value in dspy.settings.backtrack_to_args.items():
                kwargs.setdefault(key, value)
            return self.forward(**kwargs)
        else:
            # seems like a hack, but it works for now
            demos = kwargs.pop("demos", self.demos if self.demos is not None else [])
            return self.module(demos=demos, **kwargs)
