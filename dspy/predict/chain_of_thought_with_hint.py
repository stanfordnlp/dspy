import dspy

from .predict import Module


class ChainOfThoughtWithHint(Module):
    def __init__(self, signature, rationale_field_type=None, **config):
        """

        A module that reasons step by step in order to predict the output of a task, with optional hint support.

        

        This module extends ChainOfThought by allowing an optional "hint" parameter that can be provided

        during inference to guide the reasoning process.

        

        Args:

        signature (Type[dspy.Signature]): The signature of the module.

        rationale_field (Optional[Union[dspy.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.

        rationale_field_type (Type): The type of the rationale field.

        hint: The hint to provide to the module.

        **config: The configuration for the module.

        """
        self.signature = dspy.ensure_signature(signature)
        self.module = dspy.ChainOfThought(signature, rationale_field_type=rationale_field_type, **config)

    def forward(self, **kwargs):
        if kwargs.get("hint"):
            hint = f"\n\t\t(secret hint: {kwargs.pop('hint')})"
            original_kwargs = kwargs.copy()

            # Convert the first field's value to string and append the hint
            last_key = list(self.signature.input_fields.keys())[-1]
            kwargs[last_key] = str(kwargs[last_key]) + hint

            # Run CoT then update the trace with original kwargs, i.e. without the hint.
            with dspy.context(trace=[]):
                pred = self.module(**kwargs)
                this_trace = dspy.settings.trace[-1]

            dspy.settings.trace.append((this_trace[0], original_kwargs, this_trace[2]))
            return pred

        return self.module(**kwargs)
