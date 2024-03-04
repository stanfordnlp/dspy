from dspy.signatures.signature import Signature, signature_to_template
from dspy.primitives.example import Example
from dspy.primitives.template import Template

import typing as t
from .base import BaseBackend, GeneratedOutput
from .lm.litellm import BaseLM


class TemplateBackend(BaseBackend):
    """Behaves like LMs in prior versions of DSPy, using a template and parsing predictions."""

    lm: BaseLM

    def __call__(
        self,
        signature: Signature,
        demos: t.List[str] = [],
        **kwargs,
    ) -> list[GeneratedOutput]:
        """Wrap the signature and demos into an example, and pass through the Language Model, returning Signature compliant output"""

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(
                f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}."
            )

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Generate Template
        template = Template(signature)

        # Clean Up Kwargs Before Sending Through Language Model
        for input in signature.input_fields:
            del kwargs[input]

        pred = self.lm(template(example), **kwargs)

        # This returns a list of Examples
        extracted = [
            template.extract(example, prediction.message.content) for prediction in pred
        ]

        outputs = []
        for extract in extracted:
            outputs.append({})
            for key in signature.model_fields:
                outputs[-1][key] = extract.get(key)

        return outputs
