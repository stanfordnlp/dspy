import typing as t

from dspy.modeling.lm import BaseLM
from dspy.modeling.templates import BaseTemplate, TextTemplate
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature, SignatureMeta

from .base import BaseBackend


class TemplateBackend(BaseBackend):
    """Behaves like LMs in prior versions of DSPy, using a template and parsing predictions."""

    lm: BaseLM

    def generate(
        self,
        signature: Signature,
        demos: t.Optional[list[str]] = None,
        config: t.Optional[dict[str, t.Any]] = None,
        template: BaseTemplate = TextTemplate(),
        **kwargs,
    ) -> Completions:
        """Wrap the signature and demos into an example, and pass through the Language Model, returning Signature compliant output."""
        if config is None:
            config = {}

        if demos is None:
            demos = []

        # TODO: Move this check to logging
        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(
                f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.",
            )

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Pass through language model
        pred = self.lm(template.generate(signature, example), **config)

        # This returns a list of Examples
        extracted_examples = [
            template.extract(signature, example, prediction)
            for prediction in pred.generations
        ]

        if type(signature) != SignatureMeta:
            raise AssertionError("Signature not provided appropriately.")

        return Completions.new(
            signature=signature,
            examples=extracted_examples,
            prompt=pred.prompt,
            kwargs=pred.kwargs,
        )
