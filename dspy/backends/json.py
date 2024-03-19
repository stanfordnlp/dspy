import json
import typing as t

from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.primitives.template import Template
from dspy.signatures.signature import Signature

from .base import BaseBackend
from .lm import BaseLM


def patch_example(example: Example, data: dict[str, t.Any]) -> Example:
    example = example.copy()
    for k, v in data.items():
        example[k] = v

    return example


class JSONBackend(BaseBackend):
    lm: BaseLM

    def generate(
        self,
        signature: Signature,
        config: dict[str, t.Any] = {},
        demos: t.List[str] = [],
        **kwargs,
    ) -> Completions:
        """Uses response_format json to generate structured predictions."""

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Generate Template
        template = Template(signature)

        # Clean Up Kwargs Before Sending Through Language Model
        for input in signature.input_fields:
            del kwargs[input]

        pred = self.lm(
            template(example, is_json=True),
            response_format={"type": "json_object"},
            **config,
        )
        extracted = [
            json.loads(prediction["message"]["content"])
            for prediction in pred.generations
        ]

        extracted_examples = [patch_example(example, extract) for extract in extracted]

        completions = Completions.new(
            signature=signature,
            examples=extracted_examples,
            prompt=pred.prompt,
            kwargs=pred.kwargs,
        )

        return completions
