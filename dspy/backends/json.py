import json
import typing as t

from pydantic import Field
from dspy.backends.templates.json import JSONTemplate
from dspy.backends.templates.base import BaseTemplate

from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature

from .base import BaseBackend
from .lm import BaseLM


def patch_example(example: Example, data: dict[str, t.Any]) -> Example:
    example = example.copy()
    for k, v in data.items():
        example[k.lower()] = v

    return example


class JSONBackend(BaseBackend):
    lm: BaseLM

    def generate(
        self,
        signature: Signature,
        config: dict[str, t.Any] = {},
        demos: t.List[str] = [],
        template: BaseTemplate = JSONTemplate(),
        **kwargs,
    ) -> Completions:
        """Uses response_format json to generate structured predictions."""

        # Generate Example
        example = Example(demos=demos, **kwargs)

        # Clean Up Kwargs Before Sending Through Language Model
        for input in signature.input_fields:
            del kwargs[input]

        prompt = template.generate(signature=signature, example=example)
        pred = self.lm(
            prompt,
            response_format={"type": "json_object"},
            **config,
        )

        # Remove predictions which are not json valid
        extracted = [template.extract(signature=signature, example=example, raw_pred=prediction) for prediction in pred.generations]

        completions = Completions.new(
            signature=signature,
            examples=extracted,
            prompt=pred.prompt,
            kwargs=pred.kwargs,
        )

        return completions
