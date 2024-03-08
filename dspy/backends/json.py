import json
import typing as t
from dspy.signatures.signature import Signature
from dspy.primitives.example import Example
from dspy.primitives.template import Template
from dspy.primitives.prediction import (
    Completions,
    convert_to_completion,
    get_completion_data,
)


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
            **kwargs,
        )
        extracted = [
            json.loads(prediction["message"]["content"])
            for prediction in pred.generations
        ]

        extracted = [patch_example(example, extract) for extract in extracted]

        completion_list = [
            convert_to_completion(signature, example) for example in extracted
        ]
        completions = Completions(
            signature=signature,
            completions=completion_list,
            prompt=pred.prompt,
            kwargs=pred.kwargs,
            data=get_completion_data(completion_list),
        )

        return completions
