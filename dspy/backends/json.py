import json
import typing as t
from dspy.signatures.signature import Signature
from dspy.primitives.example import Example
from dspy.primitives.template import Template


from .base import BaseBackend, Completion, convert_to_completion
from .lm import BaseLM


class JSONBackend(BaseBackend):
    lm: BaseLM

    def generate(
        self,
        signature: Signature,
        demos: t.List[str] = [],
        **kwargs,
    ) -> list[Completion]:
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
        extracted = [json.loads(prediction.message.content) for prediction in pred]

        return [convert_to_completion(signature, example) for example in extracted]
