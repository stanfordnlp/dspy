import typing as t

from dspy.modeling.lm import BaseLM
from dspy.modeling.templates.base import BaseTemplate
from dspy.modeling.templates.json import JSONTemplate
from dspy.primitives.example import Example
from dspy.primitives.prediction import Completions
from dspy.signatures.signature import Signature

from .template import TemplateBackend


def patch_example(example: Example, data: dict[str, t.Any]) -> Example:
    example = example.copy()
    for k, v in data.items():
        example[k.lower()] = v

    return example


class JSONBackend(TemplateBackend):
    lm: BaseLM

    def generate(
        self,
        signature: Signature,
        demos: t.Optional[list[str]] = None,
        config: t.Optional[dict[str, t.Any]] = None,
        template: BaseTemplate = JSONTemplate(),
        **kwargs,
    ) -> Completions:
        """Uses response_format json to generate structured predictions."""
        if config is None:
            config = {}

        config.update({"response_format": {"type": "json_object"}})
        return super().generate(signature=signature, config=config, demos=demos, template=template, **kwargs)
