import typing as t

from dsp.templates.template_v3 import Template

from .base import BaseBackend
from .lm.litellm import LiteLLM


class LiteLLMBackend(BaseBackend):
    model: str
    lm: LiteLLM

    def __call__(self, template: Template, **kwargs) -> list[dict[str, t.Any]]:
        # does model support tool use? Use that for structured output with output_fields()
        completions = self.lm(template(...), tools=[(jsonschema for the template)], tool_choice="the name of the json schema to use", **kwargs)
        # additional parsing?

        # otherwise do the thing that was happening before
