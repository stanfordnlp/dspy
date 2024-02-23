import typing as t

from pydantic import Field

from dsp.templates.template_v3 import Template
from dspy.signatures.signature import Signature

from .base import BaseBackend
from .lm.litellm import LiteLLM


StructuredOutput = t.TypeVar("StructuredOutput", dict[str, t.Any])


class LiteLLMBackend(BaseBackend):
    model: str
    lm: LiteLLM = Field(default_factory=LiteLLM)

    def __call__(
        self,
        signature: Signature,
        params: dict[str, t.Any],  # "new_signature" in dspy/predict/predict.py#79
        **kwargs,
    ) -> list[StructuredOutput]:
        ...
        # does this model support tool use? use that
        """
        Create tool from signature output fields and pass as tool to use
        json.loads tool_choice to create the
        """

        # does this model support JSON mode? use that
        """
        Define JSON format and pass as response_format
        json.loads from the messages in the response
        """

        # otherwise, wrap in a Template and pass through to the LM
        """
        See existing code in dspy/predict/predict.py lines 80-87 & 98-108 for
        how it's currently being done.
        This needs to get modified because we want the signature and params to get
        passed directly to ths Backend so we can do structured output using
        tool use / JSON mode
        """
