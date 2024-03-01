from dspy.signatures.signature import Signature

from .base import BaseBackend, ReturnValue
from .lm.litellm import BaseLM


class ToolBackend(BaseBackend):
    lm: BaseLM

    def __call__(
        self,
        signature: Signature,
        temperature: float,
        max_tokens: int,
        n: int,
        **kwargs,
    ) -> list[ReturnValue]:
        """Uses tools and tool_choice to generate structured predictions."""
