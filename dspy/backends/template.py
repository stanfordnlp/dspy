from dspy.signatures.signature import Signature

from .base import BaseBackend, ReturnValue
from .lm.litellm import BaseLM


class TemplateBackend(BaseBackend):
    """Behaves like LMs in prior versions of DSPy, using a template and parsing predictions."""

    lm: BaseLM

    def __call__(
        self,
        signature: Signature,
        temperature: float,
        max_tokens: int,
        n: int,
        **kwargs,
    ) -> list[ReturnValue]:
        pass
