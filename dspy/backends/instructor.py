
from dspy.backends.lm.base import MinimalLM
from dspy.signatures.signature import Signature

from .base import BaseBackend, ReturnValue


class InstructorBackend(BaseBackend):
    lm: MinimalLM

    def __call__(
        self,
        signature: Signature,
        temperature: float,
        max_tokens: int,
        n: int,
        **kwargs,
    ) -> list[ReturnValue]:
        """Uses instructor to generate structured predictions."""
