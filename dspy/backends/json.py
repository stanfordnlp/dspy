from dspy.signatures.signature import Signature

from .base import BaseBackend, ReturnValue
from .lm.litellm import BaseLM


class JSONBackend(BaseBackend):
    lm: BaseLM

    def __call__(
        self,
        signature: Signature,
        temperature: float,
        max_tokens: int,
        n: int,
        **kwargs,
    ) -> list[ReturnValue]:
        """Uses response_format json to generate structured predictions."""
