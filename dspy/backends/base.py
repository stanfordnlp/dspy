from abc import ABC, abstractmethod
import typing as t

from pydantic import BaseModel

from dspy.signatures.signature import Signature


ReturnValue = t.TypeVar("ReturnValue", bound=dict)


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    @abstractmethod
    def __call__(
        self,
        signature: Signature,
        temperature: float,
        max_tokens: int,
        n: int,
        **kwargs,
    ) -> list[ReturnValue]:
        """Generates `n` predictions for the signature output."""
        pass
