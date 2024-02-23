from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseBackend(BaseModel, ABC):
    """A backend takes a signature, its params, and returns a list of structured predictions."""

    @abstractmethod
    def __call__(
        # self, WTF is this?
        # prompt: str,
        # temperature: float,
        # max_tokens: int,
        # n: int,
        # **kwargs,
    ) -> list[dict[str, str]]:
        """Generates `n` predictions for the signature output."""
        ...
