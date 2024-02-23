from abc import ABC, abstractmethod
from pydantic import BaseModel

from .lm.base import BaseLM


class BaseBackend(BaseModel, ABC):
    lm: BaseLM

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
