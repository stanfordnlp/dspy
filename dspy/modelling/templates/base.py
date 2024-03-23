
from abc import ABC, abstractmethod

from pydantic import BaseModel

from dspy import Example, Signature


class BaseTemplate(BaseModel, ABC):
    @abstractmethod
    def generate(self, signature: Signature, example: Example) -> str:
        """Generate a prompt given an Example"""
        ...

    @abstractmethod
    def extract(self, signature: Signature, example: Example, raw_pred: str) -> Example:
        """Extracts the answer from the LM raw prediction, and returns an updated Example"""
        ...
