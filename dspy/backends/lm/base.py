import dspy
import os
from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel
from joblib import Memory


_cachedir = os.environ.get("DSP_CACHEDIR") or str(Path.home() / ".joblib_cache")
_cache_memory = Memory(_cachedir, verbose=0)

Completion = t.TypeVar("Completion", bound=dict)


MinimalLM = t.Callable[[str, float, int, int], list[Completion]]


class BaseLM(BaseModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generate_with_cache = _cache_memory.cache(self.generate)

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """Generates `n` predictions for the signature output."""
        generator = self.generate_with_cache if dspy.settings.cache else self.generate
        return generator(prompt, **kwargs)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> list[Completion]:
        """Generates `n` predictions for the signature output."""
        ...

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        ...
