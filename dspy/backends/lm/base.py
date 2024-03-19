import os
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

from joblib import Memory
from pydantic import BaseModel, Field

import dspy

_cachedir = os.environ.get("DSP_CACHEDIR") or str(Path.home() / ".joblib_cache")
_cache_memory = Memory(_cachedir, verbose=0)

GeneratedContent = dict[str, t.Any]


class LMOutput(BaseModel):
    prompt: str
    generations: list[GeneratedContent]
    kwargs: dict[str, t.Any]


class BaseLM(BaseModel, ABC):
    history: list[LMOutput] = Field(default_factory=list)

    def __init__(self, *args: t.Any, **kwargs):
        super().__init__(*args, **kwargs)
        self._generate_with_cache = _cache_memory.cache(self.generate)

    def __call__(self, prompt: str, **kwargs) -> LMOutput:
        """Generates `n` predictions for the signature output."""
        generator = self._generate_with_cache if dspy.settings.cache else self.generate
        generations = generator(prompt, **kwargs)

        # This is necessary to satisfy the type checked for memoized functions
        if generations is None:
            raise ValueError("Generator failed to create generations.")

        output = LMOutput(prompt=prompt, generations=generations, kwargs=kwargs)
        self.history.append(output)

        return output

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> list[GeneratedContent]:
        """Generates `n` predictions for the signature output."""
        ...

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        ...
