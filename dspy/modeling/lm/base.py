import os
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
from joblib import Memory
from pydantic import BaseModel, Field

import dspy

_cachedir = os.environ.get("DSP_CACHEDIR") or str(Path.home() / ".joblib_cache")
_cache_memory = Memory(_cachedir, verbose=0)


class LMOutput(BaseModel):
    kwargs: dict[str, t.Any]
    generations: list[str]


class BaseLM(BaseModel, ABC):
    history: list[LMOutput] = Field(default_factory=list, exclude=True)

    def __init__(self, *args: t.Any, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, **kwargs) -> LMOutput:
        """Generates `n` predictions for the signature output."""

        if dspy.settings.cache:
            generations = cached_generation(self, **kwargs)
        else:
            generations = self.generate(**kwargs)

        # This is necessary to satisfy the type checked for memoized functions
        if generations is None:
            raise ValueError("Generator failed to create generations.")

        output = LMOutput(kwargs=kwargs, generations=generations)
        self.history.append(output)

        return output

    @abstractmethod
    def generate(
        self,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""
        ...


def cached_generation(cls: BaseLM, **kwargs):
    hashed = joblib.hash(cls.model_dump_json())

    @_cache_memory.cache(ignore=["cls"])
    def _cache_call(cls: BaseLM, hash: str, **kwargs):
        return cls.generate(**kwargs)

    return _cache_call(cls, hash=hashed, **kwargs)
