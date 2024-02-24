import dspy
import os
from pathlib import Path
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel
from joblib import Memory
from dsp.modules.cache_utils import cachedir


cachedir = os.environ.get("DSP_CACHEDIR") or os.path.join(Path.home(), ".joblib_cache")
_cache_memory = Memory(cachedir, verbose=0)


class BaseLM(BaseModel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = _cache_memory.cache(self._call)

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """Generates `n` predictions for the signature output."""
        if dspy.settings.cache:
            return self._cached(prompt, **kwargs)
        else:
            return self._call(prompt, **kwargs)

    @abstractmethod
    def _call(
        self,
        prompt: str,
        **kwargs,
    ) -> list[dict[str, t.Any]]:
        """Generates `n` predictions for the signature output."""
        ...

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        ...
