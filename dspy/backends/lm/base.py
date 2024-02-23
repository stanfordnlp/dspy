from abc import ABC, abstractmethod

from pydantic import BaseModel
from joblib import Memory
from dsp.modules.cache_utils import cachedir


_cache_memory = Memory(cachedir, verbose=0)


class BaseLM(BaseModel, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached = _cache_memory.cache(self._call)

    def __call__(self, prompt: str, **kwargs) -> list[str]:
        """Generates `n` predictions for the signature output."""
        return self._cached(prompt, **kwargs)

    @abstractmethod
    def _call(
        self,
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""
        ...

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        ...
