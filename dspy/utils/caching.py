from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from typing import Union
import os

import ujson


_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".dspy_cache")
DSPY_CACHEDIR = os.environ.get("DSPY_CACHEDIR") or _DEFAULT_CACHE_DIR


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the DSPy cache directory."""
    subdir = os.path.join(DSPY_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir


class LRUCache(OrderedDict):
    maxsize: int

    def __init__(self, iterable, maxsize: int):
        super().__init__(iterable)
        self.maxsize = maxsize

    @classmethod
    def load(cls, file, maxsize: int) -> "LRUCache":
        return cls(ujson.load(file), maxsize)

    def dump(self, file) -> None:
        ujson.dump([[k, v] for k, v in self.items()], file)

    def __setitem__(self, request: dict, value):
        key = self.cache_key(request)

        if key in self:
            self.move_to_end(key)
            return

        if len(self) == self.maxsize:
            self.popitem(last=False)

        super().__setitem__(key, value)

    def __getitem__(self, request: dict):
        key = self.cache_key(request)
        return super().__getitem__(key)

    def __contains__(self, request: dict):
        key = self.cache_key(request)
        return super().__contains__(key)

    def get(self, request: dict, default=None):
        key = self.cache_key(request)
        return super().get(key, default)

    def __delitem__(self, request: dict):
        key = self.cache_key(request)
        super().__delitem__(key)

    def pop(self, request: dict, default=None):
        key = self.cache_key(request)
        return super().pop(key, default)

    @staticmethod
    def cache_key(request: Union[dict, str]) -> str:
        params = request
        if isinstance(request, dict):
            params = {k: v for k, v in request.items() if not callable(v)}
        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()
