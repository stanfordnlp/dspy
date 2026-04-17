"""Restricted pickle deserialization for disk cache.

Provides a RestrictedDisk subclass that overrides diskcache's fetch to use
a restricted unpickler. Allows litellm.types.* and openai.types.* by module
prefix, numpy by exact (module, name) pairs, and user-registered safe_types.
"""

from __future__ import annotations

import io
import os.path
import pickle
from typing import Any

from diskcache import Disk
from diskcache.core import MODE_PICKLE

_SAFE_MODULE_PREFIXES = (
    "litellm.types.",
    "openai.types.",
)

_NUMPY_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("numpy", "dtype"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
})


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


def _restricted_load(f: Any, allowed: set[tuple[str, str]]) -> Any:
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> type:
            if any(module.startswith(p) for p in _SAFE_MODULE_PREFIXES):
                return super().find_class(module, name)
            if (module, name) in _NUMPY_ALLOWED or (module, name) in allowed:
                return super().find_class(module, name)
            raise DeserializationError(
                f"Type {module}.{name} is not in the safe_types allowlist. "
                f"Register it via safe_types=[...] in configure_cache()."
            )

    return _Unpickler(f).load()


class RestrictedDisk(Disk):
    """Disk subclass that restricts pickle deserialization to an allowlist."""

    _allowed: set[tuple[str, str]]

    def fetch(self, mode, filename, value, read):
        if mode == MODE_PICKLE:
            if value is None:
                with open(os.path.join(self._directory, filename), "rb") as f:
                    return _restricted_load(f, self._allowed)
            return _restricted_load(io.BytesIO(value), self._allowed)
        return super().fetch(mode, filename, value, read)


def make_restricted_disk(allowed: set[tuple[str, str]]) -> type[RestrictedDisk]:
    """Create a RestrictedDisk subclass bound to a specific allowlist."""
    return type("RestrictedDisk", (RestrictedDisk,), {"_allowed": allowed})
