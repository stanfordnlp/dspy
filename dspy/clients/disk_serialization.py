"""Restricted pickle deserialization for disk cache.

Provides a RestrictedDisk subclass that overrides diskcache's fetch to use
a restricted unpickler.

Trust model:
- litellm.types.* and openai.types.* are trusted by module prefix (pydantic
  data models only, forward-compatible with library upgrades)
- numpy reconstruction helpers are trusted by exact (module, name) pairs
- user safe_types are trusted by exact (module, qualname) pairs
"""

from __future__ import annotations

import io
import os.path
import pickle
from typing import Any

from diskcache import Disk
from diskcache.core import MODE_PICKLE

_TRUSTED_MODULE_PREFIXES = (
    "litellm.types.",
    "openai.types.",
)

_NUMPY_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("_codecs", "encode"),
})


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


class _RestrictedUnpickler(pickle.Unpickler):
    _allowed: frozenset[tuple[str, str]] = frozenset()

    def find_class(self, module: str, name: str) -> type:
        if any(module.startswith(p) for p in _TRUSTED_MODULE_PREFIXES):
            return super().find_class(module, name)
        if (module, name) in _NUMPY_ALLOWED or (module, name) in self._allowed:
            return super().find_class(module, name)
        raise DeserializationError(
            f"Type {module}.{name} is not in the safe_types allowlist. "
            f"Register it via dspy.configure_cache(safe_types=[...])."
        )


def _restricted_load(f: Any, allowed: frozenset[tuple[str, str]]) -> Any:
    unpickler = _RestrictedUnpickler(f)
    unpickler._allowed = allowed
    try:
        return unpickler.load()
    except DeserializationError:
        raise
    except Exception as e:
        raise DeserializationError(f"Corrupt cache entry: {e}") from e


class _RestrictedDisk(Disk):
    """Disk subclass that restricts pickle deserialization to an allowlist."""

    _allowed: frozenset[tuple[str, str]]

    def _load(self, f: Any) -> Any:
        unpickler = _RestrictedUnpickler(f)
        unpickler._allowed = self._allowed
        try:
            return unpickler.load()
        except DeserializationError:
            raise
        except Exception as e:
            raise DeserializationError(f"Corrupt cache entry: {e}") from e

    def fetch(self, mode, filename, value, read):
        if mode == MODE_PICKLE:
            if value is None:
                with open(os.path.join(self._directory, filename), "rb") as f:
                    return self._load(f)
            return self._load(io.BytesIO(value))
        return super().fetch(mode, filename, value, read)


def restricted_disk(allowed: frozenset[tuple[str, str]]) -> type[_RestrictedDisk]:
    """Return a Disk subclass bound to the given allowlist.

    diskcache expects ``disk=`` to be a class it instantiates itself, so
    we return a subclass with ``_allowed`` baked in as a class attribute.
    """
    return type("RestrictedDisk", (_RestrictedDisk,), {"_allowed": allowed})
