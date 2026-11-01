"""Restricted pickle deserialization for disk cache.

Provides a SafeDisk from diskcache_community configured for the DSPy cache.

Trust model:
- openai.types.* is trusted by module prefix (pydantic
  data models only, forward-compatible with library upgrades)
- numpy arrays are stored by the disk in its own array format
- user safe_types are trusted by identity

SafeDisk resolves classes only in modules that are already imported, so this
module imports openai.types when openai is installed. An entry that the disk
refuses is treated as a cache miss.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable

from diskcache_community import SafeDisk

_TRUSTED_MODULES = ("openai.types",)

try:
    import openai.types  # noqa: F401
except ImportError:
    pass

# Kept for code that imports it from earlier releases.
DeserializationError = pickle.UnpicklingError


def restricted_disk(safe_types: Iterable[type] = ()) -> type[SafeDisk]:
    """Return a SafeDisk subclass that also allows OpenAI data models, numpy arrays and the given types."""
    return (
        SafeDisk.with_types(*safe_types, modules=_TRUSTED_MODULES)
        .with_numpy()
        .with_options(on_error="miss")
    )
