"""Safe cache serialization without pickle.

Encodes pydantic models and dataclasses as JSON with embedded type
metadata (module + class name), then reconstructs via importlib on
decode. An allowlist of (module, classname) pairs controls which types
may be reconstructed. Numpy arrays use a separate binary prefix.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
from typing import Any

import orjson
import pydantic
from diskcache import Disk
from diskcache.core import MODE_BINARY, MODE_RAW, UNKNOWN

_NPY_PREFIX = b"npy:"


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


class LegacyFormatError(DeserializationError):
    """Raised when a cache entry was written by the legacy pickle-based backend."""


def _is_ndarray(value: Any) -> bool:
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:
        return False


def default_allowed_types() -> set[tuple[str, str]]:
    """Return the built-in set of allowed (module, classname) pairs."""
    allowed: set[tuple[str, str]] = set()

    try:
        from openai.types.chat.chat_completion import ChatCompletion
        from openai.types.responses.response import Response
    except ImportError:
        pass
    else:
        for cls in (ChatCompletion, Response):
            allowed.add((cls.__module__, cls.__name__))

    try:
        from litellm.types.utils import EmbeddingResponse, ModelResponse, ModelResponseStream
    except ImportError:
        pass
    else:
        for cls in (EmbeddingResponse, ModelResponse, ModelResponseStream):
            allowed.add((cls.__module__, cls.__name__))

    return allowed


def encode(value: Any) -> bytes:
    """Serialize a value to the safe cache wire format."""
    if _is_ndarray(value):
        buffer = io.BytesIO()
        import numpy as np

        np.save(buffer, value, allow_pickle=False)
        return _NPY_PREFIX + buffer.getvalue()

    if isinstance(value, pydantic.BaseModel):
        return orjson.dumps({
            "kind": "pydantic",
            "module": value.__class__.__module__,
            "cls": value.__class__.__name__,
            "data": value.model_dump(mode="json"),
        })

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return orjson.dumps({
            "kind": "dataclass",
            "module": value.__class__.__module__,
            "cls": value.__class__.__name__,
            "data": {f.name: getattr(value, f.name) for f in dataclasses.fields(value)},
        })

    try:
        return orjson.dumps({"kind": "value", "data": value})
    except orjson.JSONEncodeError as e:
        raise TypeError(
            f"Disk cache only supports JSON values, pydantic models, dataclasses, "
            f"and numpy arrays; got {type(value).__module__}.{type(value).__qualname__}"
        ) from e


def decode(data: bytes, *, allowed: set[tuple[str, str]]) -> Any:
    """Deserialize a value, restricting typed reconstruction to the allowlist."""
    if data.startswith(_NPY_PREFIX):
        try:
            import numpy as np
        except ImportError as e:
            raise DeserializationError("Cannot import module 'numpy'") from e
        try:
            return np.load(io.BytesIO(data[len(_NPY_PREFIX) :]), allow_pickle=False)
        except (TypeError, ValueError) as e:
            raise DeserializationError("Invalid ndarray payload in cache entry") from e

    try:
        payload = orjson.loads(data)
    except orjson.JSONDecodeError as e:
        raise DeserializationError("Invalid JSON in cache entry") from e

    kind = payload.get("kind")
    if kind == "value":
        return payload["data"]

    if kind in ("pydantic", "dataclass"):
        mod_name = payload["module"]
        cls_name = payload["cls"]

        if (mod_name, cls_name) not in allowed:
            raise DeserializationError(
                f"Type {mod_name}.{cls_name} is not in the safe_types allowlist. "
                f"Register it via safe_types=[...] in configure_cache()."
            )

        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as e:
            raise DeserializationError(f"Cannot import {mod_name}.{cls_name}") from e

        try:
            return cls(**payload["data"])
        except (TypeError, pydantic.ValidationError) as e:
            raise DeserializationError(f"Failed to reconstruct {mod_name}.{cls_name}") from e

    raise DeserializationError(f"Unknown cache entry kind: {kind!r}")


class SafeDisk(Disk):
    """Disk backend that uses safe JSON serialization with an allowlist.

    Use ``make_safe_disk(allowed)`` to create a subclass bound to a
    specific allowlist.
    """

    _allowed: set[tuple[str, str]]

    def store(self, value, read, key=UNKNOWN):
        if not read:
            return super().store(encode(value), False, key=key)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        if mode not in (MODE_RAW, MODE_BINARY):
            raise LegacyFormatError(
                f"Unsupported diskcache mode {mode}; refusing to unpickle legacy entry"
            )
        data = super().fetch(mode, filename, value, read)
        if read:
            return data
        return decode(data, allowed=self._allowed)


def make_safe_disk(allowed: set[tuple[str, str]]) -> type[SafeDisk]:
    """Create a SafeDisk subclass bound to a specific allowlist."""
    return type("SafeDisk", (SafeDisk,), {"_allowed": allowed})
