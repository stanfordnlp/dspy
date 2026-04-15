"""Safe cache serialization without pickle.

Encodes pydantic models and dataclasses as JSON with embedded type
metadata (module + class name), then reconstructs via importlib on
decode. An allowlist of (module, classname) pairs controls which types
may be reconstructed. Numpy arrays use a separate binary prefix.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import io
from hashlib import sha256 as _sha256
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


@functools.lru_cache(maxsize=256)
def _schema_hash(cls: type) -> str:
    """Short hash of a class's schema; detects stale entries after library upgrades."""
    if issubclass(cls, pydantic.BaseModel):
        raw = orjson.dumps(cls.model_json_schema(), option=orjson.OPT_SORT_KEYS)
    else:  # dataclass
        raw = orjson.dumps([f.name for f in dataclasses.fields(cls)])
    return _sha256(raw).hexdigest()[:16]


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


def _typed_envelope(value: Any, kind: str, data: Any, allowed: set[tuple[str, str]] | None) -> bytes:
    cls = value.__class__
    mod, name = cls.__module__, cls.__name__
    if allowed is not None and (mod, name) not in allowed:
        raise TypeError(
            f"Type {mod}.{name} is not in the safe_types allowlist. "
            f"Register it via safe_types=[...] in configure_cache()."
        )
    return orjson.dumps({
        "v": 1, "kind": kind, "module": mod, "cls": name,
        "schema": _schema_hash(cls), "data": data,
    })


def encode(value: Any, *, allowed: set[tuple[str, str]] | None = None) -> bytes:
    """Serialize *value* to the safe cache wire format.

    When *allowed* is provided, pydantic/dataclass types not in the set raise TypeError.
    """
    if _is_ndarray(value):
        import numpy as np

        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return _NPY_PREFIX + buffer.getvalue()

    if isinstance(value, pydantic.BaseModel):
        return _typed_envelope(value, "pydantic", value.model_dump(mode="json"), allowed)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        data = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        return _typed_envelope(value, "dataclass", data, allowed)

    try:
        return orjson.dumps({"v": 1, "kind": "value", "data": value})
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

        stored_schema = payload.get("schema")
        if stored_schema is not None and _schema_hash(cls) != stored_schema:
            raise DeserializationError(
                f"Schema for {mod_name}.{cls_name} has changed since this entry was cached"
            )

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
            return super().store(encode(value, allowed=self._allowed), False, key=key)
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
