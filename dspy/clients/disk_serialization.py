"""Safe cache serialization without pickle.

Uses a routing-byte prefix (inspired by CocoIndex) to dispatch between
serialization engines:

    0x01  msgspec (msgpack) -- plain values, dataclasses
    0x02  pydantic          -- BaseModel subclasses (type metadata in msgpack body)
    0x80  restricted pickle -- numpy ndarray only

See: https://cocoindex.io/blogs/type-guided-serde
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import io
import pickle
from hashlib import sha256 as _sha256
from typing import Any

import msgspec.msgpack
import pydantic
from diskcache import Disk
from diskcache.core import MODE_BINARY, MODE_RAW, UNKNOWN

SAFE_TAG = "s"

_TAG_MSGSPEC = b"\x01"
_TAG_PYDANTIC = b"\x02"
_TAG_PICKLE = b"\x80"


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


class LegacyFormatError(DeserializationError):
    """Raised when a cache entry was written by the legacy pickle-based backend."""


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise DeserializationError(f"Refusing to unpickle {module}.{name}")
        return super().find_class(module, name)


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
        raw = msgspec.json.encode(cls.model_json_schema())
    else:  # dataclass
        raw = msgspec.json.encode([f.name for f in dataclasses.fields(cls)])
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


def _check_allowlist(value: Any, allowed: set[tuple[str, str]] | None) -> None:
    if allowed is None:
        return
    cls = value.__class__
    mod, name = cls.__module__, cls.__name__
    if (mod, name) not in allowed:
        raise TypeError(
            f"Type {mod}.{name} is not in the safe_types allowlist. "
            f"Register it via safe_types=[...] in configure_cache()."
        )


def encode(value: Any, *, allowed: set[tuple[str, str]] | None = None) -> bytes:
    """Serialize *value* to the safe cache wire format."""
    if _is_ndarray(value):
        return _TAG_PICKLE + pickle.dumps(value)

    if isinstance(value, pydantic.BaseModel):
        _check_allowlist(value, allowed)
        cls = value.__class__
        payload = {
            "module": cls.__module__,
            "cls": cls.__name__,
            "schema": _schema_hash(cls),
            "data": value.model_dump(mode="json"),
        }
        return _TAG_PYDANTIC + msgspec.msgpack.encode(payload)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _check_allowlist(value, allowed)
        cls = value.__class__
        payload = {
            "module": cls.__module__,
            "cls": cls.__name__,
            "schema": _schema_hash(cls),
            "data": {f.name: getattr(value, f.name) for f in dataclasses.fields(value)},
        }
        return _TAG_PYDANTIC + msgspec.msgpack.encode(payload)

    try:
        return _TAG_MSGSPEC + msgspec.msgpack.encode(value)
    except TypeError as e:
        raise TypeError(
            f"Disk cache only supports JSON values, pydantic models, dataclasses, "
            f"and numpy arrays; got {type(value).__module__}.{type(value).__qualname__}"
        ) from e


def decode(data: bytes, *, allowed: set[tuple[str, str]]) -> Any:
    """Deserialize a value, restricting typed reconstruction to the allowlist."""
    if not data:
        raise DeserializationError("Empty cache entry")

    tag = data[0:1]
    body = data[1:]

    if tag == _TAG_MSGSPEC:
        try:
            return msgspec.msgpack.decode(body)
        except msgspec.DecodeError as e:
            raise DeserializationError("Invalid msgpack in cache entry") from e

    if tag == _TAG_PYDANTIC:
        try:
            payload = msgspec.msgpack.decode(body)
        except msgspec.DecodeError as e:
            raise DeserializationError("Invalid msgpack in cache entry") from e

        try:
            mod_name = payload["module"]
            cls_name = payload["cls"]
        except (KeyError, TypeError) as e:
            raise DeserializationError("Cache entry missing 'module' or 'cls' field") from e

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
            return pydantic.TypeAdapter(cls).validate_python(payload["data"])
        except (TypeError, pydantic.ValidationError, KeyError) as e:
            raise DeserializationError(f"Failed to reconstruct {mod_name}.{cls_name}") from e

    if tag == _TAG_PICKLE:
        try:
            return _RestrictedUnpickler(io.BytesIO(body)).load()
        except DeserializationError:
            raise
        except Exception as e:
            raise DeserializationError("Invalid pickle payload in cache entry") from e

    raise DeserializationError(f"Unknown routing byte: {tag!r}")


class SafeDisk(Disk):
    """Disk backend that uses safe serialization with an allowlist.

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
        if not isinstance(data, bytes):
            raise LegacyFormatError(
                f"Expected bytes from disk cache, got {type(data).__name__}; "
                f"this entry was likely written by the legacy pickle backend"
            )
        return decode(data, allowed=self._allowed)


def make_safe_disk(allowed: set[tuple[str, str]]) -> type[SafeDisk]:
    """Create a SafeDisk subclass bound to a specific allowlist."""
    return type("SafeDisk", (SafeDisk,), {"_allowed": allowed})
