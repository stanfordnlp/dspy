"""Custom diskcache Disk backend for safe cache serialization.

Supports JSON-native values, explicitly registered pydantic/dataclass types,
and numpy arrays stored as .npy payloads.
"""

from __future__ import annotations

import io
from dataclasses import is_dataclass
from typing import Any

import orjson
import pydantic
from diskcache import Disk
from diskcache.core import MODE_BINARY, MODE_RAW, UNKNOWN
from litellm.types.utils import EmbeddingResponse, ModelResponse, ModelResponseStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses.response import Response

_JSON_PREFIX = b"json:"
_NPY_PREFIX = b"npy:"
_TYPE_KEY = "type"
_DATA_KEY = "data"

_TYPE_TAGS: dict[type[Any], str] = {}
_TAG_ADAPTERS: dict[str, pydantic.TypeAdapter[Any]] = {}


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


def register_safe_type(cls: type[Any], *, tag: str | None = None) -> None:
    """Register a top-level cache value type for safe-mode roundtrips."""
    if not isinstance(cls, type):
        raise TypeError(f"Expected a type to register, got {type(cls).__name__}")
    if not (issubclass(cls, pydantic.BaseModel) or is_dataclass(cls)):
        raise TypeError(
            "Safe cache registration only supports pydantic BaseModel subclasses and dataclasses; "
            f"got {cls.__module__}.{cls.__qualname__}"
        )

    tag = tag or f"{cls.__module__}.{cls.__qualname__}"
    existing_tag = _TYPE_TAGS.get(cls)
    if existing_tag is not None and existing_tag != tag:
        raise ValueError(f"{cls.__module__}.{cls.__qualname__} is already registered as {existing_tag!r}")
    if tag in _TAG_ADAPTERS and existing_tag != tag:
        raise ValueError(f"Safe cache tag {tag!r} is already registered")

    _TYPE_TAGS[cls] = tag
    _TAG_ADAPTERS[tag] = pydantic.TypeAdapter(cls)


def _is_ndarray(value: Any) -> bool:
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:
        return False


def _is_json_value(value: Any) -> bool:
    if isinstance(value, str | int | float | bool) or value is None:
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _encode_value(value: Any) -> bytes:
    """Serialize *value* into the safe cache wire format."""
    if _is_ndarray(value):
        buffer = io.BytesIO()
        import numpy as np

        np.save(buffer, value, allow_pickle=False)
        return _NPY_PREFIX + buffer.getvalue()

    tag = _TYPE_TAGS.get(type(value))
    if tag is None:
        if not _is_json_value(value):
            raise TypeError(
                "Disk cache only supports JSON values, registered pydantic models/dataclasses, "
                f"and numpy arrays; got {type(value).__module__}.{type(value).__qualname__}"
            )
        payload = {_TYPE_KEY: None, _DATA_KEY: value}
    else:
        try:
            payload = {
                _TYPE_KEY: tag,
                _DATA_KEY: _TAG_ADAPTERS[tag].dump_python(
                    value,
                    mode="json",
                    by_alias=True,
                    exclude_unset=True,
                ),
            }
        except Exception as e:
            raise TypeError(f"Cannot cache {tag}: {e}") from e

    try:
        return _JSON_PREFIX + orjson.dumps(payload)
    except orjson.JSONEncodeError as e:
        raise TypeError(
            "Disk cache only supports JSON values, registered pydantic models/dataclasses, "
            f"and numpy arrays; got {type(value).__module__}.{type(value).__qualname__}"
        ) from e


def _decode_value(data: bytes) -> Any:
    """Deserialize *data* from the safe cache wire format."""
    if data.startswith(_NPY_PREFIX):
        try:
            import numpy as np
        except ImportError as e:
            raise DeserializationError("Cannot import module 'numpy'") from e

        try:
            return np.load(io.BytesIO(data[len(_NPY_PREFIX):]), allow_pickle=False)
        except (TypeError, ValueError) as e:
            raise DeserializationError("Invalid ndarray payload in cache entry") from e

    if not data.startswith(_JSON_PREFIX):
        raise DeserializationError("Unknown cache payload format")

    try:
        payload = orjson.loads(data[len(_JSON_PREFIX):])
    except orjson.JSONDecodeError as e:
        raise DeserializationError("Invalid orjson payload in cache entry") from e

    if not isinstance(payload, dict):
        raise DeserializationError(
            f"Expected top-level cache payload to be a dict, got {type(payload).__name__}"
        )
    if _TYPE_KEY not in payload or _DATA_KEY not in payload:
        raise DeserializationError("Missing type metadata in cache payload")

    type_tag = payload[_TYPE_KEY]
    if type_tag is None:
        return payload[_DATA_KEY]
    if not isinstance(type_tag, str):
        raise DeserializationError("Encoded cache type metadata must be a string or null")

    adapter = _TAG_ADAPTERS.get(type_tag)
    if adapter is None:
        raise DeserializationError(f"Unsupported cached type {type_tag!r}")

    try:
        return adapter.validate_python(payload[_DATA_KEY])
    except (pydantic.PydanticUserError, pydantic.ValidationError, TypeError, ValueError) as e:
        raise DeserializationError(f"{type_tag} failed validation") from e


class OrjsonDisk(Disk):
    """Disk backend that serializes values with the safe cache format."""

    def store(self, value, read, key=UNKNOWN):
        """Serialize *value* and return fields for the Cache table."""
        if not read:
            return super().store(_encode_value(value), False, key=key)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        """Deserialize a previously-stored safe cache payload."""
        if mode not in (MODE_RAW, MODE_BINARY):
            raise DeserializationError(
                f"Unsupported diskcache mode {mode} for OrjsonDisk entry; refusing legacy format"
            )

        data = super().fetch(mode, filename, value, read)
        if read:
            return data
        if not isinstance(data, bytes):
            raise DeserializationError(
                f"Expected safe cache bytes for OrjsonDisk entry, got {type(data).__name__}"
            )
        return _decode_value(data)


for _safe_type in (EmbeddingResponse, ModelResponse, ModelResponseStream, ChatCompletion, Response):
    register_safe_type(_safe_type)
