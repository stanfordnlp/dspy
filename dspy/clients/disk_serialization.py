"""Safe cache serialization utilities.

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
from diskcache.core import MODE_BINARY, MODE_RAW

_JSON_PREFIX = b"json:"
_NPY_PREFIX = b"npy:"
_TYPE_KEY = "type"
_DATA_KEY = "data"


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


class LegacyFormatError(DeserializationError):
    """Raised when a cache entry was written by the legacy pickle-based backend."""


class SafeTypeRegistry:
    """Registry of types that can be safely serialized in the disk cache."""

    def __init__(self):
        self._type_tags: dict[type[Any], str] = {}
        self._tag_adapters: dict[str, pydantic.TypeAdapter[Any]] = {}

    def register(self, cls: type[Any], *, tag: str | None = None) -> None:
        """Register a top-level cache value type for safe-mode roundtrips."""
        if not isinstance(cls, type):
            raise TypeError(f"Expected a type to register, got {type(cls).__name__}")
        if not (issubclass(cls, pydantic.BaseModel) or is_dataclass(cls)):
            raise TypeError(
                "Safe cache registration only supports pydantic BaseModel subclasses and dataclasses; "
                f"got {cls.__module__}.{cls.__qualname__}"
            )

        tag = tag or f"{cls.__module__}.{cls.__qualname__}"
        existing_tag = self._type_tags.get(cls)
        if existing_tag is not None and existing_tag != tag:
            raise ValueError(f"{cls.__module__}.{cls.__qualname__} is already registered as {existing_tag!r}")
        if tag in self._tag_adapters and existing_tag != tag:
            raise ValueError(f"Safe cache tag {tag!r} is already registered")

        self._type_tags[cls] = tag
        self._tag_adapters[tag] = pydantic.TypeAdapter(cls)

    def get_tag(self, cls: type[Any]) -> str | None:
        return self._type_tags.get(cls)

    def get_adapter(self, tag: str) -> pydantic.TypeAdapter[Any] | None:
        return self._tag_adapters.get(tag)


def create_default_registry() -> SafeTypeRegistry:
    """Create a registry pre-populated with the default safe types."""
    registry = SafeTypeRegistry()

    try:
        from openai.types.chat.chat_completion import ChatCompletion
        from openai.types.responses.response import Response
    except ImportError:
        pass
    else:
        for cls in (ChatCompletion, Response):
            registry.register(cls)

    try:
        from litellm.types.utils import EmbeddingResponse, ModelResponse, ModelResponseStream
    except ImportError:
        pass
    else:
        for cls in (EmbeddingResponse, ModelResponse, ModelResponseStream):
            registry.register(cls)

    return registry


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


def encode_value(value: Any, registry: SafeTypeRegistry) -> bytes:
    """Serialize *value* into the safe cache wire format."""
    if _is_ndarray(value):
        buffer = io.BytesIO()
        import numpy as np

        np.save(buffer, value, allow_pickle=False)
        return _NPY_PREFIX + buffer.getvalue()

    tag = registry.get_tag(type(value))
    if tag is None:
        if not _is_json_value(value):
            raise TypeError(
                "Disk cache only supports JSON values, registered pydantic models/dataclasses, "
                f"and numpy arrays; got {type(value).__module__}.{type(value).__qualname__}"
            )
        payload = {_TYPE_KEY: None, _DATA_KEY: value}
    else:
        adapter = registry.get_adapter(tag)
        try:
            payload = {
                _TYPE_KEY: tag,
                _DATA_KEY: adapter.dump_python(
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


def decode_value(data: bytes, registry: SafeTypeRegistry) -> Any:
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

    adapter = registry.get_adapter(type_tag)
    if adapter is None:
        raise DeserializationError(f"Unsupported cached type {type_tag!r}")

    try:
        return adapter.validate_python(payload[_DATA_KEY])
    except (pydantic.PydanticUserError, pydantic.ValidationError, TypeError, ValueError) as e:
        raise DeserializationError(f"{type_tag} failed validation") from e


class NoPickleDisk(Disk):
    """Disk backend that refuses to deserialize legacy pickle entries."""

    def fetch(self, mode, filename, value, read):
        if mode not in (MODE_RAW, MODE_BINARY):
            raise LegacyFormatError(
                f"Unsupported diskcache mode {mode}; refusing to unpickle legacy entry"
            )
        return super().fetch(mode, filename, value, read)
