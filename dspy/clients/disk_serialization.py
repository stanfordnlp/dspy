"""Custom diskcache Disk backend using orjson serialization.

Replaces pickle with orjson + pydantic model_validate for safe deserialization.
Supports JSON-native types, pydantic models, and numpy arrays.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import orjson
import pydantic
from diskcache import Disk
from diskcache.core import MODE_BINARY, MODE_RAW, UNKNOWN

if TYPE_CHECKING:
    from collections.abc import Sequence

_ENCODED_TYPE_KEY = "__dspy_cache_type__"
_ENCODED_MODULE_KEY = "__dspy_cache_module__"
_ENCODED_QUALNAME_KEY = "__dspy_cache_qualname__"
_ENCODED_DATA_KEY = "__dspy_cache_data__"
_ENCODED_DTYPE_KEY = "__dspy_cache_dtype__"
_ENVELOPE_KEY = "_data"
_PYDANTIC_TYPE = "pydantic"
_NDARRAY_TYPE = "ndarray"

DEFAULT_ALLOWED_NAMESPACES: tuple[str, ...] = ()

_REGISTERED_PYDANTIC_TYPES: frozenset[tuple[str, str]] = frozenset({
    ("litellm.types.utils", "EmbeddingResponse"),
    ("litellm.types.utils", "ModelResponse"),
    ("litellm.types.utils", "ModelResponseStream"),
    ("litellm.types.utils", "ChatCompletionMessageToolCall"),
    ("openai.types.chat.chat_completion", "ChatCompletion"),
    ("openai.types.responses.response", "Response"),
})


def _is_ndarray(value: Any) -> bool:
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:
        return False


def _encode_value(
    value: Any, allowed_namespaces: Sequence[str] = DEFAULT_ALLOWED_NAMESPACES,
) -> Any:
    """Convert *value* to a JSON-safe structure, wrapping pydantic models and ndarrays in envelopes."""
    if isinstance(value, pydantic.BaseModel):
        model_cls = type(value)
        model_key = (model_cls.__module__, model_cls.__qualname__)
        root_namespace = model_cls.__module__.split(".")[0]
        if model_key not in _REGISTERED_PYDANTIC_TYPES and root_namespace not in allowed_namespaces:
            raise TypeError(
                f"Cannot cache {model_cls.__module__}.{model_cls.__qualname__}: the type is not in "
                f"the registered cache type registry and namespace {root_namespace!r} is not in "
                f"allowed_namespaces {list(allowed_namespaces)}"
            )

        return {
            _ENCODED_TYPE_KEY: _PYDANTIC_TYPE,
            _ENCODED_MODULE_KEY: model_cls.__module__,
            _ENCODED_QUALNAME_KEY: model_cls.__qualname__,
            _ENCODED_DATA_KEY: value.model_dump(mode="json", by_alias=True, exclude_unset=True),
        }
    if _is_ndarray(value):
        return {
            _ENCODED_TYPE_KEY: _NDARRAY_TYPE,
            _ENCODED_DTYPE_KEY: str(value.dtype),
            _ENCODED_DATA_KEY: value.tolist(),
        }
    if isinstance(value, dict):
        return {k: _encode_value(v, allowed_namespaces) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_value(item, allowed_namespaces) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(
        "Disk cache only supports JSON values, pydantic models, and numpy arrays; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


_PYDANTIC_ENVELOPE_KEYS = frozenset({
    _ENCODED_TYPE_KEY, _ENCODED_MODULE_KEY, _ENCODED_QUALNAME_KEY, _ENCODED_DATA_KEY,
})
_NDARRAY_ENVELOPE_KEYS = frozenset({_ENCODED_TYPE_KEY, _ENCODED_DTYPE_KEY, _ENCODED_DATA_KEY})


def _decode_value(
    value: Any, allowed_namespaces: Sequence[str] = DEFAULT_ALLOWED_NAMESPACES,
) -> Any:
    """Reconstruct Python objects from a JSON structure produced by ``_encode_value``."""
    if isinstance(value, dict):
        keys = value.keys()
        encoded_type = value.get(_ENCODED_TYPE_KEY)
        if encoded_type == _PYDANTIC_TYPE and keys == _PYDANTIC_ENVELOPE_KEYS:
            cls = _resolve_class(
                value[_ENCODED_MODULE_KEY], value[_ENCODED_QUALNAME_KEY], allowed_namespaces,
            )
            try:
                return cls.model_validate(value[_ENCODED_DATA_KEY])
            except pydantic.ValidationError as e:
                raise DeserializationError(
                    f"{value[_ENCODED_MODULE_KEY]}.{value[_ENCODED_QUALNAME_KEY]} failed validation"
                ) from e
        if encoded_type == _NDARRAY_TYPE and keys == _NDARRAY_ENVELOPE_KEYS:
            try:
                import numpy as np
            except ImportError as e:
                raise DeserializationError("Cannot import module 'numpy'") from e

            try:
                dtype = np.dtype(value[_ENCODED_DTYPE_KEY])
                return np.asarray(value[_ENCODED_DATA_KEY], dtype=dtype)
            except (TypeError, ValueError) as e:
                raise DeserializationError("Invalid ndarray payload in cache entry") from e
        return {k: _decode_value(v, allowed_namespaces) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(item, allowed_namespaces) for item in value]
    return value


def _resolve_class(module_name: str, qualname: str, allowed_namespaces: Sequence[str]) -> type:
    """Import and return the pydantic BaseModel class at *module_name*.*qualname*."""
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise DeserializationError("Encoded pydantic type metadata must be strings")

    if (module_name, qualname) not in _REGISTERED_PYDANTIC_TYPES:
        root_namespace = module_name.split(".")[0]
        if root_namespace not in allowed_namespaces:
            raise DeserializationError(
                f"{module_name}.{qualname} is not in the registered cache type registry and "
                f"namespace {root_namespace!r} is not in allowed_namespaces {list(allowed_namespaces)}"
            )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise DeserializationError(f"Cannot import module {module_name!r}") from e
    obj = module
    for attr in qualname.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as e:
            raise DeserializationError(f"{module_name}.{qualname} could not be resolved") from e
    if not (isinstance(obj, type) and issubclass(obj, pydantic.BaseModel)):
        raise DeserializationError(
            f"{module_name}.{qualname} is not a pydantic BaseModel subclass"
        )
    return obj


class DeserializationError(Exception):
    """Raised when a cached value cannot be deserialized."""


class OrjsonDisk(Disk):
    """Disk backend that serializes values with orjson instead of pickle.

    Handles pydantic models, numpy arrays, and JSON-native types.
    Raises TypeError for unsupported types rather than falling back to pickle.

    Built-in cache response types are resolved from a small explicit registry.
    Pass `disk_allowed_namespaces` as a comma-separated string to `FanoutCache`
    to opt additional top-level module names into dynamic pydantic imports.
    """

    def __init__(self, directory, allowed_namespaces=None, **kwargs):
        super().__init__(directory, **kwargs)
        if allowed_namespaces is None:
            self._allowed_namespaces = DEFAULT_ALLOWED_NAMESPACES
        elif isinstance(allowed_namespaces, str):
            self._allowed_namespaces = tuple(
                namespace.strip() for namespace in allowed_namespaces.split(",") if namespace.strip()
            )
        else:
            raise TypeError(
                f"allowed_namespaces must be a comma-separated string, "
                f"got {type(allowed_namespaces).__name__}"
            )

    def store(self, value, read, key=UNKNOWN):
        """Serialize *value* to an orjson blob and return fields for the Cache table."""
        if not read:
            blob = orjson.dumps({_ENVELOPE_KEY: _encode_value(value, self._allowed_namespaces)})
            return super().store(blob, False, key=key)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        """Deserialize a previously-stored orjson blob back into a Python object."""
        if mode not in (MODE_RAW, MODE_BINARY):
            raise DeserializationError(
                f"Unsupported diskcache mode {mode} for OrjsonDisk entry; refusing legacy format"
            )

        data = super().fetch(mode, filename, value, read)
        if read:
            return data

        if not isinstance(data, bytes):
            raise DeserializationError(
                f"Expected orjson-encoded bytes for OrjsonDisk entry, got {type(data).__name__}"
            )

        try:
            envelope = orjson.loads(data)
        except orjson.JSONDecodeError as e:
            raise DeserializationError("Invalid orjson payload in cache entry") from e

        if not isinstance(envelope, dict):
            raise DeserializationError(
                f"Expected top-level cache envelope to be a dict, got {type(envelope).__name__}"
            )

        try:
            payload = envelope[_ENVELOPE_KEY]
        except KeyError as e:
            raise DeserializationError(f"Missing {_ENVELOPE_KEY!r} in cache entry envelope") from e

        try:
            return _decode_value(payload, self._allowed_namespaces)
        except RecursionError as e:
            raise DeserializationError("Cache entry exceeded recursion limit during deserialization") from e
