"""Custom diskcache Disk backend using orjson serialization.

Replaces pickle with orjson + pydantic model_validate for safe deserialization.
Supports JSON-native types, pydantic models, and numpy arrays.
"""

import importlib
import sqlite3
from typing import Any

import orjson
import pydantic
from diskcache import Disk
from diskcache.core import MODE_RAW, UNKNOWN

_ENCODED_TYPE_KEY = "__dspy_cache_type__"
_ENCODED_MODULE_KEY = "__dspy_cache_module__"
_ENCODED_QUALNAME_KEY = "__dspy_cache_qualname__"
_ENCODED_DATA_KEY = "__dspy_cache_data__"
_ENCODED_DTYPE_KEY = "__dspy_cache_dtype__"
_PYDANTIC_TYPE = "pydantic"
_NDARRAY_TYPE = "ndarray"


def _is_ndarray(value: Any) -> bool:
    try:
        import numpy as np

        return isinstance(value, np.ndarray)
    except ImportError:
        return False


def _encode_value(value: Any) -> Any:
    if isinstance(value, pydantic.BaseModel):
        return {
            _ENCODED_TYPE_KEY: _PYDANTIC_TYPE,
            _ENCODED_MODULE_KEY: type(value).__module__,
            _ENCODED_QUALNAME_KEY: type(value).__qualname__,
            _ENCODED_DATA_KEY: value.model_dump(mode="json", exclude_none=True, by_alias=True),
        }
    if _is_ndarray(value):
        return {
            _ENCODED_TYPE_KEY: _NDARRAY_TYPE,
            _ENCODED_DTYPE_KEY: str(value.dtype),
            _ENCODED_DATA_KEY: value.tolist(),
        }
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(
        "Disk cache only supports JSON values, pydantic models, and numpy arrays; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


_PYDANTIC_ENVELOPE_KEYS = frozenset({_ENCODED_TYPE_KEY, _ENCODED_MODULE_KEY, _ENCODED_QUALNAME_KEY, _ENCODED_DATA_KEY})
_NDARRAY_ENVELOPE_KEYS = frozenset({_ENCODED_TYPE_KEY, _ENCODED_DTYPE_KEY, _ENCODED_DATA_KEY})


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        keys = value.keys()
        encoded_type = value.get(_ENCODED_TYPE_KEY)
        if encoded_type == _PYDANTIC_TYPE and keys == _PYDANTIC_ENVELOPE_KEYS:
            cls = _resolve_class(value[_ENCODED_MODULE_KEY], value[_ENCODED_QUALNAME_KEY])
            return cls.model_validate(value[_ENCODED_DATA_KEY])
        if encoded_type == _NDARRAY_TYPE and keys == _NDARRAY_ENVELOPE_KEYS:
            import numpy as np

            return np.asarray(value[_ENCODED_DATA_KEY], dtype=np.dtype(value[_ENCODED_DTYPE_KEY]))
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _resolve_class(module_name: str, qualname: str) -> type:
    module = importlib.import_module(module_name)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
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
    """

    def store(self, value, read, key=UNKNOWN):
        if not read:
            blob = orjson.dumps({"_data": _encode_value(value)})
            return len(blob), MODE_RAW, None, sqlite3.Binary(blob)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read and mode == MODE_RAW and isinstance(data, bytes):
            try:
                envelope = orjson.loads(data)
                return _decode_value(envelope["_data"])
            except (ValueError, TypeError, KeyError, ImportError, AttributeError) as e:
                raise DeserializationError(e) from e
        return data
