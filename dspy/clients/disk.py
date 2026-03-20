"""Custom diskcache Disk backend using orjson serialization.

Replaces pickle with orjson + pydantic model_validate for safe deserialization.
Supports JSON-native types, pydantic models, and numpy arrays.
"""

import importlib
import sqlite3
from typing import Any

import numpy as np
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


def _encode_value(value: Any) -> Any:
    if isinstance(value, pydantic.BaseModel):
        return {
            _ENCODED_TYPE_KEY: _PYDANTIC_TYPE,
            _ENCODED_MODULE_KEY: type(value).__module__,
            _ENCODED_QUALNAME_KEY: type(value).__qualname__,
            _ENCODED_DATA_KEY: {k: _encode_value(v) for k, v in _pydantic_to_dict(value).items()},
        }
    if isinstance(value, np.ndarray):
        return {
            _ENCODED_TYPE_KEY: _NDARRAY_TYPE,
            _ENCODED_DTYPE_KEY: str(value.dtype),
            _ENCODED_DATA_KEY: value.tolist(),
        }
    if isinstance(value, tuple):
        raise TypeError(
            "SQLite disk cache only supports JSON values, pydantic models, and numpy arrays; "
            f"got {type(value).__module__}.{type(value).__qualname__}"
        )
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "SQLite disk cache only supports JSON values, pydantic models, and numpy arrays; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        encoded_type = value.get(_ENCODED_TYPE_KEY)
        if encoded_type == _PYDANTIC_TYPE:
            cls = _resolve_class(value[_ENCODED_MODULE_KEY], value[_ENCODED_QUALNAME_KEY])
            return cls.model_validate(_decode_value(value[_ENCODED_DATA_KEY]))
        if encoded_type == _NDARRAY_TYPE:
            return np.asarray(_decode_value(value[_ENCODED_DATA_KEY]), dtype=np.dtype(value[_ENCODED_DTYPE_KEY]))
        return {k: _decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _resolve_class(module_name: str, qualname: str) -> type:
    module = importlib.import_module(module_name)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _pydantic_to_dict(value: pydantic.BaseModel) -> dict[str, Any]:
    data = {}
    for name, field in type(value).model_fields.items():
        if name not in value.__dict__:
            continue
        key = field.serialization_alias or field.alias or name
        data[key] = value.__dict__[name]

    extra = getattr(value, "__pydantic_extra__", None)
    if extra:
        for key, item in extra.items():
            data.setdefault(key, item)
    return data


class DSPyDisk(Disk):
    """Disk backend that serializes values with orjson instead of pickle.

    Handles pydantic models, numpy arrays, and JSON-native types.
    Raises TypeError for unsupported types rather than falling back to pickle.
    """

    def store(self, value, read, key=UNKNOWN):
        if not read:
            blob = orjson.dumps({"_data": _encode_value(value)})
            return 0, MODE_RAW, None, sqlite3.Binary(blob)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read and mode == MODE_RAW and isinstance(data, bytes):
            envelope = orjson.loads(data)
            return _decode_value(envelope["_data"])
        return data
