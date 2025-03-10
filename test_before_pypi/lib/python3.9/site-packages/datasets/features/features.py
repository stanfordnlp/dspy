# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""This class handle features definition in datasets and some utilities to display table type."""

import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype

from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import experimental, logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages


logger = logging.get_logger(__name__)


def _arrow_to_datasets_dtype(arrow_type: pa.DataType) -> str:
    """
    _arrow_to_datasets_dtype takes a pyarrow.DataType and converts it to a datasets string dtype.
    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`
    """
    if pyarrow.types.is_null(arrow_type):
        return "null"
    elif pyarrow.types.is_boolean(arrow_type):
        return "bool"
    elif pyarrow.types.is_int8(arrow_type):
        return "int8"
    elif pyarrow.types.is_int16(arrow_type):
        return "int16"
    elif pyarrow.types.is_int32(arrow_type):
        return "int32"
    elif pyarrow.types.is_int64(arrow_type):
        return "int64"
    elif pyarrow.types.is_uint8(arrow_type):
        return "uint8"
    elif pyarrow.types.is_uint16(arrow_type):
        return "uint16"
    elif pyarrow.types.is_uint32(arrow_type):
        return "uint32"
    elif pyarrow.types.is_uint64(arrow_type):
        return "uint64"
    elif pyarrow.types.is_float16(arrow_type):
        return "float16"  # pyarrow dtype is "halffloat"
    elif pyarrow.types.is_float32(arrow_type):
        return "float32"  # pyarrow dtype is "float"
    elif pyarrow.types.is_float64(arrow_type):
        return "float64"  # pyarrow dtype is "double"
    elif pyarrow.types.is_time32(arrow_type):
        return f"time32[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pyarrow.types.is_time64(arrow_type):
        return f"time64[{pa.type_for_alias(str(arrow_type)).unit}]"
    elif pyarrow.types.is_timestamp(arrow_type):
        if arrow_type.tz is None:
            return f"timestamp[{arrow_type.unit}]"
        elif arrow_type.tz:
            return f"timestamp[{arrow_type.unit}, tz={arrow_type.tz}]"
        else:
            raise ValueError(f"Unexpected timestamp object {arrow_type}.")
    elif pyarrow.types.is_date32(arrow_type):
        return "date32"  # pyarrow dtype is "date32[day]"
    elif pyarrow.types.is_date64(arrow_type):
        return "date64"  # pyarrow dtype is "date64[ms]"
    elif pyarrow.types.is_duration(arrow_type):
        return f"duration[{arrow_type.unit}]"
    elif pyarrow.types.is_decimal128(arrow_type):
        return f"decimal128({arrow_type.precision}, {arrow_type.scale})"
    elif pyarrow.types.is_decimal256(arrow_type):
        return f"decimal256({arrow_type.precision}, {arrow_type.scale})"
    elif pyarrow.types.is_binary(arrow_type):
        return "binary"
    elif pyarrow.types.is_large_binary(arrow_type):
        return "large_binary"
    elif pyarrow.types.is_string(arrow_type):
        return "string"
    elif pyarrow.types.is_large_string(arrow_type):
        return "large_string"
    elif pyarrow.types.is_dictionary(arrow_type):
        return _arrow_to_datasets_dtype(arrow_type.value_type)
    else:
        raise ValueError(f"Arrow type {arrow_type} does not have a datasets dtype equivalent.")


def string_to_arrow(datasets_dtype: str) -> pa.DataType:
    """
    string_to_arrow takes a datasets string dtype and converts it to a pyarrow.DataType.

    In effect, `dt == string_to_arrow(_arrow_to_datasets_dtype(dt))`

    This is necessary because the datasets.Value() primitive type is constructed using a string dtype

    Value(dtype=str)

    But Features.type (via `get_nested_type()` expects to resolve Features into a pyarrow Schema,
        which means that each Value() must be able to resolve into a corresponding pyarrow.DataType, which is the
        purpose of this function.
    """

    def _dtype_error_msg(dtype, pa_dtype, examples=None, urls=None):
        msg = f"{dtype} is not a validly formatted string representation of the pyarrow {pa_dtype} type."
        if examples:
            examples = ", ".join(examples[:-1]) + " or " + examples[-1] if len(examples) > 1 else examples[0]
            msg += f"\nValid examples include: {examples}."
        if urls:
            urls = ", ".join(urls[:-1]) + " and " + urls[-1] if len(urls) > 1 else urls[0]
            msg += f"\nFor more insformation, see: {urls}."
        return msg

    if datasets_dtype in pa.__dict__:
        return pa.__dict__[datasets_dtype]()

    if (datasets_dtype + "_") in pa.__dict__:
        return pa.__dict__[datasets_dtype + "_"]()

    timestamp_matches = re.search(r"^timestamp\[(.*)\]$", datasets_dtype)
    if timestamp_matches:
        timestamp_internals = timestamp_matches.group(1)
        internals_matches = re.search(r"^(s|ms|us|ns),\s*tz=([a-zA-Z0-9/_+\-:]*)$", timestamp_internals)
        if timestamp_internals in ["s", "ms", "us", "ns"]:
            return pa.timestamp(timestamp_internals)
        elif internals_matches:
            return pa.timestamp(internals_matches.group(1), internals_matches.group(2))
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "timestamp",
                    examples=["timestamp[us]", "timestamp[us, tz=America/New_York"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.timestamp.html"],
                )
            )

    duration_matches = re.search(r"^duration\[(.*)\]$", datasets_dtype)
    if duration_matches:
        duration_internals = duration_matches.group(1)
        if duration_internals in ["s", "ms", "us", "ns"]:
            return pa.duration(duration_internals)
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "duration",
                    examples=["duration[s]", "duration[us]"],
                    urls=["https://arrow.apache.org/docs/python/generated/pyarrow.duration.html"],
                )
            )

    time_matches = re.search(r"^time(.*)\[(.*)\]$", datasets_dtype)
    if time_matches:
        time_internals_bits = time_matches.group(1)
        if time_internals_bits == "32":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["s", "ms"]:
                return pa.time32(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time32 type. Supported units: s (second) and ms (millisecond)."
                )
        elif time_internals_bits == "64":
            time_internals_unit = time_matches.group(2)
            if time_internals_unit in ["us", "ns"]:
                return pa.time64(time_internals_unit)
            else:
                raise ValueError(
                    f"{time_internals_unit} is not a valid unit for the pyarrow time64 type. Supported units: us (microsecond) and ns (nanosecond)."
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "time",
                    examples=["time32[s]", "time64[us]"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time32.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.time64.html",
                    ],
                )
            )

    decimal_matches = re.search(r"^decimal(.*)\((.*)\)$", datasets_dtype)
    if decimal_matches:
        decimal_internals_bits = decimal_matches.group(1)
        if decimal_internals_bits == "128":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal128(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal128",
                        examples=["decimal128(10, 2)", "decimal128(4, -2)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html"],
                    )
                )
        elif decimal_internals_bits == "256":
            decimal_internals_precision_and_scale = re.search(r"^(\d+),\s*(-?\d+)$", decimal_matches.group(2))
            if decimal_internals_precision_and_scale:
                precision = decimal_internals_precision_and_scale.group(1)
                scale = decimal_internals_precision_and_scale.group(2)
                return pa.decimal256(int(precision), int(scale))
            else:
                raise ValueError(
                    _dtype_error_msg(
                        datasets_dtype,
                        "decimal256",
                        examples=["decimal256(30, 2)", "decimal256(38, -4)"],
                        urls=["https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html"],
                    )
                )
        else:
            raise ValueError(
                _dtype_error_msg(
                    datasets_dtype,
                    "decimal",
                    examples=["decimal128(12, 3)", "decimal256(40, 6)"],
                    urls=[
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal128.html",
                        "https://arrow.apache.org/docs/python/generated/pyarrow.decimal256.html",
                    ],
                )
            )

    raise ValueError(
        f"Neither {datasets_dtype} nor {datasets_dtype + '_'} seems to be a pyarrow data type. "
        f"Please make sure to use a correct data type, see: "
        f"https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions"
    )


def _cast_to_python_objects(obj: Any, only_1d_for_numpy: bool, optimize_list_casting: bool) -> Tuple[Any, bool]:
    """
    Cast pytorch/tensorflow/pandas objects to python numpy array/lists.
    It works recursively.

    If `optimize_list_casting` is True, to avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast.
        only_1d_for_numpy (bool): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
        has_changed (bool): True if the object has been changed, False if it is identical
    """

    if config.TF_AVAILABLE and "tensorflow" in sys.modules:
        import tensorflow as tf

    if config.TORCH_AVAILABLE and "torch" in sys.modules:
        import torch

    if config.JAX_AVAILABLE and "jax" in sys.modules:
        import jax.numpy as jnp

    if config.PIL_AVAILABLE and "PIL" in sys.modules:
        import PIL.Image

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj, False
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj
                ],
                True,
            )
    elif config.TORCH_AVAILABLE and "torch" in sys.modules and isinstance(obj, torch.Tensor):
        if obj.dtype == torch.bfloat16:
            return _cast_to_python_objects(
                obj.detach().to(torch.float).cpu().numpy(),
                only_1d_for_numpy=only_1d_for_numpy,
                optimize_list_casting=optimize_list_casting,
            )[0], True
        if obj.ndim == 0:
            return obj.detach().cpu().numpy()[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj.detach().cpu().numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj.detach().cpu().numpy()
                ],
                True,
            )
    elif config.TF_AVAILABLE and "tensorflow" in sys.modules and isinstance(obj, tf.Tensor):
        if obj.ndim == 0:
            return obj.numpy()[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return obj.numpy(), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in obj.numpy()
                ],
                True,
            )
    elif config.JAX_AVAILABLE and "jax" in sys.modules and isinstance(obj, jnp.ndarray):
        if obj.ndim == 0:
            return np.asarray(obj)[()], True
        elif not only_1d_for_numpy or obj.ndim == 1:
            return np.asarray(obj), True
        else:
            return (
                [
                    _cast_to_python_objects(
                        x, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                    )[0]
                    for x in np.asarray(obj)
                ],
                True,
            )
    elif config.PIL_AVAILABLE and "PIL" in sys.modules and isinstance(obj, PIL.Image.Image):
        return encode_pil_image(obj), True
    elif isinstance(obj, pd.Series):
        return (
            _cast_to_python_objects(
                obj.tolist(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )[0],
            True,
        )
    elif isinstance(obj, pd.DataFrame):
        return (
            {
                key: _cast_to_python_objects(
                    value, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                )[0]
                for key, value in obj.to_dict("series").items()
            },
            True,
        )
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime(), True
    elif isinstance(obj, pd.Timedelta):
        return obj.to_pytimedelta(), True
    elif isinstance(obj, Mapping):
        has_changed = not isinstance(obj, dict)
        output = {}
        for k, v in obj.items():
            casted_v, has_changed_v = _cast_to_python_objects(
                v, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )
            has_changed |= has_changed_v
            output[k] = casted_v
        return output if has_changed else obj, has_changed
    elif hasattr(obj, "__array__"):
        return (
            _cast_to_python_objects(
                obj.__array__(), only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )[0],
            True,
        )
    elif isinstance(obj, (list, tuple)):
        if len(obj) > 0:
            for first_elmt in obj:
                if _check_non_null_non_empty_recursive(first_elmt):
                    break
            casted_first_elmt, has_changed_first_elmt = _cast_to_python_objects(
                first_elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
            )
            if has_changed_first_elmt or not optimize_list_casting:
                return (
                    [
                        _cast_to_python_objects(
                            elmt, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
                        )[0]
                        for elmt in obj
                    ],
                    True,
                )
            else:
                if isinstance(obj, (list, tuple)):
                    return obj, False
                else:
                    return list(obj), True
        else:
            return obj, False
    else:
        return obj, False


def cast_to_python_objects(obj: Any, only_1d_for_numpy=False, optimize_list_casting=True) -> Any:
    """
    Cast numpy/pytorch/tensorflow/pandas objects to python lists.
    It works recursively.

    If `optimize_list_casting` is True, To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be casted.
    If the first element needs to be casted, then all the elements of the list will be casted, otherwise they'll stay the same.
    This trick allows to cast objects that contain tokenizers outputs without iterating over every single token for example.

    Args:
        obj: the object (nested struct) to cast
        only_1d_for_numpy (bool, default ``False``): whether to keep the full multi-dim tensors as multi-dim numpy arrays, or convert them to
            nested lists of 1-dimensional numpy arrays. This can be useful to keep only 1-d arrays to instantiate Arrow arrays.
            Indeed Arrow only support converting 1-dimensional array values.
        optimize_list_casting (bool, default ``True``): whether to optimize list casting by checking the first non-null element to see if it needs to be casted
            and if it doesn't, not checking the rest of the list elements.

    Returns:
        casted_obj: the casted object
    """
    return _cast_to_python_objects(
        obj, only_1d_for_numpy=only_1d_for_numpy, optimize_list_casting=optimize_list_casting
    )[0]


@dataclass
class Value:
    """
    The `Value` dtypes are as follows:

    - `null`
    - `bool`
    - `int8`
    - `int16`
    - `int32`
    - `int64`
    - `uint8`
    - `uint16`
    - `uint32`
    - `uint64`
    - `float16`
    - `float32` (alias float)
    - `float64` (alias double)
    - `time32[(s|ms)]`
    - `time64[(us|ns)]`
    - `timestamp[(s|ms|us|ns)]`
    - `timestamp[(s|ms|us|ns), tz=(tzstring)]`
    - `date32`
    - `date64`
    - `duration[(s|ms|us|ns)]`
    - `decimal128(precision, scale)`
    - `decimal256(precision, scale)`
    - `binary`
    - `large_binary`
    - `string`
    - `large_string`

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'stars': Value(dtype='int32')})
    >>> features
    {'stars': Value(dtype='int32', id=None)}
    ```
    """

    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Value", init=False, repr=False)

    def __post_init__(self):
        if self.dtype == "double":  # fix inferred type
            self.dtype = "float64"
        if self.dtype == "float":  # fix inferred type
            self.dtype = "float32"
        self.pa_type = string_to_arrow(self.dtype)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value):
        if pa.types.is_boolean(self.pa_type):
            return bool(value)
        elif pa.types.is_integer(self.pa_type):
            return int(value)
        elif pa.types.is_floating(self.pa_type):
            return float(value)
        elif pa.types.is_string(self.pa_type):
            return str(value)
        else:
            return value


class _ArrayXD:
    def __post_init__(self):
        self.shape = tuple(self.shape)

    def __call__(self):
        pa_type = globals()[self.__class__.__name__ + "ExtensionType"](self.shape, self.dtype)
        return pa_type

    def encode_example(self, value):
        return value


@dataclass
class Array2D(_ArrayXD):
    """Create a two-dimensional array.

    Args:
        shape (`tuple`):
            The size of each dimension.
        dtype (`str`):
            The value of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array2D(shape=(1, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array2D", init=False, repr=False)


@dataclass
class Array3D(_ArrayXD):
    """Create a three-dimensional array.

    Args:
        shape (`tuple`):
            The size of each dimension.
        dtype (`str`):
            The value of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array3D(shape=(1, 2, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array3D", init=False, repr=False)


@dataclass
class Array4D(_ArrayXD):
    """Create a four-dimensional array.

    Args:
        shape (`tuple`):
            The size of each dimension.
        dtype (`str`):
            The value of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array4D(shape=(1, 2, 2, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array4D", init=False, repr=False)


@dataclass
class Array5D(_ArrayXD):
    """Create a five-dimensional array.

    Args:
        shape (`tuple`):
            The size of each dimension.
        dtype (`str`):
            The value of the data type.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'x': Array5D(shape=(1, 2, 2, 3, 3), dtype='int32')})
    ```
    """

    shape: tuple
    dtype: str
    id: Optional[str] = None
    # Automatically constructed
    _type: str = field(default="Array5D", init=False, repr=False)


class _ArrayXDExtensionType(pa.ExtensionType):
    ndims: Optional[int] = None

    def __init__(self, shape: tuple, dtype: str):
        if self.ndims is None or self.ndims <= 1:
            raise ValueError("You must instantiate an array type with a value for dim that is > 1")
        if len(shape) != self.ndims:
            raise ValueError(f"shape={shape} and ndims={self.ndims} don't match")
        for dim in range(1, self.ndims):
            if shape[dim] is None:
                raise ValueError(f"Support only dynamic size on first dimension. Got: {shape}")
        self.shape = tuple(shape)
        self.value_type = dtype
        self.storage_dtype = self._generate_dtype(self.value_type)
        pa.ExtensionType.__init__(self, self.storage_dtype, f"{self.__class__.__module__}.{self.__class__.__name__}")

    def __arrow_ext_serialize__(self):
        return json.dumps((self.shape, self.value_type)).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        args = json.loads(serialized)
        return cls(*args)

    # This was added to pa.ExtensionType in pyarrow >= 13.0.0
    def __reduce__(self):
        return self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__())

    def __hash__(self):
        return hash((self.__class__, self.shape, self.value_type))

    def __arrow_ext_class__(self):
        return ArrayExtensionArray

    def _generate_dtype(self, dtype):
        dtype = string_to_arrow(dtype)
        for d in reversed(self.shape):
            dtype = pa.list_(dtype)
            # Don't specify the size of the list, since fixed length list arrays have issues
            # being validated after slicing in pyarrow 0.17.1
        return dtype

    def to_pandas_dtype(self):
        return PandasArrayExtensionDtype(self.value_type)


class Array2DExtensionType(_ArrayXDExtensionType):
    ndims = 2


class Array3DExtensionType(_ArrayXDExtensionType):
    ndims = 3


class Array4DExtensionType(_ArrayXDExtensionType):
    ndims = 4


class Array5DExtensionType(_ArrayXDExtensionType):
    ndims = 5


# Register the extension types for deserialization
pa.register_extension_type(Array2DExtensionType((1, 2), "int64"))
pa.register_extension_type(Array3DExtensionType((1, 2, 3), "int64"))
pa.register_extension_type(Array4DExtensionType((1, 2, 3, 4), "int64"))
pa.register_extension_type(Array5DExtensionType((1, 2, 3, 4, 5), "int64"))


def _is_zero_copy_only(pa_type: pa.DataType, unnest: bool = False) -> bool:
    """
    When converting a pyarrow array to a numpy array, we must know whether this could be done in zero-copy or not.
    This function returns the value of the ``zero_copy_only`` parameter to pass to ``.to_numpy()``, given the type of the pyarrow array.

    # zero copy is available for all primitive types except booleans and temporal types (date, time, timestamp or duration)
    # primitive types are types for which the physical representation in arrow and in numpy
    # https://github.com/wesm/arrow/blob/c07b9b48cf3e0bbbab493992a492ae47e5b04cad/python/pyarrow/types.pxi#L821
    # see https://arrow.apache.org/docs/python/generated/pyarrow.Array.html#pyarrow.Array.to_numpy
    # and https://issues.apache.org/jira/browse/ARROW-2871?jql=text%20~%20%22boolean%20to_numpy%22
    """

    def _unnest_pa_type(pa_type: pa.DataType) -> pa.DataType:
        if pa.types.is_list(pa_type):
            return _unnest_pa_type(pa_type.value_type)
        return pa_type

    if unnest:
        pa_type = _unnest_pa_type(pa_type)
    return pa.types.is_primitive(pa_type) and not (pa.types.is_boolean(pa_type) or pa.types.is_temporal(pa_type))


class ArrayExtensionArray(pa.ExtensionArray):
    def __array__(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        return self.to_numpy(zero_copy_only=zero_copy_only)

    def __getitem__(self, i):
        return self.storage[i]

    def to_numpy(self, zero_copy_only=True):
        storage: pa.ListArray = self.storage
        null_mask = storage.is_null().to_numpy(zero_copy_only=False)

        if self.type.shape[0] is not None:
            size = 1
            null_indices = np.arange(len(storage))[null_mask] - np.arange(np.sum(null_mask))

            for i in range(self.type.ndims):
                size *= self.type.shape[i]
                storage = storage.flatten()
            numpy_arr = storage.to_numpy(zero_copy_only=zero_copy_only)
            numpy_arr = numpy_arr.reshape(len(self) - len(null_indices), *self.type.shape)

            if len(null_indices):
                numpy_arr = np.insert(numpy_arr.astype(np.float64), null_indices, np.nan, axis=0)

        else:
            shape = self.type.shape
            ndims = self.type.ndims
            arrays = []
            first_dim_offsets = np.array([off.as_py() for off in storage.offsets])
            for i, is_null in enumerate(null_mask):
                if is_null:
                    arrays.append(np.nan)
                else:
                    storage_el = storage[i : i + 1]
                    first_dim = first_dim_offsets[i + 1] - first_dim_offsets[i]
                    # flatten storage
                    for _ in range(ndims):
                        storage_el = storage_el.flatten()

                    numpy_arr = storage_el.to_numpy(zero_copy_only=zero_copy_only)
                    arrays.append(numpy_arr.reshape(first_dim, *shape[1:]))

            if len(np.unique(np.diff(first_dim_offsets))) > 1:
                # ragged
                numpy_arr = np.empty(len(arrays), dtype=object)
                numpy_arr[:] = arrays
            else:
                numpy_arr = np.array(arrays)

        return numpy_arr

    def to_pylist(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        numpy_arr = self.to_numpy(zero_copy_only=zero_copy_only)
        if self.type.shape[0] is None and numpy_arr.dtype == object:
            return [arr.tolist() for arr in numpy_arr.tolist()]
        else:
            return numpy_arr.tolist()


class PandasArrayExtensionDtype(PandasExtensionDtype):
    _metadata = "value_type"

    def __init__(self, value_type: Union["PandasArrayExtensionDtype", np.dtype]):
        self._value_type = value_type

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
        if isinstance(array, pa.ChunkedArray):
            array = array.type.wrap_array(pa.concat_arrays([chunk.storage for chunk in array.chunks]))
        zero_copy_only = _is_zero_copy_only(array.storage.type, unnest=True)
        numpy_arr = array.to_numpy(zero_copy_only=zero_copy_only)
        return PandasArrayExtensionArray(numpy_arr)

    @classmethod
    def construct_array_type(cls):
        return PandasArrayExtensionArray

    @property
    def type(self) -> type:
        return np.ndarray

    @property
    def kind(self) -> str:
        return "O"

    @property
    def name(self) -> str:
        return f"array[{self.value_type}]"

    @property
    def value_type(self) -> np.dtype:
        return self._value_type


class PandasArrayExtensionArray(PandasExtensionArray):
    def __init__(self, data: np.ndarray, copy: bool = False):
        self._data = data if not copy else np.array(data)
        self._dtype = PandasArrayExtensionDtype(data.dtype)

    def __array__(self, dtype=None):
        """
        Convert to NumPy Array.
        Note that Pandas expects a 1D array when dtype is set to object.
        But for other dtypes, the returned shape is the same as the one of ``data``.

        More info about pandas 1D requirement for PandasExtensionArray here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

        """
        if dtype == np.dtype(object):
            out = np.empty(len(self._data), dtype=object)
            for i in range(len(self._data)):
                out[i] = self._data[i]
            return out
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype)

    def copy(self, deep: bool = False) -> "PandasArrayExtensionArray":
        return PandasArrayExtensionArray(self._data, copy=True)

    @classmethod
    def _from_sequence(
        cls, scalars, dtype: Optional[PandasArrayExtensionDtype] = None, copy: bool = False
    ) -> "PandasArrayExtensionArray":
        if len(scalars) > 1 and all(
            isinstance(x, np.ndarray) and x.shape == scalars[0].shape and x.dtype == scalars[0].dtype for x in scalars
        ):
            data = np.array(scalars, dtype=dtype if dtype is None else dtype.value_type, copy=copy)
        else:
            data = np.empty(len(scalars), dtype=object)
            data[:] = scalars
        return cls(data, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence_["PandasArrayExtensionArray"]) -> "PandasArrayExtensionArray":
        if len(to_concat) > 1 and all(
            va._data.shape == to_concat[0]._data.shape and va._data.dtype == to_concat[0]._data.dtype
            for va in to_concat
        ):
            data = np.vstack([va._data for va in to_concat])
        else:
            data = np.empty(len(to_concat), dtype=object)
            data[:] = [va._data for va in to_concat]
        return cls(data, copy=False)

    @property
    def dtype(self) -> PandasArrayExtensionDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        return np.array([pd.isna(arr).any() for arr in self._data])

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        raise NotImplementedError()

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> Union[np.ndarray, "PandasArrayExtensionArray"]:
        if isinstance(item, int):
            return self._data[item]
        return PandasArrayExtensionArray(self._data[item], copy=False)

    def take(
        self, indices: Sequence_[int], allow_fill: bool = False, fill_value: bool = None
    ) -> "PandasArrayExtensionArray":
        indices: np.ndarray = np.asarray(indices, dtype=int)
        if allow_fill:
            fill_value = (
                self.dtype.na_value if fill_value is None else np.asarray(fill_value, dtype=self.dtype.value_type)
            )
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError("Invalid value in `indices`, must be all >= -1 for `allow_fill` is True")
            elif len(self) > 0:
                pass
            elif not np.all(mask):
                raise IndexError("Invalid take for empty PandasArrayExtensionArray, must be all -1.")
            else:
                data = np.array([fill_value] * len(indices), dtype=self.dtype.value_type)
                return PandasArrayExtensionArray(data, copy=False)
        took = self._data.take(indices, axis=0)
        if allow_fill and mask.any():
            took[mask] = [fill_value] * np.sum(mask)
        return PandasArrayExtensionArray(took, copy=False)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if not isinstance(other, PandasArrayExtensionArray):
            raise NotImplementedError(f"Invalid type to compare to: {type(other)}")
        return (self._data == other._data).all()


def pandas_types_mapper(dtype):
    if isinstance(dtype, _ArrayXDExtensionType):
        return PandasArrayExtensionDtype(dtype.value_type)


@dataclass
class ClassLabel:
    """Feature type for integer class labels.

    There are 3 ways to define a `ClassLabel`, which correspond to the 3 arguments:

     * `num_classes`: Create 0 to (num_classes-1) labels.
     * `names`: List of label strings.
     * `names_file`: File containing the list of labels.

    Under the hood the labels are stored as integers.
    You can use negative integers to represent unknown/missing labels.

    Args:
        num_classes (`int`, *optional*):
            Number of classes. All labels must be < `num_classes`.
        names (`list` of `str`, *optional*):
            String names for the integer classes.
            The order in which the names are provided is kept.
        names_file (`str`, *optional*):
            Path to a file with names for the integer classes, one per line.

    Example:

    ```py
    >>> from datasets import Features
    >>> features = Features({'label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'])})
    >>> features
    {'label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'], id=None)}
    ```
    """

    num_classes: InitVar[Optional[int]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    names: List[str] = None
    names_file: InitVar[Optional[str]] = None  # Pseudo-field: ignored by asdict/fields when converting to/from dict
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "int64"
    pa_type: ClassVar[Any] = pa.int64()
    _str2int: ClassVar[Dict[str, int]] = None
    _int2str: ClassVar[Dict[int, int]] = None
    _type: str = field(default="ClassLabel", init=False, repr=False)

    def __post_init__(self, num_classes, names_file):
        self.num_classes = num_classes
        self.names_file = names_file
        if self.names_file is not None and self.names is not None:
            raise ValueError("Please provide either names or names_file but not both.")
        # Set self.names
        if self.names is None:
            if self.names_file is not None:
                self.names = self._load_names_from_file(self.names_file)
            elif self.num_classes is not None:
                self.names = [str(i) for i in range(self.num_classes)]
            else:
                raise ValueError("Please provide either num_classes, names or names_file.")
        elif not isinstance(self.names, SequenceABC):
            raise TypeError(f"Please provide names as a list, is {type(self.names)}")
        # Set self.num_classes
        if self.num_classes is None:
            self.num_classes = len(self.names)
        elif self.num_classes != len(self.names):
            raise ValueError(
                "ClassLabel number of names do not match the defined num_classes. "
                f"Got {len(self.names)} names VS {self.num_classes} num_classes"
            )
        # Prepare mappings
        self._int2str = [str(name) for name in self.names]
        self._str2int = {name: i for i, name in enumerate(self._int2str)}
        if len(self._int2str) != len(self._str2int):
            raise ValueError("Some label names are duplicated. Each label name should be unique.")

    def __call__(self):
        return self.pa_type

    def str2int(self, values: Union[str, Iterable]) -> Union[int, Iterable]:
        """Conversion class name `string` => `integer`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].str2int('neg')
        0
        ```
        """
        if not isinstance(values, str) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be a string or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, str):
            values = [values]
            return_list = False

        output = [self._strval2int(value) for value in values]
        return output if return_list else output[0]

    def _strval2int(self, value: str) -> int:
        failed_parse = False
        value = str(value)
        # first attempt - raw string value
        int_value = self._str2int.get(value)
        if int_value is None:
            # second attempt - strip whitespace
            int_value = self._str2int.get(value.strip())
            if int_value is None:
                # third attempt - convert str to int
                try:
                    int_value = int(value)
                except ValueError:
                    failed_parse = True
                else:
                    if int_value < -1 or int_value >= self.num_classes:
                        failed_parse = True
        if failed_parse:
            raise ValueError(f"Invalid string class label {value}")
        return int_value

    def int2str(self, values: Union[int, Iterable]) -> Union[str, Iterable]:
        """Conversion `integer` => class name `string`.

        Regarding unknown/missing labels: passing negative integers raises `ValueError`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> ds.features["label"].int2str(0)
        'neg'
        ```
        """
        if not isinstance(values, int) and not isinstance(values, Iterable):
            raise ValueError(
                f"Values {values} should be an integer or an Iterable (list, numpy array, pytorch, tensorflow tensors)"
            )
        return_list = True
        if isinstance(values, int):
            values = [values]
            return_list = False

        for v in values:
            if not 0 <= v < self.num_classes:
                raise ValueError(f"Invalid integer class label {v:d}")

        output = [self._int2str[int(v)] for v in values]
        return output if return_list else output[0]

    def encode_example(self, example_data):
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel feature with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if not -1 <= example_data < self.num_classes:
            raise ValueError(f"Class label {example_data:d} greater than configured num_classes {self.num_classes}")
        return example_data

    def cast_storage(self, storage: Union[pa.StringArray, pa.IntegerArray]) -> pa.Int64Array:
        """Cast an Arrow array to the `ClassLabel` arrow storage type.
        The Arrow types that can be converted to the `ClassLabel` pyarrow storage type are:

        - `pa.string()`
        - `pa.int()`

        Args:
            storage (`Union[pa.StringArray, pa.IntegerArray]`):
                PyArrow array to cast.

        Returns:
            `pa.Int64Array`: Array in the `ClassLabel` arrow storage type.
        """
        if isinstance(storage, pa.IntegerArray) and len(storage) > 0:
            min_max = pc.min_max(storage).as_py()
            if min_max["max"] is not None and min_max["max"] >= self.num_classes:
                raise ValueError(
                    f"Class label {min_max['max']} greater than configured num_classes {self.num_classes}"
                )
        elif isinstance(storage, pa.StringArray):
            storage = pa.array(
                [self._strval2int(label) if label is not None else None for label in storage.to_pylist()]
            )
        return array_cast(storage, self.pa_type)

    @staticmethod
    def _load_names_from_file(names_filepath):
        with open(names_filepath, encoding="utf-8") as f:
            return [name.strip() for name in f.read().split("\n") if name.strip()]  # Filter empty names


@dataclass
class Sequence:
    """Construct a list of feature from a single type or a dict of types.
    Mostly here for compatiblity with tfds.

    Args:
        feature:
            A list of features of a single type or a dictionary of types.
        length (`int`):
            Length of the sequence.

    Example:

    ```py
    >>> from datasets import Features, Sequence, Value, ClassLabel
    >>> features = Features({'post': Sequence(feature={'text': Value(dtype='string'), 'upvotes': Value(dtype='int32'), 'label': ClassLabel(num_classes=2, names=['hot', 'cold'])})})
    >>> features
    {'post': Sequence(feature={'text': Value(dtype='string', id=None), 'upvotes': Value(dtype='int32', id=None), 'label': ClassLabel(num_classes=2, names=['hot', 'cold'], id=None)}, length=-1, id=None)}
    ```
    """

    feature: Any
    length: int = -1
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "list"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Sequence", init=False, repr=False)


@dataclass
class LargeList:
    """Feature type for large list data composed of child feature data type.

    It is backed by `pyarrow.LargeListType`, which is like `pyarrow.ListType` but with 64-bit rather than 32-bit offsets.

    Args:
        dtype:
            Child feature data type of each item within the large list.
    """

    dtype: Any
    id: Optional[str] = None
    # Automatically constructed
    pa_type: ClassVar[Any] = None
    _type: str = field(default="LargeList", init=False, repr=False)


FeatureType = Union[
    dict,
    list,
    tuple,
    Value,
    ClassLabel,
    Translation,
    TranslationVariableLanguages,
    LargeList,
    Sequence,
    Array2D,
    Array3D,
    Array4D,
    Array5D,
    Audio,
    Image,
]


def _check_non_null_non_empty_recursive(obj, schema: Optional[FeatureType] = None) -> bool:
    """
    Check if the object is not None.
    If the object is a list or a tuple, recursively check the first element of the sequence and stop if at any point the first element is not a sequence or is an empty sequence.
    """
    if obj is None:
        return False
    elif isinstance(obj, (list, tuple)) and (schema is None or isinstance(schema, (list, tuple, LargeList, Sequence))):
        if len(obj) > 0:
            if schema is None:
                pass
            elif isinstance(schema, (list, tuple)):
                schema = schema[0]
            elif isinstance(schema, LargeList):
                schema = schema.dtype
            else:
                schema = schema.feature
            return _check_non_null_non_empty_recursive(obj[0], schema)
        else:
            return False
    else:
        return True


def get_nested_type(schema: FeatureType) -> pa.DataType:
    """
    get_nested_type() converts a datasets.FeatureType into a pyarrow.DataType, and acts as the inverse of
        generate_from_arrow_type().

    It performs double-duty as the implementation of Features.type and handles the conversion of
        datasets.Feature->pa.struct
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, Features):
        return pa.struct(
            {key: get_nested_type(schema[key]) for key in schema}
        )  # Features is subclass of dict, and dict order is deterministic since Python 3.6
    elif isinstance(schema, dict):
        return pa.struct(
            {key: get_nested_type(schema[key]) for key in schema}
        )  # however don't sort on struct types since the order matters
    elif isinstance(schema, (list, tuple)):
        if len(schema) != 1:
            raise ValueError("When defining list feature, you should just provide one example of the inner type")
        value_type = get_nested_type(schema[0])
        return pa.list_(value_type)
    elif isinstance(schema, LargeList):
        value_type = get_nested_type(schema.dtype)
        return pa.large_list(value_type)
    elif isinstance(schema, Sequence):
        value_type = get_nested_type(schema.feature)
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            data_type = pa.struct({f.name: pa.list_(f.type, schema.length) for f in value_type})
        else:
            data_type = pa.list_(value_type, schema.length)
        return data_type

    # Other objects are callable which returns their data type (ClassLabel, Array2D, Translation, Arrow datatype creation methods)
    return schema()


def encode_nested_example(schema, obj, level=0):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return (
            {k: encode_nested_example(schema[k], obj.get(k), level=level + 1) for k in schema}
            if obj is not None
            else None
        )

    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return encode_nested_example(schema, obj.tolist())
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.dtype
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if encode_nested_example(sub_schema, first_elmt, level=level + 1) != first_elmt:
                    return [encode_nested_example(sub_schema, o, level=level + 1) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k in schema.feature:
                    list_dict[k] = [encode_nested_example(schema.feature[k], o.get(k), level=level + 1) for o in obj]
                return list_dict
            else:
                # obj is a single dict
                for k in schema.feature:
                    list_dict[k] = (
                        [encode_nested_example(schema.feature[k], o, level=level + 1) for o in obj[k]]
                        if k in obj
                        else None
                    )
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.feature):
                        break
                # be careful when comparing tensors here
                if (
                    not isinstance(first_elmt, list)
                    or encode_nested_example(schema.feature, first_elmt, level=level + 1) != first_elmt
                ):
                    return [encode_nested_example(schema.feature, o, level=level + 1) for o in obj]
            return list(obj)
    # Object with special encoding:
    # ClassLabel will convert from string to int, TranslationVariableLanguages does some checks
    elif isinstance(schema, (Audio, Image, ClassLabel, TranslationVariableLanguages, Value, _ArrayXD)):
        return schema.encode_example(obj) if obj is not None else None
    # Other object should be directly convertible to a native Arrow type (like Translation and Translation)
    return obj


def decode_nested_example(schema, obj, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be decoded.
    If the first element needs to be decoded, then all the elements of the list will be decoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return (
            {k: decode_nested_example(sub_schema, sub_obj) for k, (sub_schema, sub_obj) in zip_dict(schema, obj)}
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            sub_schema = schema.dtype
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            return {k: decode_nested_example([schema.feature[k]], obj[k]) for k in schema.feature}
        else:
            return decode_nested_example([schema.feature], obj)
    # Object with special decoding:
    elif isinstance(schema, (Audio, Image)):
        # we pass the token to read and decode files from private repositories in streaming mode
        if obj is not None and schema.decode:
            return schema.decode_example(obj, token_per_repo_id=token_per_repo_id)
    return obj


_FEATURE_TYPES: Dict[str, FeatureType] = {
    Value.__name__: Value,
    ClassLabel.__name__: ClassLabel,
    Translation.__name__: Translation,
    TranslationVariableLanguages.__name__: TranslationVariableLanguages,
    LargeList.__name__: LargeList,
    Sequence.__name__: Sequence,
    Array2D.__name__: Array2D,
    Array3D.__name__: Array3D,
    Array4D.__name__: Array4D,
    Array5D.__name__: Array5D,
    Audio.__name__: Audio,
    Image.__name__: Image,
}


@experimental
def register_feature(
    feature_cls: type,
    feature_type: str,
):
    """
    Register a Feature object using a name and class.
    This function must be used on a Feature class.
    """
    if feature_type in _FEATURE_TYPES:
        logger.warning(
            f"Overwriting feature type '{feature_type}' ({_FEATURE_TYPES[feature_type].__name__} -> {feature_cls.__name__})"
        )
    _FEATURE_TYPES[feature_type] = feature_cls


def generate_from_dict(obj: Any):
    """Regenerate the nested feature object from a deserialized dict.
    We use the '_type' fields to get the dataclass name to load.

    generate_from_dict is the recursive helper for Features.from_dict, and allows for a convenient constructor syntax
    to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
    a :class:`DatasetInfo` that was dumped to a JSON object. This acts as an analogue to
    :meth:`Features.from_arrow_schema` and handles the recursive field-by-field instantiation, but doesn't require any
    mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive dtypes
    that :class:`Value` automatically performs.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(obj, list):
        return [generate_from_dict(value) for value in obj]
    # Otherwise we have a dict or a dataclass
    if "_type" not in obj or isinstance(obj["_type"], dict):
        return {key: generate_from_dict(value) for key, value in obj.items()}
    obj = dict(obj)
    _type = obj.pop("_type")
    class_type = _FEATURE_TYPES.get(_type, None) or globals().get(_type, None)

    if class_type is None:
        raise ValueError(f"Feature type '{_type}' not found. Available feature types: {list(_FEATURE_TYPES.keys())}")

    if class_type == LargeList:
        dtype = obj.pop("dtype")
        return LargeList(generate_from_dict(dtype), **obj)
    if class_type == Sequence:
        feature = obj.pop("feature")
        return Sequence(feature=generate_from_dict(feature), **obj)

    field_names = {f.name for f in fields(class_type)}
    return class_type(**{k: v for k, v in obj.items() if k in field_names})


def generate_from_arrow_type(pa_type: pa.DataType) -> FeatureType:
    """
    generate_from_arrow_type accepts an arrow DataType and returns a datasets FeatureType to be used as the type for
        a single field.

    This is the high-level arrow->datasets type conversion and is inverted by get_nested_type().

    This operates at the individual *field* level, whereas Features.from_arrow_schema() operates at the
        full schema level and holds the methods that represent the bijection from Features<->pyarrow.Schema
    """
    if isinstance(pa_type, pa.StructType):
        return {field.name: generate_from_arrow_type(field.type) for field in pa_type}
    elif isinstance(pa_type, pa.FixedSizeListType):
        return Sequence(feature=generate_from_arrow_type(pa_type.value_type), length=pa_type.list_size)
    elif isinstance(pa_type, pa.ListType):
        feature = generate_from_arrow_type(pa_type.value_type)
        if isinstance(feature, (dict, tuple, list)):
            return [feature]
        return Sequence(feature=feature)
    elif isinstance(pa_type, pa.LargeListType):
        dtype = generate_from_arrow_type(pa_type.value_type)
        return LargeList(dtype)
    elif isinstance(pa_type, _ArrayXDExtensionType):
        array_feature = [None, None, Array2D, Array3D, Array4D, Array5D][pa_type.ndims]
        return array_feature(shape=pa_type.shape, dtype=pa_type.value_type)
    elif isinstance(pa_type, pa.DataType):
        return Value(dtype=_arrow_to_datasets_dtype(pa_type))
    else:
        raise ValueError(f"Cannot convert {pa_type} to a Feature type.")


def numpy_to_pyarrow_listarray(arr: np.ndarray, type: pa.DataType = None) -> pa.ListArray:
    """Build a PyArrow ListArray from a multidimensional NumPy array"""
    arr = np.array(arr)
    values = pa.array(arr.flatten(), type=type)
    for i in range(arr.ndim - 1):
        n_offsets = reduce(mul, arr.shape[: arr.ndim - i - 1], 1)
        step_offsets = arr.shape[arr.ndim - i - 1]
        offsets = pa.array(np.arange(n_offsets + 1) * step_offsets, type=pa.int32())
        values = pa.ListArray.from_arrays(offsets, values)
    return values


def list_of_pa_arrays_to_pyarrow_listarray(l_arr: List[Optional[pa.Array]]) -> pa.ListArray:
    null_mask = np.array([arr is None for arr in l_arr])
    null_indices = np.arange(len(null_mask))[null_mask] - np.arange(np.sum(null_mask))
    l_arr = [arr for arr in l_arr if arr is not None]
    offsets = np.cumsum(
        [0] + [len(arr) for arr in l_arr], dtype=object
    )  # convert to dtype object to allow None insertion
    offsets = np.insert(offsets, null_indices, None)
    offsets = pa.array(offsets, type=pa.int32())
    values = pa.concat_arrays(l_arr)
    return pa.ListArray.from_arrays(offsets, values)


def list_of_np_array_to_pyarrow_listarray(l_arr: List[np.ndarray], type: pa.DataType = None) -> pa.ListArray:
    """Build a PyArrow ListArray from a possibly nested list of NumPy arrays"""
    if len(l_arr) > 0:
        return list_of_pa_arrays_to_pyarrow_listarray(
            [numpy_to_pyarrow_listarray(arr, type=type) if arr is not None else None for arr in l_arr]
        )
    else:
        return pa.array([], type=type)


def contains_any_np_array(data: Any):
    """Return `True` if data is a NumPy ndarray or (recursively) if first non-null value in list is a NumPy ndarray.

    Args:
        data (Any): Data.

    Returns:
        bool
    """
    if isinstance(data, np.ndarray):
        return True
    elif isinstance(data, list):
        return contains_any_np_array(first_non_null_value(data)[1])
    else:
        return False


def any_np_array_to_pyarrow_listarray(data: Union[np.ndarray, List], type: pa.DataType = None) -> pa.ListArray:
    """Convert to PyArrow ListArray either a NumPy ndarray or (recursively) a list that may contain any NumPy ndarray.

    Args:
        data (Union[np.ndarray, List]): Data.
        type (pa.DataType): Explicit PyArrow DataType passed to coerce the ListArray data type.

    Returns:
        pa.ListArray
    """
    if isinstance(data, np.ndarray):
        return numpy_to_pyarrow_listarray(data, type=type)
    elif isinstance(data, list):
        return list_of_pa_arrays_to_pyarrow_listarray([any_np_array_to_pyarrow_listarray(i, type=type) for i in data])


def to_pyarrow_listarray(data: Any, pa_type: _ArrayXDExtensionType) -> pa.Array:
    """Convert to PyArrow ListArray.

    Args:
        data (Any): Sequence, iterable, np.ndarray or pd.Series.
        pa_type (_ArrayXDExtensionType): Any of the ArrayNDExtensionType.

    Returns:
        pyarrow.Array
    """
    if contains_any_np_array(data):
        return any_np_array_to_pyarrow_listarray(data, type=pa_type.value_type)
    else:
        return pa.array(data, pa_type.storage_dtype)


def _visit(feature: FeatureType, func: Callable[[FeatureType], Optional[FeatureType]]) -> FeatureType:
    """Visit a (possibly nested) feature.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        visited feature (FeatureType)
    """
    if isinstance(feature, dict):
        out = func({k: _visit(f, func) for k, f in feature.items()})
    elif isinstance(feature, (list, tuple)):
        out = func([_visit(feature[0], func)])
    elif isinstance(feature, LargeList):
        out = func(LargeList(_visit(feature.dtype, func)))
    elif isinstance(feature, Sequence):
        out = func(Sequence(_visit(feature.feature, func), length=feature.length))
    else:
        out = func(feature)
    return feature if out is None else out


def require_decoding(feature: FeatureType, ignore_decode_attribute: bool = False) -> bool:
    """Check if a (possibly nested) feature requires decoding.

    Args:
        feature (FeatureType): the feature type to be checked
        ignore_decode_attribute (:obj:`bool`, default ``False``): Whether to ignore the current value
            of the `decode` attribute of the decodable feature types.
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_decoding(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_decoding(feature[0])
    elif isinstance(feature, LargeList):
        return require_decoding(feature.dtype)
    elif isinstance(feature, Sequence):
        return require_decoding(feature.feature)
    else:
        return hasattr(feature, "decode_example") and (feature.decode if not ignore_decode_attribute else True)


def require_storage_cast(feature: FeatureType) -> bool:
    """Check if a (possibly nested) feature requires storage casting.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_storage_cast(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_storage_cast(feature[0])
    elif isinstance(feature, LargeList):
        return require_storage_cast(feature.dtype)
    elif isinstance(feature, Sequence):
        return require_storage_cast(feature.feature)
    else:
        return hasattr(feature, "cast_storage")


def require_storage_embed(feature: FeatureType) -> bool:
    """Check if a (possibly nested) feature requires embedding data into storage.

    Args:
        feature (FeatureType): the feature type to be checked
    Returns:
        :obj:`bool`
    """
    if isinstance(feature, dict):
        return any(require_storage_cast(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return require_storage_cast(feature[0])
    elif isinstance(feature, LargeList):
        return require_storage_cast(feature.dtype)
    elif isinstance(feature, Sequence):
        return require_storage_cast(feature.feature)
    else:
        return hasattr(feature, "embed_storage")


def keep_features_dicts_synced(func):
    """
    Wrapper to keep the secondary dictionary, which tracks whether keys are decodable, of the :class:`datasets.Features` object
    in sync with the main dictionary.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Features" = args[0]
            args = args[1:]
        else:
            self: "Features" = kwargs.pop("self")
        out = func(self, *args, **kwargs)
        assert hasattr(self, "_column_requires_decoding")
        self._column_requires_decoding = {col: require_decoding(feature) for col, feature in self.items()}
        return out

    wrapper._decorator_name_ = "_keep_dicts_synced"
    return wrapper


class Features(dict):
    """A special dictionary that defines the internal structure of a dataset.

    Instantiated with a dictionary of type `dict[str, FieldType]`, where keys are the desired column names,
    and values are the type of that column.

    `FieldType` can be one of the following:
        - a [`~datasets.Value`] feature specifies a single typed value, e.g. `int64` or `string`.
        - a [`~datasets.ClassLabel`] feature specifies a field with a predefined set of classes which can have labels
          associated to them and will be stored as integers in the dataset.
        - a python `dict` which specifies that the field is a nested field containing a mapping of sub-fields to sub-fields
          features. It's possible to have nested fields of nested fields in an arbitrary manner.
        - a python `list` or a [`~datasets.Sequence`] specifies that the field contains a list of objects. The python
          `list` or [`~datasets.Sequence`] should be provided with a single sub-feature as an example of the feature
          type hosted in this list.

          <Tip>

           A [`~datasets.Sequence`] with a internal dictionary feature will be automatically converted into a dictionary of
           lists. This behavior is implemented to have a compatibility layer with the TensorFlow Datasets library but may be
           un-wanted in some cases. If you don't want this behavior, you can use a python `list` instead of the
           [`~datasets.Sequence`].

          </Tip>

        - a [`Array2D`], [`Array3D`], [`Array4D`] or [`Array5D`] feature for multidimensional arrays.
        - an [`Audio`] feature to store the absolute path to an audio file or a dictionary with the relative path
          to an audio file ("path" key) and its bytes content ("bytes" key). This feature extracts the audio data.
        - an [`Image`] feature to store the absolute path to an image file, an `np.ndarray` object, a `PIL.Image.Image` object
          or a dictionary with the relative path to an image file ("path" key) and its bytes content ("bytes" key). This feature extracts the image data.
        - [`~datasets.Translation`] and [`~datasets.TranslationVariableLanguages`], the two features specific to Machine Translation.
    """

    def __init__(*args, **kwargs):
        # self not in the signature to allow passing self as a kwarg
        if not args:
            raise TypeError("descriptor '__init__' of 'Features' object needs an argument")
        self, *args = args
        super(Features, self).__init__(*args, **kwargs)
        self._column_requires_decoding: Dict[str, bool] = {
            col: require_decoding(feature) for col, feature in self.items()
        }

    __setitem__ = keep_features_dicts_synced(dict.__setitem__)
    __delitem__ = keep_features_dicts_synced(dict.__delitem__)
    update = keep_features_dicts_synced(dict.update)
    setdefault = keep_features_dicts_synced(dict.setdefault)
    pop = keep_features_dicts_synced(dict.pop)
    popitem = keep_features_dicts_synced(dict.popitem)
    clear = keep_features_dicts_synced(dict.clear)

    def __reduce__(self):
        return Features, (dict(self),)

    @property
    def type(self):
        """
        Features field types.

        Returns:
            :obj:`pyarrow.DataType`
        """
        return get_nested_type(self)

    @property
    def arrow_schema(self):
        """
        Features schema.

        Returns:
            :obj:`pyarrow.Schema`
        """
        hf_metadata = {"info": {"features": self.to_dict()}}
        return pa.schema(self.type).with_metadata({"huggingface": json.dumps(hf_metadata)})

    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema) -> "Features":
        """
        Construct [`Features`] from Arrow Schema.
        It also checks the schema metadata for Hugging Face Datasets features.
        Non-nullable fields are not supported and set to nullable.

        Also, pa.dictionary is not supported and it uses its underlying type instead.
        Therefore datasets convert DictionaryArray objects to their actual values.

        Args:
            pa_schema (`pyarrow.Schema`):
                Arrow Schema.

        Returns:
            [`Features`]
        """
        # try to load features from the arrow schema metadata
        metadata_features = Features()
        if pa_schema.metadata is not None and "huggingface".encode("utf-8") in pa_schema.metadata:
            metadata = json.loads(pa_schema.metadata["huggingface".encode("utf-8")].decode())
            if "info" in metadata and "features" in metadata["info"] and metadata["info"]["features"] is not None:
                metadata_features = Features.from_dict(metadata["info"]["features"])
        metadata_features_schema = metadata_features.arrow_schema
        obj = {
            field.name: (
                metadata_features[field.name]
                if field.name in metadata_features and metadata_features_schema.field(field.name) == field
                else generate_from_arrow_type(field.type)
            )
            for field in pa_schema
        }
        return cls(**obj)

    @classmethod
    def from_dict(cls, dic) -> "Features":
        """
        Construct [`Features`] from dict.

        Regenerate the nested feature object from a deserialized dict.
        We use the `_type` key to infer the dataclass name of the feature `FieldType`.

        It allows for a convenient constructor syntax
        to define features from deserialized JSON dictionaries. This function is used in particular when deserializing
        a [`DatasetInfo`] that was dumped to a JSON object. This acts as an analogue to
        [`Features.from_arrow_schema`] and handles the recursive field-by-field instantiation, but doesn't require
        any mapping to/from pyarrow, except for the fact that it takes advantage of the mapping of pyarrow primitive
        dtypes that [`Value`] automatically performs.

        Args:
            dic (`dict[str, Any]`):
                Python dictionary.

        Returns:
            `Features`

        Example::
            >>> Features.from_dict({'_type': {'dtype': 'string', 'id': None, '_type': 'Value'}})
            {'_type': Value(dtype='string', id=None)}
        """
        obj = generate_from_dict(dic)
        return cls(**obj)

    def to_dict(self):
        return asdict(self)

    def _to_yaml_list(self) -> list:
        # we compute the YAML list from the dict representation that is used for JSON dump
        yaml_data = self.to_dict()

        def simplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type:                ->              list_type: int32
                #   dtype: int32            ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["dtype"]:
                    feature[list_type] = feature[list_type]["dtype"]

                #
                # list_type:                ->              list_type:
                #   struct:                 ->              - name: foo
                #   - name: foo             ->                dtype: int32
                #     dtype: int32          ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["struct"]:
                    feature[list_type] = feature[list_type]["struct"]

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #   - negative              ->                  '0': negative
            #   - positive              ->                  '1': positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), list):
                # server-side requirement: keys must be strings
                feature["class_label"]["names"] = {
                    str(label_id): label_name for label_id, label_name in enumerate(feature["class_label"]["names"])
                }
            return feature

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop("_type", None)
                if _type == "LargeList":
                    value_type = obj.pop("dtype")
                    return simplify({"large_list": to_yaml_inner(value_type), **obj})
                elif _type == "Sequence":
                    _feature = obj.pop("feature")
                    return simplify({"sequence": to_yaml_inner(_feature), **obj})
                elif _type == "Value":
                    return obj
                elif _type and not obj:
                    return {"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    return {"dtype": simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {"struct": [{"name": name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
            elif isinstance(obj, list):
                return simplify({"list": simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        def to_yaml_types(obj: dict) -> dict:
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj

        return to_yaml_types(to_yaml_inner(yaml_data)["struct"])

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "Features":
        yaml_data = copy.deepcopy(yaml_data)

        # we convert the list obtained from YAML data into the dict representation that is used for JSON dump

        def unsimplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type: int32          ->              list_type:
                #                           ->                dtype: int32
                #
                if isinstance(feature.get(list_type), str):
                    feature[list_type] = {"dtype": feature[list_type]}

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #     '0': negative              ->               - negative
            #     '1': positive              ->               - positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), dict):
                label_ids = sorted(feature["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
                    raise ValueError(
                        f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                    )
                feature["class_label"]["names"] = [feature["class_label"]["names"][label_id] for label_id in label_ids]
            return feature

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == "large_list":
                    _dtype = unsimplify(obj).pop(_type)
                    return {"dtype": from_yaml_inner(_dtype), **obj, "_type": "LargeList"}
                if _type == "sequence":
                    _feature = unsimplify(obj).pop(_type)
                    return {"feature": from_yaml_inner(_feature), **obj, "_type": "Sequence"}
                if _type == "list":
                    return [from_yaml_inner(unsimplify(obj)[_type])]
                if _type == "struct":
                    return from_yaml_inner(obj["struct"])
                elif _type == "dtype":
                    if isinstance(obj["dtype"], str):
                        # e.g. int32, float64, string, audio, image
                        try:
                            Value(obj["dtype"])
                            return {**obj, "_type": "Value"}
                        except ValueError:
                            # e.g. Audio, Image, ArrayXD
                            return {"_type": snakecase_to_camelcase(obj["dtype"])}
                    else:
                        return from_yaml_inner(obj["dtype"])
                else:
                    return {"_type": snakecase_to_camelcase(_type), **unsimplify(obj)[_type]}
            elif isinstance(obj, list):
                names = [_feature.pop("name") for _feature in obj]
                return {name: from_yaml_inner(_feature) for name, _feature in zip(names, obj)}
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        return cls.from_dict(from_yaml_inner(yaml_data))

    def encode_example(self, example):
        """
        Encode example into a format for Arrow.

        Args:
            example (`dict[str, Any]`):
                Data in a Dataset row.

        Returns:
            `dict[str, Any]`
        """
        example = cast_to_python_objects(example)
        return encode_nested_example(self, example)

    def encode_column(self, column, column_name: str):
        """
        Encode column into a format for Arrow.

        Args:
            column (`list[Any]`):
                Data in a Dataset column.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        column = cast_to_python_objects(column)
        return [encode_nested_example(self[column_name], obj, level=1) for obj in column]

    def encode_batch(self, batch):
        """
        Encode batch into a format for Arrow.

        Args:
            batch (`dict[str, list[Any]]`):
                Data in a Dataset batch.

        Returns:
            `dict[str, list[Any]]`
        """
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(f"Column mismatch between batch {set(batch)} and features {set(self)}")
        for key, column in batch.items():
            column = cast_to_python_objects(column)
            encoded_batch[key] = [encode_nested_example(self[key], obj, level=1) for obj in column]
        return encoded_batch

    def decode_example(self, example: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None):
        """Decode example with custom feature decoding.

        Args:
            example (`dict[str, Any]`):
                Dataset row data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary `repo_id (str) -> token (bool or str)`.

        Returns:
            `dict[str, Any]`
        """

        return {
            column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
            if self._column_requires_decoding[column_name]
            else value
            for column_name, (feature, value) in zip_dict(
                {key: value for key, value in self.items() if key in example}, example
            )
        }

    def decode_column(self, column: list, column_name: str):
        """Decode column with custom feature decoding.

        Args:
            column (`list[Any]`):
                Dataset column data.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return (
            [decode_nested_example(self[column_name], value) if value is not None else None for value in column]
            if self._column_requires_decoding[column_name]
            else column
        )

    def decode_batch(self, batch: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None):
        """Decode batch with custom feature decoding.

        Args:
            batch (`dict[str, list[Any]]`):
                Dataset batch data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary repo_id (str) -> token (bool or str)

        Returns:
            `dict[str, list[Any]]`
        """
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = (
                [
                    decode_nested_example(self[column_name], value, token_per_repo_id=token_per_repo_id)
                    if value is not None
                    else None
                    for value in column
                ]
                if self._column_requires_decoding[column_name]
                else column
            )
        return decoded_batch

    def copy(self) -> "Features":
        """
        Make a deep copy of [`Features`].

        Returns:
            [`Features`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> copy_of_features = ds.features.copy()
        >>> copy_of_features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        ```
        """
        return copy.deepcopy(self)

    def reorder_fields_as(self, other: "Features") -> "Features":
        """
        Reorder Features fields to match the field order of other [`Features`].

        The order of the fields is important since it matters for the underlying arrow data.
        Re-ordering the fields allows to make the underlying arrow data type match.

        Args:
            other ([`Features`]):
                The other [`Features`] to align with.

        Returns:
            [`Features`]

        Example::

            >>> from datasets import Features, Sequence, Value
            >>> # let's say we have two features with a different order of nested fields (for a and b for example)
            >>> f1 = Features({"root": Sequence({"a": Value("string"), "b": Value("string")})})
            >>> f2 = Features({"root": {"b": Sequence(Value("string")), "a": Sequence(Value("string"))}})
            >>> assert f1.type != f2.type
            >>> # re-ordering keeps the base structure (here Sequence is defined at the root level), but makes the fields order match
            >>> f1.reorder_fields_as(f2)
            {'root': Sequence(feature={'b': Value(dtype='string', id=None), 'a': Value(dtype='string', id=None)}, length=-1, id=None)}
            >>> assert f1.reorder_fields_as(f2).type == f2.type
        """

        def recursive_reorder(source, target, stack=""):
            stack_position = " at " + stack[1:] if stack else ""
            if isinstance(target, Sequence):
                target = target.feature
                if isinstance(target, dict):
                    target = {k: [v] for k, v in target.items()}
                else:
                    target = [target]
            if isinstance(source, Sequence):
                sequence_kwargs = vars(source).copy()
                source = sequence_kwargs.pop("feature")
                if isinstance(source, dict):
                    source = {k: [v] for k, v in source.items()}
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence({k: v[0] for k, v in reordered.items()}, **sequence_kwargs)
                else:
                    source = [source]
                    reordered = recursive_reorder(source, target, stack)
                    return Sequence(reordered[0], **sequence_kwargs)
            elif isinstance(source, dict):
                if not isinstance(target, dict):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                if sorted(source) != sorted(target):
                    message = (
                        f"Keys mismatch: between {source} (source) and {target} (target).\n"
                        f"{source.keys()-target.keys()} are missing from target "
                        f"and {target.keys()-source.keys()} are missing from source" + stack_position
                    )
                    raise ValueError(message)
                return {key: recursive_reorder(source[key], target[key], stack + f".{key}") for key in target}
            elif isinstance(source, list):
                if not isinstance(target, list):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                if len(source) != len(target):
                    raise ValueError(f"Length mismatch: between {source} and {target}" + stack_position)
                return [recursive_reorder(source[i], target[i], stack + ".<list>") for i in range(len(target))]
            elif isinstance(source, LargeList):
                if not isinstance(target, LargeList):
                    raise ValueError(f"Type mismatch: between {source} and {target}" + stack_position)
                return LargeList(recursive_reorder(source.dtype, target.dtype, stack))
            else:
                return source

        return Features(recursive_reorder(self, other))

    def flatten(self, max_depth=16) -> "Features":
        """Flatten the features. Every dictionary column is removed and is replaced by
        all the subfields it contains. The new fields are named by concatenating the
        name of the original column and the subfield name like this: `<original>.<subfield>`.

        If a column contains nested dictionaries, then all the lower-level subfields names are
        also concatenated to form new columns: `<original>.<subfield>.<subsubfield>`, etc.

        Returns:
            [`Features`]:
                The flattened features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("squad", split="train")
        >>> ds.features.flatten()
        {'answers.answer_start': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
         'answers.text': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
         'context': Value(dtype='string', id=None),
         'id': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None),
         'title': Value(dtype='string', id=None)}
        ```
        """
        for depth in range(1, max_depth):
            no_change = True
            flattened = self.copy()
            for column_name, subfeature in self.items():
                if isinstance(subfeature, dict):
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.items()})
                    del flattened[column_name]
                elif isinstance(subfeature, Sequence) and isinstance(subfeature.feature, dict):
                    no_change = False
                    flattened.update(
                        {
                            f"{column_name}.{k}": Sequence(v) if not isinstance(v, dict) else [v]
                            for k, v in subfeature.feature.items()
                        }
                    )
                    del flattened[column_name]
                elif hasattr(subfeature, "flatten") and subfeature.flatten() != subfeature:
                    no_change = False
                    flattened.update({f"{column_name}.{k}": v for k, v in subfeature.flatten().items()})
                    del flattened[column_name]
            self = flattened
            if no_change:
                break
        return self


def _align_features(features_list: List[Features]) -> List[Features]:
    """Align dictionaries of features so that the keys that are found in multiple dictionaries share the same feature."""
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k in name2feature and isinstance(v, dict):
                # Recursively align features.
                name2feature[k] = _align_features([name2feature[k], v])[0]
            elif k not in name2feature or (isinstance(name2feature[k], Value) and name2feature[k].dtype == "null"):
                name2feature[k] = v

    return [Features({k: name2feature[k] for k in features.keys()}) for features in features_list]


def _check_if_features_can_be_aligned(features_list: List[Features]):
    """Check if the dictionaries of features can be aligned.

    Two dictonaries of features can be aligned if the keys they share have the same type or some of them is of type `Value("null")`.
    """
    name2feature = {}
    for features in features_list:
        for k, v in features.items():
            if k not in name2feature or (isinstance(name2feature[k], Value) and name2feature[k].dtype == "null"):
                name2feature[k] = v

    for features in features_list:
        for k, v in features.items():
            if isinstance(v, dict) and isinstance(name2feature[k], dict):
                # Deep checks for structure.
                _check_if_features_can_be_aligned([name2feature[k], v])
            elif not (isinstance(v, Value) and v.dtype == "null") and name2feature[k] != v:
                raise ValueError(
                    f'The features can\'t be aligned because the key {k} of features {features} has unexpected type - {v} (expected either {name2feature[k]} or Value("null").'
                )
