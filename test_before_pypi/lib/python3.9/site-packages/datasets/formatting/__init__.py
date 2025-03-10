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

from typing import Dict, List, Optional, Type

from .. import config
from ..utils import logging
from .formatting import (
    ArrowFormatter,
    CustomFormatter,
    Formatter,
    PandasFormatter,
    PythonFormatter,
    TensorFormatter,
    format_table,
    query_table,
)
from .np_formatter import NumpyFormatter


logger = logging.get_logger(__name__)

_FORMAT_TYPES: Dict[Optional[str], Type[Formatter]] = {}
_FORMAT_TYPES_ALIASES: Dict[Optional[str], str] = {}
_FORMAT_TYPES_ALIASES_UNAVAILABLE: Dict[Optional[str], Exception] = {}


def _register_formatter(
    formatter_cls: type,
    format_type: Optional[str],
    aliases: Optional[List[str]] = None,
):
    """
    Register a Formatter object using a name and optional aliases.
    This function must be used on a Formatter class.
    """
    aliases = aliases if aliases is not None else []
    if format_type in _FORMAT_TYPES:
        logger.warning(
            f"Overwriting format type '{format_type}' ({_FORMAT_TYPES[format_type].__name__} -> {formatter_cls.__name__})"
        )
    _FORMAT_TYPES[format_type] = formatter_cls
    for alias in set(aliases + [format_type]):
        if alias in _FORMAT_TYPES_ALIASES:
            logger.warning(
                f"Overwriting format type alias '{alias}' ({_FORMAT_TYPES_ALIASES[alias]} -> {format_type})"
            )
        _FORMAT_TYPES_ALIASES[alias] = format_type


def _register_unavailable_formatter(
    unavailable_error: Exception, format_type: Optional[str], aliases: Optional[List[str]] = None
):
    """
    Register an unavailable Formatter object using a name and optional aliases.
    This function must be used on an Exception object that is raised when trying to get the unavailable formatter.
    """
    aliases = aliases if aliases is not None else []
    for alias in set(aliases + [format_type]):
        _FORMAT_TYPES_ALIASES_UNAVAILABLE[alias] = unavailable_error


# Here we define all the available formatting functions that can be used by `Dataset.set_format`
_register_formatter(PythonFormatter, None, aliases=["python"])
_register_formatter(ArrowFormatter, "arrow", aliases=["pa", "pyarrow"])
_register_formatter(NumpyFormatter, "numpy", aliases=["np"])
_register_formatter(PandasFormatter, "pandas", aliases=["pd"])
_register_formatter(CustomFormatter, "custom")

if config.POLARS_AVAILABLE:
    from .polars_formatter import PolarsFormatter

    _register_formatter(PolarsFormatter, "polars", aliases=["pl"])
else:
    _polars_error = ValueError("Polars needs to be installed to be able to return Polars dataframes.")
    _register_unavailable_formatter(_polars_error, "polars", aliases=["pl"])

if config.TORCH_AVAILABLE:
    from .torch_formatter import TorchFormatter

    _register_formatter(TorchFormatter, "torch", aliases=["pt", "pytorch"])
else:
    _torch_error = ValueError("PyTorch needs to be installed to be able to return PyTorch tensors.")
    _register_unavailable_formatter(_torch_error, "torch", aliases=["pt", "pytorch"])

if config.TF_AVAILABLE:
    from .tf_formatter import TFFormatter

    _register_formatter(TFFormatter, "tensorflow", aliases=["tf"])
else:
    _tf_error = ValueError("Tensorflow needs to be installed to be able to return Tensorflow tensors.")
    _register_unavailable_formatter(_tf_error, "tensorflow", aliases=["tf"])

if config.JAX_AVAILABLE:
    from .jax_formatter import JaxFormatter

    _register_formatter(JaxFormatter, "jax", aliases=[])
else:
    _jax_error = ValueError("JAX needs to be installed to be able to return JAX arrays.")
    _register_unavailable_formatter(_jax_error, "jax", aliases=[])


def get_format_type_from_alias(format_type: Optional[str]) -> Optional[str]:
    """If the given format type is a known alias, then return its main type name. Otherwise return the type with no change."""
    if format_type in _FORMAT_TYPES_ALIASES:
        return _FORMAT_TYPES_ALIASES[format_type]
    else:
        return format_type


def get_formatter(format_type: Optional[str], **format_kwargs) -> Formatter:
    """
    Factory function to get a Formatter given its type name and keyword arguments.
    A formatter is an object that extracts and formats data from pyarrow table.
    It defines the formatting for rows, colums and batches.
    If the formatter for a given type name doesn't exist or is not available, an error is raised.
    """
    format_type = get_format_type_from_alias(format_type)
    if format_type in _FORMAT_TYPES:
        return _FORMAT_TYPES[format_type](**format_kwargs)
    if format_type in _FORMAT_TYPES_ALIASES_UNAVAILABLE:
        raise _FORMAT_TYPES_ALIASES_UNAVAILABLE[format_type]
    else:
        raise ValueError(f"Format type should be one of {list(_FORMAT_TYPES.keys())}, but got '{format_type}'")
