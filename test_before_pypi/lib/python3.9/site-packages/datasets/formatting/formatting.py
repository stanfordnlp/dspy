# Copyright 2020 The HuggingFace Authors.
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

import operator
from collections.abc import Mapping, MutableMapping
from functools import partial

# Lint as: python3
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version

from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null


T = TypeVar("T")

RowFormat = TypeVar("RowFormat")
ColumnFormat = TypeVar("ColumnFormat")
BatchFormat = TypeVar("BatchFormat")


def _is_range_contiguous(key: range) -> bool:
    return key.step == 1 and key.stop >= key.start


def _raise_bad_key_type(key: Any):
    raise TypeError(
        f"Wrong key type: '{key}' of type '{type(key)}'. Expected one of int, slice, range, str or Iterable."
    )


def _query_table_with_indices_mapping(
    table: Table, key: Union[int, slice, range, str, Iterable], indices: Table
) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    The :obj:`indices` parameter corresponds to the indices mapping in case we cant to take into
    account a shuffling or an indices selection for example.
    The indices table must contain one column named "indices" of type uint64.
    """
    if isinstance(key, int):
        key = indices.fast_slice(key % indices.num_rows, 1).column(0)[0].as_py()
        return _query_table(table, key)
    if isinstance(key, slice):
        key = range(*key.indices(indices.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return _query_table(
                table, [i.as_py() for i in indices.fast_slice(key.start, key.stop - key.start).column(0)]
            )
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        table = table.select([key])
        return _query_table(table, indices.column(0).to_pylist())
    if isinstance(key, Iterable):
        return _query_table(table, [indices.fast_slice(i, 1).column(0)[0].as_py() for i in key])

    _raise_bad_key_type(key)


def _query_table(table: Table, key: Union[int, slice, range, str, Iterable]) -> pa.Table:
    """
    Query a pyarrow Table to extract the subtable that correspond to the given key.
    """
    if isinstance(key, int):
        return table.fast_slice(key % table.num_rows, 1)
    if isinstance(key, slice):
        key = range(*key.indices(table.num_rows))
    if isinstance(key, range):
        if _is_range_contiguous(key) and key.start >= 0:
            return table.fast_slice(key.start, key.stop - key.start)
        else:
            pass  # treat as an iterable
    if isinstance(key, str):
        return table.table.drop([column for column in table.column_names if column != key])
    if isinstance(key, Iterable):
        key = np.fromiter(key, np.int64)
        if len(key) == 0:
            return table.table.slice(0, 0)
        # don't use pyarrow.Table.take even for pyarrow >=1.0 (see https://issues.apache.org/jira/browse/ARROW-9773)
        return table.fast_gather(key % table.num_rows)

    _raise_bad_key_type(key)


def _is_array_with_nulls(pa_array: pa.Array) -> bool:
    return pa_array.null_count > 0


class BaseArrowExtractor(Generic[RowFormat, ColumnFormat, BatchFormat]):
    """
    Arrow extractor are used to extract data from pyarrow tables.
    It makes it possible to extract rows, columns and batches.
    These three extractions types have to be implemented.
    """

    def extract_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def extract_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def extract_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError


def _unnest(py_dict: Dict[str, List[T]]) -> Dict[str, T]:
    """Return the first element of a batch (dict) as a row (dict)"""
    return {key: array[0] for key, array in py_dict.items()}


class SimpleArrowExtractor(BaseArrowExtractor[pa.Table, pa.Array, pa.Table]):
    def extract_row(self, pa_table: pa.Table) -> pa.Table:
        return pa_table

    def extract_column(self, pa_table: pa.Table) -> pa.Array:
        return pa_table.column(0)

    def extract_batch(self, pa_table: pa.Table) -> pa.Table:
        return pa_table


class PythonArrowExtractor(BaseArrowExtractor[dict, list, dict]):
    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(pa_table.to_pydict())

    def extract_column(self, pa_table: pa.Table) -> list:
        return pa_table.column(0).to_pylist()

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return pa_table.to_pydict()


class NumpyArrowExtractor(BaseArrowExtractor[dict, np.ndarray, dict]):
    def __init__(self, **np_array_kwargs):
        self.np_array_kwargs = np_array_kwargs

    def extract_row(self, pa_table: pa.Table) -> dict:
        return _unnest(self.extract_batch(pa_table))

    def extract_column(self, pa_table: pa.Table) -> np.ndarray:
        return self._arrow_array_to_numpy(pa_table[pa_table.column_names[0]])

    def extract_batch(self, pa_table: pa.Table) -> dict:
        return {col: self._arrow_array_to_numpy(pa_table[col]) for col in pa_table.column_names}

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: List = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                    not _is_array_with_nulls(chunk) for chunk in pa_array.chunks
                )
                array: List = [
                    row for chunk in pa_array.chunks for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(pa_array.type.storage_dtype, unnest=True)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and not _is_array_with_nulls(pa_array)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()

        if len(array) > 0:
            if any(
                (isinstance(x, np.ndarray) and (x.dtype == object or x.shape != array[0].shape))
                or (isinstance(x, float) and np.isnan(x))
                for x in array
            ):
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
                    return np.asarray(array, dtype=object)
                return np.array(array, copy=False, dtype=object)
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
            return np.asarray(array)
        else:
            return np.array(array, copy=False)


class PandasArrowExtractor(BaseArrowExtractor[pd.DataFrame, pd.Series, pd.DataFrame]):
    def extract_row(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.slice(length=1).to_pandas(types_mapper=pandas_types_mapper)

    def extract_column(self, pa_table: pa.Table) -> pd.Series:
        return pa_table.select([0]).to_pandas(types_mapper=pandas_types_mapper)[pa_table.column_names[0]]

    def extract_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        return pa_table.to_pandas(types_mapper=pandas_types_mapper)


class PythonFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features

    def decode_row(self, row: dict) -> dict:
        return self.features.decode_example(row) if self.features else row

    def decode_column(self, column: list, column_name: str) -> list:
        return self.features.decode_column(column, column_name) if self.features else column

    def decode_batch(self, batch: dict) -> dict:
        return self.features.decode_batch(batch) if self.features else batch


class PandasFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features

    def decode_row(self, row: pd.DataFrame) -> pd.DataFrame:
        decode = (
            {
                column_name: no_op_if_value_is_null(partial(decode_nested_example, feature))
                for column_name, feature in self.features.items()
                if self.features._column_requires_decoding[column_name]
            }
            if self.features
            else {}
        )
        if decode:
            row[list(decode.keys())] = row.transform(decode)
        return row

    def decode_column(self, column: pd.Series, column_name: str) -> pd.Series:
        decode = (
            no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name]))
            if self.features and column_name in self.features and self.features._column_requires_decoding[column_name]
            else None
        )
        if decode:
            column = column.transform(decode)
        return column

    def decode_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.decode_row(batch)


class LazyDict(MutableMapping):
    """A dictionary backed by Arrow data. The values are formatted on-the-fly when accessing the dictionary."""

    def __init__(self, pa_table: pa.Table, formatter: "Formatter"):
        self.pa_table = pa_table
        self.formatter = formatter

        self.data = {key: None for key in pa_table.column_names}
        self.keys_to_format = set(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        value = self.data[key]
        if key in self.keys_to_format:
            value = self.format(key)
            self.data[key] = value
            self.keys_to_format.remove(key)
        return value

    def __setitem__(self, key, value):
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        self.data[key] = value

    def __delitem__(self, key) -> None:
        if key in self.keys_to_format:
            self.keys_to_format.remove(key)
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __repr__(self):
        self._format_all()
        return repr(self.data)

    if config.PY_VERSION >= version.parse("3.9"):
        # merging with the union ("|") operator is supported in Python 3.9+

        def __or__(self, other):
            if isinstance(other, LazyDict):
                inst = self.copy()
                other = other.copy()
                other._format_all()
                inst.keys_to_format -= other.data.keys()
                inst.data = inst.data | other.data
                return inst
            if isinstance(other, dict):
                inst = self.copy()
                inst.keys_to_format -= other.keys()
                inst.data = inst.data | other
                return inst
            return NotImplemented

        def __ror__(self, other):
            if isinstance(other, LazyDict):
                inst = self.copy()
                other = other.copy()
                other._format_all()
                inst.keys_to_format -= other.data.keys()
                inst.data = other.data | inst.data
                return inst
            if isinstance(other, dict):
                inst = self.copy()
                inst.keys_to_format -= other.keys()
                inst.data = other | inst.data
                return inst
            return NotImplemented

        def __ior__(self, other):
            if isinstance(other, LazyDict):
                other = other.copy()
                other._format_all()
                self.keys_to_format -= other.data.keys()
                self.data |= other.data
            else:
                self.keys_to_format -= other.keys()
                self.data |= other
            return self

    def __copy__(self):
        # Identical to `UserDict.__copy__`
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        inst.__dict__["keys_to_format"] = self.__dict__["keys_to_format"].copy()
        return inst

    def copy(self):
        import copy

        return copy.copy(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        raise NotImplementedError

    def format(self, key):
        raise NotImplementedError

    def _format_all(self):
        for key in self.keys_to_format:
            self.data[key] = self.format(key)
        self.keys_to_format.clear()


class LazyRow(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))[0]


class LazyBatch(LazyDict):
    def format(self, key):
        return self.formatter.format_column(self.pa_table.select([key]))


class Formatter(Generic[RowFormat, ColumnFormat, BatchFormat]):
    """
    A formatter is an object that extracts and formats data from pyarrow tables.
    It defines the formatting for rows, columns and batches.
    """

    simple_arrow_extractor = SimpleArrowExtractor
    python_arrow_extractor = PythonArrowExtractor
    numpy_arrow_extractor = NumpyArrowExtractor
    pandas_arrow_extractor = PandasArrowExtractor

    def __init__(self, features: Optional[Features] = None):
        self.features = features
        self.python_features_decoder = PythonFeaturesDecoder(self.features)
        self.pandas_features_decoder = PandasFeaturesDecoder(self.features)

    def __call__(self, pa_table: pa.Table, query_type: str) -> Union[RowFormat, ColumnFormat, BatchFormat]:
        if query_type == "row":
            return self.format_row(pa_table)
        elif query_type == "column":
            return self.format_column(pa_table)
        elif query_type == "batch":
            return self.format_batch(pa_table)

    def format_row(self, pa_table: pa.Table) -> RowFormat:
        raise NotImplementedError

    def format_column(self, pa_table: pa.Table) -> ColumnFormat:
        raise NotImplementedError

    def format_batch(self, pa_table: pa.Table) -> BatchFormat:
        raise NotImplementedError


class TensorFormatter(Formatter[RowFormat, ColumnFormat, BatchFormat]):
    def recursive_tensorize(self, data_struct: dict):
        raise NotImplementedError


class ArrowFormatter(Formatter[pa.Table, pa.Array, pa.Table]):
    def format_row(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_row(pa_table)

    def format_column(self, pa_table: pa.Table) -> pa.Array:
        return self.simple_arrow_extractor().extract_column(pa_table)

    def format_batch(self, pa_table: pa.Table) -> pa.Table:
        return self.simple_arrow_extractor().extract_batch(pa_table)


class PythonFormatter(Formatter[Mapping, list, Mapping]):
    def __init__(self, features=None, lazy=False):
        super().__init__(features)
        self.lazy = lazy

    def format_row(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyRow(pa_table, self)
        row = self.python_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> list:
        column = self.python_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        if self.lazy:
            return LazyBatch(pa_table, self)
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return batch


class PandasFormatter(Formatter[pd.DataFrame, pd.Series, pd.DataFrame]):
    def format_row(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_row(pa_table)
        row = self.pandas_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> pd.Series:
        column = self.pandas_arrow_extractor().extract_column(pa_table)
        column = self.pandas_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> pd.DataFrame:
        row = self.pandas_arrow_extractor().extract_batch(pa_table)
        row = self.pandas_features_decoder.decode_batch(row)
        return row


class CustomFormatter(Formatter[dict, ColumnFormat, dict]):
    """
    A user-defined custom formatter function defined by a ``transform``.
    The transform must take as input a batch of data extracted for an arrow table using the python extractor,
    and return a batch.
    If the output batch is not a dict, then output_all_columns won't work.
    If the ouput batch has several fields, then querying a single column won't work since we don't know which field
    to return.
    """

    def __init__(self, transform: Callable[[dict], dict], features=None, **kwargs):
        super().__init__(features=features)
        self.transform = transform

    def format_row(self, pa_table: pa.Table) -> dict:
        formatted_batch = self.format_batch(pa_table)
        try:
            return _unnest(formatted_batch)
        except Exception as exc:
            raise TypeError(
                f"Custom formatting function must return a dict of sequences to be able to pick a row, but got {formatted_batch}"
            ) from exc

    def format_column(self, pa_table: pa.Table) -> ColumnFormat:
        formatted_batch = self.format_batch(pa_table)
        if hasattr(formatted_batch, "keys"):
            if len(formatted_batch.keys()) > 1:
                raise TypeError(
                    "Tried to query a column but the custom formatting function returns too many columns. "
                    f"Only one column was expected but got columns {list(formatted_batch.keys())}."
                )
        else:
            raise TypeError(
                f"Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}"
            )
        try:
            return formatted_batch[pa_table.column_names[0]]
        except Exception as exc:
            raise TypeError(
                f"Custom formatting function must return a dict to be able to pick a row, but got {formatted_batch}"
            ) from exc

    def format_batch(self, pa_table: pa.Table) -> dict:
        batch = self.python_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        return self.transform(batch)


def _check_valid_column_key(key: str, columns: List[str]) -> None:
    if key not in columns:
        raise KeyError(f"Column {key} not in the dataset. Current columns in the dataset: {columns}")


def _check_valid_index_key(key: Union[int, slice, range, Iterable], size: int) -> None:
    if isinstance(key, int):
        if (key < 0 and key + size < 0) or (key >= size):
            raise IndexError(f"Invalid key: {key} is out of bounds for size {size}")
        return
    elif isinstance(key, slice):
        pass
    elif isinstance(key, range):
        if len(key) > 0:
            _check_valid_index_key(max(key), size=size)
            _check_valid_index_key(min(key), size=size)
    elif isinstance(key, Iterable):
        if len(key) > 0:
            _check_valid_index_key(int(max(key)), size=size)
            _check_valid_index_key(int(min(key)), size=size)
    else:
        _raise_bad_key_type(key)


def key_to_query_type(key: Union[int, slice, range, str, Iterable]) -> str:
    if isinstance(key, int):
        return "row"
    elif isinstance(key, str):
        return "column"
    elif isinstance(key, (slice, range, Iterable)):
        return "batch"
    _raise_bad_key_type(key)


def query_table(
    table: Table,
    key: Union[int, slice, range, str, Iterable],
    indices: Optional[Table] = None,
) -> pa.Table:
    """
    Query a Table to extract the subtable that correspond to the given key.

    Args:
        table (``datasets.table.Table``): The input Table to query from
        key (``Union[int, slice, range, str, Iterable]``): The key can be of different types:
            - an integer i: the subtable containing only the i-th row
            - a slice [i:j:k]: the subtable containing the rows that correspond to this slice
            - a range(i, j, k): the subtable containing the rows that correspond to this range
            - a string c: the subtable containing all the rows but only the column c
            - an iterable l: the subtable that is the concatenation of all the i-th rows for all i in the iterable
        indices (Optional ``datasets.table.Table``): If not None, it is used to re-map the given key to the table rows.
            The indices table must contain one column named "indices" of type uint64.
            This is used in case of shuffling or rows selection.


    Returns:
        ``pyarrow.Table``: the result of the query on the input table
    """
    # Check if key is valid
    if not isinstance(key, (int, slice, range, str, Iterable)):
        try:
            key = operator.index(key)
        except TypeError:
            _raise_bad_key_type(key)
    if isinstance(key, str):
        _check_valid_column_key(key, table.column_names)
    else:
        size = indices.num_rows if indices is not None else table.num_rows
        _check_valid_index_key(key, size)
    # Query the main table
    if indices is None:
        pa_subtable = _query_table(table, key)
    else:
        pa_subtable = _query_table_with_indices_mapping(table, key, indices=indices)
    return pa_subtable


def format_table(
    table: Table,
    key: Union[int, slice, range, str, Iterable],
    formatter: Formatter,
    format_columns: Optional[list] = None,
    output_all_columns=False,
):
    """
    Format a Table depending on the key that was used and a Formatter object.

    Args:
        table (``datasets.table.Table``): The input Table to format
        key (``Union[int, slice, range, str, Iterable]``): Depending on the key that was used, the formatter formats
            the table as either a row, a column or a batch.
        formatter (``datasets.formatting.formatting.Formatter``): Any subclass of a Formatter such as
            PythonFormatter, NumpyFormatter, etc.
        format_columns (:obj:`List[str]`, optional): if not None, it defines the columns that will be formatted using the
            given formatter. Other columns are discarded (unless ``output_all_columns`` is True)
        output_all_columns (:obj:`bool`, defaults to False). If True, the formatted output is completed using the columns
            that are not in the ``format_columns`` list. For these columns, the PythonFormatter is used.


    Returns:
        A row, column or batch formatted object defined by the Formatter:
        - the PythonFormatter returns a dictionary for a row or a batch, and a list for a column.
        - the NumpyFormatter returns a dictionary for a row or a batch, and a np.array for a column.
        - the PandasFormatter returns a pd.DataFrame for a row or a batch, and a pd.Series for a column.
        - the TorchFormatter returns a dictionary for a row or a batch, and a torch.Tensor for a column.
        - the TFFormatter returns a dictionary for a row or a batch, and a tf.Tensor for a column.
    """
    if isinstance(table, Table):
        pa_table = table.table
    else:
        pa_table = table
    query_type = key_to_query_type(key)
    python_formatter = PythonFormatter(features=formatter.features)
    if format_columns is None:
        return formatter(pa_table, query_type=query_type)
    elif query_type == "column":
        if key in format_columns:
            return formatter(pa_table, query_type)
        else:
            return python_formatter(pa_table, query_type=query_type)
    else:
        pa_table_to_format = pa_table.drop(col for col in pa_table.column_names if col not in format_columns)
        formatted_output = formatter(pa_table_to_format, query_type=query_type)
        if output_all_columns:
            if isinstance(formatted_output, MutableMapping):
                pa_table_with_remaining_columns = pa_table.drop(
                    col for col in pa_table.column_names if col in format_columns
                )
                remaining_columns_dict = python_formatter(pa_table_with_remaining_columns, query_type=query_type)
                formatted_output.update(remaining_columns_dict)
            else:
                raise TypeError(
                    f"Custom formatting function must return a dict to work with output_all_columns=True, but got {formatted_output}"
                )
        return formatted_output
