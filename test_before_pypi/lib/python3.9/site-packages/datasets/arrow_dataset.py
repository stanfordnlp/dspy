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

# Lint as: python3
"""Simple Dataset wrapping an Arrow Table."""

import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from typing import Sequence as Sequence_

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from fsspec.core import url_to_fs
from huggingface_hub import (
    CommitInfo,
    CommitOperationAdd,
    CommitOperationDelete,
    DatasetCard,
    DatasetCardData,
    HfApi,
)
from huggingface_hub.hf_api import RepoFile
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map

from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
    FeatureType,
    _align_features,
    _check_if_features_can_be_aligned,
    generate_from_arrow_type,
    pandas_types_mapper,
    require_decoding,
)
from .filesystems import is_remote_filesystem
from .fingerprint import (
    fingerprint_transform,
    format_kwargs_for_fingerprint,
    format_transform_for_fingerprint,
    generate_fingerprint,
    generate_random_fingerprint,
    get_temporary_cache_files_directory,
    is_caching_enabled,
    maybe_register_dataset_for_temp_dir_deletion,
    update_fingerprint,
    validate_fingerprint,
)
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
    InMemoryTable,
    MemoryMappedTable,
    Table,
    _memory_mapped_record_batch_reader_from_file,
    cast_array_to_feature,
    concat_tables,
    embed_table_storage,
    list_table_cache_files,
    table_cast,
    table_iter,
    table_visitor,
)
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
    Literal,
    asdict,
    convert_file_size_to_int,
    glob_pattern_to_regex,
    iflatmap_unordered,
    string_to_dict,
    unique_values,
)
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike


if TYPE_CHECKING:
    import sqlite3

    import polars as pl
    import pyspark
    import sqlalchemy

    from .dataset_dict import DatasetDict
    from .iterable_dataset import IterableDataset

logger = logging.get_logger(__name__)

PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED = (
    "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.parquet"
)


class DatasetInfoMixin:
    """This base class exposes some attributes of DatasetInfo
    at the base level of the Dataset for easy access.
    """

    def __init__(self, info: DatasetInfo, split: Optional[NamedSplit]):
        self._info = info
        self._split = split

    @property
    def info(self):
        """[`~datasets.DatasetInfo`] object containing all the metadata in the dataset."""
        return self._info

    @property
    def split(self):
        """[`~datasets.NamedSplit`] object corresponding to a named dataset split."""
        return self._split

    @property
    def builder_name(self) -> str:
        return self._info.builder_name

    @property
    def citation(self) -> str:
        return self._info.citation

    @property
    def config_name(self) -> str:
        return self._info.config_name

    @property
    def dataset_size(self) -> Optional[int]:
        return self._info.dataset_size

    @property
    def description(self) -> str:
        return self._info.description

    @property
    def download_checksums(self) -> Optional[dict]:
        return self._info.download_checksums

    @property
    def download_size(self) -> Optional[int]:
        return self._info.download_size

    @property
    def features(self) -> Optional[Features]:
        return self._info.features.copy() if self._info.features is not None else None

    @property
    def homepage(self) -> Optional[str]:
        return self._info.homepage

    @property
    def license(self) -> Optional[str]:
        return self._info.license

    @property
    def size_in_bytes(self) -> Optional[int]:
        return self._info.size_in_bytes

    @property
    def supervised_keys(self):
        return self._info.supervised_keys

    @property
    def task_templates(self):
        return self._info.task_templates

    @property
    def version(self):
        return self._info.version


class TensorflowDatasetMixin:
    _TF_DATASET_REFS = set()

    @staticmethod
    def _get_output_signature(
        dataset: "Dataset",
        collate_fn: Callable,
        collate_fn_args: dict,
        cols_to_retain: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        num_test_batches: int = 20,
    ):
        """Private method used by `to_tf_dataset()` to find the shapes and dtypes of samples from this dataset
           after being passed through the collate_fn. Tensorflow needs an exact signature for tf.numpy_function, so
           the only way to do this is to run test batches - the collator may add or rename columns, so we can't figure
           it out just by inspecting the dataset.

        Args:
            dataset (`Dataset`): Dataset to load samples from.
            collate_fn(`bool`): Shuffle the dataset order when loading. Recommended True for training, False for
                validation/evaluation.
            collate_fn(`Callable`): A function or callable object (such as a `DataCollator`) that will collate
                lists of samples into a batch.
            collate_fn_args (`Dict`): A `dict` of keyword arguments to be passed to the
                `collate_fn`.
            batch_size (`int`, optional): The size of batches loaded from the dataset. Used for shape inference.
                Can be None, which indicates that batch sizes can be variable.
            num_test_batches (`int`): The number of batches to load from the dataset for shape inference.

        Returns:
            `dict`: Dict mapping column names to tf.Tensorspec objects
            `dict`: Dict mapping column names to np.dtype objects
        """
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        if len(dataset) == 0:
            raise ValueError("Unable to get the output signature because the dataset is empty.")
        if batch_size is not None:
            batch_size = min(len(dataset), batch_size)
        test_batch_size = 1

        if cols_to_retain is not None:
            cols_to_retain = list(set(cols_to_retain + ["label_ids", "label", "labels"]))

        test_batches = []
        for _ in range(num_test_batches):
            indices = sample(range(len(dataset)), test_batch_size)
            test_batch = dataset[indices]
            if cols_to_retain is not None:
                test_batch = {key: value for key, value in test_batch.items() if key in cols_to_retain}
            test_batch = [{key: value[i] for key, value in test_batch.items()} for i in range(test_batch_size)]
            test_batch = collate_fn(test_batch, **collate_fn_args)
            test_batches.append(test_batch)

        tf_columns_to_signatures = {}
        np_columns_to_dtypes = {}
        for column in test_batches[0].keys():
            raw_arrays = [batch[column] for batch in test_batches]
            # In case the collate_fn returns something strange
            np_arrays = []
            for array in raw_arrays:
                if isinstance(array, np.ndarray):
                    np_arrays.append(array)
                elif isinstance(array, tf.Tensor):
                    np_arrays.append(array.numpy())
                else:
                    np_arrays.append(np.array(array))

            if np.issubdtype(np_arrays[0].dtype, np.integer) or np_arrays[0].dtype == bool:
                tf_dtype = tf.int64
                np_dtype = np.int64
            elif np.issubdtype(np_arrays[0].dtype, np.number):
                tf_dtype = tf.float32
                np_dtype = np.float32
            elif np_arrays[0].dtype.kind == "U":  # Unicode strings
                np_dtype = np.unicode_
                tf_dtype = tf.string
            else:
                raise RuntimeError(
                    f"Unrecognized array dtype {np_arrays[0].dtype}. \n"
                    "Nested types and image/audio types are not supported yet."
                )
            shapes = [array.shape for array in np_arrays]
            static_shape = []
            for dim in range(len(shapes[0])):
                sizes = {shape[dim] for shape in shapes}
                if dim == 0:
                    static_shape.append(batch_size)
                    continue
                if len(sizes) == 1:  # This dimension looks constant
                    static_shape.append(sizes.pop())
                else:  # Use None for variable dimensions
                    static_shape.append(None)
            tf_columns_to_signatures[column] = tf.TensorSpec(shape=static_shape, dtype=tf_dtype)
            np_columns_to_dtypes[column] = np_dtype

        return tf_columns_to_signatures, np_columns_to_dtypes

    def to_tf_dataset(
        self,
        batch_size: Optional[int] = None,
        columns: Optional[Union[str, List[str]]] = None,
        shuffle: bool = False,
        collate_fn: Optional[Callable] = None,
        drop_remainder: bool = False,
        collate_fn_args: Optional[Dict[str, Any]] = None,
        label_cols: Optional[Union[str, List[str]]] = None,
        prefetch: bool = True,
        num_workers: int = 0,
        num_test_batches: int = 20,
    ):
        """Create a `tf.data.Dataset` from the underlying Dataset. This `tf.data.Dataset` will load and collate batches from
        the Dataset, and is suitable for passing to methods like `model.fit()` or `model.predict()`. The dataset will yield
        `dicts` for both inputs and labels unless the `dict` would contain only a single key, in which case a raw
        `tf.Tensor` is yielded instead.

        Args:
            batch_size (`int`, *optional*):
                Size of batches to load from the dataset. Defaults to `None`, which implies that the dataset won't be
                batched, but the returned dataset can be batched later with `tf_dataset.batch(batch_size)`.
            columns (`List[str]` or `str`, *optional*):
                Dataset column(s) to load in the `tf.data.Dataset`.
                Column names that are created by the `collate_fn` and that do not exist in the original dataset can be used.
            shuffle(`bool`, defaults to `False`):
                Shuffle the dataset order when loading. Recommended `True` for training, `False` for
                validation/evaluation.
            drop_remainder(`bool`, defaults to `False`):
                Drop the last incomplete batch when loading. Ensures
                that all batches yielded by the dataset will have the same length on the batch dimension.
            collate_fn(`Callable`, *optional*):
                A function or callable object (such as a `DataCollator`) that will collate
                lists of samples into a batch.
            collate_fn_args (`Dict`, *optional*):
                An optional `dict` of keyword arguments to be passed to the
                `collate_fn`.
            label_cols (`List[str]` or `str`, defaults to `None`):
                Dataset column(s) to load as labels.
                Note that many models compute loss internally rather than letting Keras do it, in which case
                passing the labels here is optional, as long as they're in the input `columns`.
            prefetch (`bool`, defaults to `True`):
                Whether to run the dataloader in a separate thread and maintain
                a small buffer of batches for training. Improves performance by allowing data to be loaded in the
                background while the model is training.
            num_workers (`int`, defaults to `0`):
                Number of workers to use for loading the dataset. Only supported on Python versions >= 3.8.
            num_test_batches (`int`, defaults to `20`):
                Number of batches to use to infer the output signature of the dataset.
                The higher this number, the more accurate the signature will be, but the longer it will take to
                create the dataset.

        Returns:
            `tf.data.Dataset`

        Example:

        ```py
        >>> ds_train = ds["train"].to_tf_dataset(
        ...    columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
        ...    shuffle=True,
        ...    batch_size=16,
        ...    collate_fn=data_collator,
        ... )
        ```
        """
        if config.TF_AVAILABLE:
            import tensorflow as tf
        else:
            raise ImportError("Called a Tensorflow-specific function but Tensorflow is not installed.")

        if (isinstance(columns, list) and len(columns) == 1) or (
            isinstance(label_cols, list) and len(label_cols) == 1
        ):
            warnings.warn(
                "The output of `to_tf_dataset` will change when a passing single element list for `labels` or "
                "`columns` in the next datasets version. To return a tuple structure rather than dict, pass a "
                "single string.\n"
                "Old behaviour: columns=['a'], labels=['labels'] -> (tf.Tensor, tf.Tensor)  \n"
                "             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor)  \n"
                "New behaviour: columns=['a'],labels=['labels'] -> ({'a': tf.Tensor}, {'labels': tf.Tensor})  \n"
                "             : columns='a', labels='labels' -> (tf.Tensor, tf.Tensor) ",
                FutureWarning,
            )

        if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
            logger.warning(
                "Note that to_tf_dataset() loads the data with a generator rather than a full tf.data "
                "pipeline and is not compatible with remote TPU connections. If you encounter errors, please "
                "try using a TPU VM or, if your data can fit in memory, loading it into memory as a dict of "
                "Tensors instead of streaming with to_tf_dataset()."
            )

        if collate_fn is None:
            # Set a very simple default collator that just stacks things together
            collate_fn = minimal_tf_collate_fn
        if collate_fn_args is None:
            collate_fn_args = {}
        if label_cols and not columns:
            raise ValueError("Cannot specify label_cols without specifying columns!")
        if label_cols is None:
            label_cols = []
        elif isinstance(label_cols, str):
            label_cols = [label_cols]
        if len(set(label_cols)) < len(label_cols):
            raise ValueError("List of label_cols contains duplicates.")
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            if len(set(columns)) < len(columns):
                raise ValueError("List of columns contains duplicates.")
            cols_to_retain = list(set(columns + label_cols))
        else:
            cols_to_retain = None  # Indicates keeping all valid columns
            columns = []

        if self.format["type"] not in ["custom", "numpy"]:
            dataset = self.with_format("numpy")
        else:
            dataset = self

        # TODO(Matt, QL): deprecate the retention of label_ids and label

        output_signature, columns_to_np_types = dataset._get_output_signature(
            dataset,
            collate_fn=collate_fn,
            collate_fn_args=collate_fn_args,
            cols_to_retain=cols_to_retain,
            batch_size=batch_size if drop_remainder else None,
            num_test_batches=num_test_batches,
        )

        if "labels" in output_signature:
            if ("label_ids" in columns or "label" in columns) and "labels" not in columns:
                columns = [col for col in columns if col not in ["label_ids", "label"]] + ["labels"]
            if ("label_ids" in label_cols or "label" in label_cols) and "labels" not in label_cols:
                label_cols = [col for col in label_cols if col not in ["label_ids", "label"]] + ["labels"]

        for col in columns:
            if col not in output_signature:
                raise ValueError(f"Column {col} not found in dataset!")

        for col in label_cols:
            if col not in output_signature:
                raise ValueError(f"Label column {col} not found in dataset!")

        if num_workers == 0:
            tf_dataset = dataset_to_tf(
                dataset=dataset,
                cols_to_retain=cols_to_retain,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                columns_to_np_types=columns_to_np_types,
                output_signature=output_signature,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        elif num_workers > 0:
            if batch_size is None:
                raise NotImplementedError(
                    "`batch_size` must be specified when using multiple workers, as unbatched multiprocessing "
                    "is not supported yet. Please provide a `batch_size` if `num_workers` is greater than 0."
                )
            tf_dataset = multiprocess_dataset_to_tf(
                dataset=dataset,
                cols_to_retain=cols_to_retain,
                collate_fn=collate_fn,
                collate_fn_args=collate_fn_args,
                columns_to_np_types=columns_to_np_types,
                output_signature=output_signature,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                num_workers=num_workers,
            )
        else:
            raise ValueError("num_workers must be >= 0")

        def split_features_and_labels(input_batch):
            # TODO(Matt, QL): deprecate returning the dict content when there's only one key
            features = {key: tensor for key, tensor in input_batch.items() if key in columns}
            labels = {key: tensor for key, tensor in input_batch.items() if key in label_cols}
            if len(features) == 1:
                features = list(features.values())[0]
            if len(labels) == 1:
                labels = list(labels.values())[0]
            if isinstance(labels, dict) and len(labels) == 0:
                return features
            else:
                return features, labels

        if cols_to_retain is not None:
            tf_dataset = tf_dataset.map(split_features_and_labels)

        if prefetch:
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Remove a reference to the open Arrow file on delete
        def cleanup_callback(ref):
            dataset.__del__()
            self._TF_DATASET_REFS.remove(ref)

        self._TF_DATASET_REFS.add(weakref.ref(tf_dataset, cleanup_callback))

        return tf_dataset


class DatasetTransformationNotAllowedError(Exception):
    pass


def transmit_format(func):
    """Wrapper for dataset transforms that recreate a new Dataset to transmit the format of the original dataset to the new dataset"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Dataset" = args[0]
            args = args[1:]
        else:
            self: "Dataset" = kwargs.pop("self")
        # don't use self.format since it returns a list of columns for 'columns' even if self_format_columns is None
        unformatted_columns = set(self.column_names) - set(self._format_columns or [])
        self_format = {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self._format_columns,
            "output_all_columns": self._output_all_columns,
        }
        # apply actual function
        out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
        datasets: List["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
        # re-apply format to the output
        for dataset in datasets:
            new_format = self_format.copy()
            if new_format["columns"] is not None:  # new formatted columns = (columns - previously unformatted columns)
                # sort the columns to have a deterministic list of columns that we can compare with `out_format`
                new_format["columns"] = sorted(set(dataset.column_names) - unformatted_columns)
            out_format = {
                "type": dataset._format_type,
                "format_kwargs": dataset._format_kwargs,
                "columns": sorted(dataset._format_columns) if dataset._format_columns is not None else None,
                "output_all_columns": dataset._output_all_columns,
            }
            if out_format != new_format:
                fingerprint = dataset._fingerprint
                dataset.set_format(**new_format)
                dataset._fingerprint = fingerprint
        return out

    wrapper._decorator_name_ = "transmit_format"
    return wrapper


def transmit_tasks(func):
    """Wrapper for dataset transforms that recreate a new Dataset to transmit the task templates of the original dataset to the new dataset"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            self: "Dataset" = args[0]
            args = args[1:]
        else:
            self: "Dataset" = kwargs.pop("self")
        # apply actual function
        out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
        datasets: List["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
        for dataset in datasets:
            # Remove task templates if a column mapping of the template is no longer valid
            if self.info.task_templates is not None:
                dataset.info.task_templates = [
                    template
                    for template in self.info.task_templates
                    if all(
                        dataset._info.features.get(k) == self._info.features.get(k)
                        for k in template.column_mapping.keys()
                    )
                ]
        return out

    wrapper._decorator_name_ = "transmit_tasks"
    return wrapper


def update_metadata_with_features(table: Table, features: Features):
    """To be used in dataset transforms that modify the features of the dataset, in order to update the features stored in the metadata of its schema."""
    features = Features({col_name: features[col_name] for col_name in table.column_names})
    if table.schema.metadata is None or b"huggingface" not in table.schema.metadata:
        pa_metadata = ArrowWriter._build_metadata(DatasetInfo(features=features))
    else:
        metadata = json.loads(table.schema.metadata[b"huggingface"].decode())
        if "info" not in metadata:
            metadata["info"] = asdict(DatasetInfo(features=features))
        else:
            metadata["info"]["features"] = asdict(DatasetInfo(features=features))["features"]
        pa_metadata = {"huggingface": json.dumps(metadata)}
    table = table.replace_schema_metadata(pa_metadata)
    return table


def _check_table(table) -> Table:
    """We check the table type to make sure it's an instance of :class:`datasets.table.Table`"""
    if isinstance(table, pa.Table):
        # for a pyarrow table, we can just consider it as a in-memory table
        # this is here for backward compatibility
        return InMemoryTable(table)
    elif isinstance(table, Table):
        return table
    else:
        raise TypeError(f"Expected a pyarrow.Table or a datasets.table.Table object, but got {table}.")


def _check_column_names(column_names: List[str]):
    """Check the column names to make sure they don't contain duplicates."""
    counter = Counter(column_names)
    if not all(count == 1 for count in counter.values()):
        duplicated_columns = [col for col in counter if counter[col] > 1]
        raise ValueError(f"The table can't have duplicated columns but columns {duplicated_columns} are duplicated.")


def _check_valid_indices_value(index, size):
    if (index < 0 and index + size < 0) or (index >= size):
        raise IndexError(f"Index {index} out of range for dataset of size {size}.")


class NonExistentDatasetError(Exception):
    """Used when we expect the existence of a dataset"""

    pass


class Dataset(DatasetInfoMixin, IndexableMixin, TensorflowDatasetMixin):
    """A Dataset backed by an Arrow table."""

    def __init__(
        self,
        arrow_table: Table,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_table: Optional[Table] = None,
        fingerprint: Optional[str] = None,
    ):
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)
        IndexableMixin.__init__(self)

        self._data: Table = _check_table(arrow_table)
        self._indices: Optional[Table] = _check_table(indices_table) if indices_table is not None else None
        maybe_register_dataset_for_temp_dir_deletion(self)

        self._format_type: Optional[str] = None
        self._format_kwargs: dict = {}
        self._format_columns: Optional[list] = None
        self._output_all_columns: bool = False
        self._fingerprint: str = fingerprint

        # Read metadata

        if self._data.schema.metadata is not None and b"huggingface" in self._data.schema.metadata:
            metadata = json.loads(self._data.schema.metadata[b"huggingface"].decode())
            if (
                "fingerprint" in metadata and self._fingerprint is None
            ):  # try to load fingerprint from the arrow file metadata
                self._fingerprint = metadata["fingerprint"]

        # Infer features if None
        inferred_features = Features.from_arrow_schema(arrow_table.schema)
        if self.info.features is None:
            self.info.features = inferred_features
        else:  # make sure the nested columns are in the right order
            try:
                self.info.features = self.info.features.reorder_fields_as(inferred_features)
            except ValueError as e:
                raise ValueError(
                    f"{e}\nThe 'source' features come from dataset_info.json, and the 'target' ones are those of the dataset arrow file."
                )

        # In case there are types like pa.dictionary that we need to convert to the underlying type

        if self.data.schema != self.info.features.arrow_schema:
            self._data = self.data.cast(self.info.features.arrow_schema)

        # Infer fingerprint if None

        if self._fingerprint is None:
            self._fingerprint = generate_fingerprint(self)

        # Sanity checks

        if self._info.features is None:
            raise ValueError("Features can't be None in a Dataset object")
        if self._fingerprint is None:
            raise ValueError("Fingerprint can't be None in a Dataset object")
        if self.info.features.type != inferred_features.type:
            raise ValueError(
                f"External features info don't match the dataset:\nGot\n{self.info.features}\nwith type\n{self.info.features.type}\n\nbut expected something like\n{inferred_features}\nwith type\n{inferred_features.type}"
            )

        if self._indices is not None:
            if not pa.types.is_unsigned_integer(self._indices.column(0).type):
                raise ValueError(
                    f"indices must be an Arrow table of unsigned integers, current type is {self._indices.column(0).type}"
                )
        _check_column_names(self._data.column_names)

        self._data = update_metadata_with_features(self._data, self._info.features)

    @property
    def features(self) -> Features:
        features = super().features
        if features is None:  # this is already checked in __init__
            raise ValueError("Features can't be None in a Dataset object")
        return features

    @classmethod
    def from_file(
        cls,
        filename: str,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_filename: Optional[str] = None,
        in_memory: bool = False,
    ) -> "Dataset":
        """Instantiate a Dataset backed by an Arrow table at filename.

        Args:
            filename (`str`):
                File name of the dataset.
            info (`DatasetInfo`, *optional*):
                Dataset information, like description, citation, etc.
            split (`NamedSplit`, *optional*):
                Name of the dataset split.
            indices_filename (`str`, *optional*):
                File names of the indices.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            [`Dataset`]
        """
        table = ArrowReader.read_table(filename, in_memory=in_memory)

        if indices_filename is not None:
            indices_pa_table = ArrowReader.read_table(indices_filename, in_memory=in_memory)
        else:
            indices_pa_table = None

        return cls(
            arrow_table=table,
            info=info,
            split=split,
            indices_table=indices_pa_table,
        )

    @classmethod
    def from_buffer(
        cls,
        buffer: pa.Buffer,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        indices_buffer: Optional[pa.Buffer] = None,
    ) -> "Dataset":
        """Instantiate a Dataset backed by an Arrow buffer.

        Args:
            buffer (`pyarrow.Buffer`):
                Arrow buffer.
            info (`DatasetInfo`, *optional*):
                Dataset information, like description, citation, etc.
            split (`NamedSplit`, *optional*):
                Name of the dataset split.
            indices_buffer (`pyarrow.Buffer`, *optional*):
                Indices Arrow buffer.

        Returns:
            [`Dataset`]
        """
        table = InMemoryTable.from_buffer(buffer)

        if indices_buffer is not None:
            indices_table = InMemoryTable.from_buffer(buffer)
        else:
            indices_table = None

        return cls(table, info=info, split=split, indices_table=indices_table)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
        preserve_index: Optional[bool] = None,
    ) -> "Dataset":
        """
        Convert `pandas.DataFrame` to a `pyarrow.Table` to create a [`Dataset`].

        The column types in the resulting Arrow Table are inferred from the dtypes of the `pandas.Series` in the
        DataFrame. In the case of non-object Series, the NumPy dtype is translated to its Arrow equivalent. In the
        case of `object`, we need to guess the datatype by looking at the Python objects in this Series.

        Be aware that Series of the `object` dtype don't carry enough information to always lead to a meaningful Arrow
        type. In the case that we cannot infer a type, e.g. because the DataFrame is of length 0 or the Series only
        contains `None/nan` objects, the type is set to `null`. This behavior can be avoided by constructing explicit
        features and passing it to this function.

        Args:
            df (`pandas.DataFrame`):
                Dataframe that contains the dataset.
            features ([`Features`], *optional*):
                Dataset features.
            info (`DatasetInfo`, *optional*):
                Dataset information, like description, citation, etc.
            split (`NamedSplit`, *optional*):
                Name of the dataset split.
            preserve_index (`bool`, *optional*):
                Whether to store the index as an additional column in the resulting Dataset.
                The default of `None` will store the index as a column, except for `RangeIndex` which is stored as metadata only.
                Use `preserve_index=True` to force it to be stored as a column.

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> ds = Dataset.from_pandas(df)
        ```
        """
        if info is not None and features is not None and info.features != features:
            raise ValueError(
                f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}"
            )
        features = features if features is not None else info.features if info is not None else None
        if info is None:
            info = DatasetInfo()
        info.features = features
        table = InMemoryTable.from_pandas(
            df=df,
            preserve_index=preserve_index,
        )
        if features is not None:
            # more expensive cast than InMemoryTable.from_pandas(..., schema=features.arrow_schema)
            # needed to support the str to Audio conversion for instance
            table = table.cast(features.arrow_schema)
        return cls(table, info=info, split=split)

    @classmethod
    def from_polars(
        cls,
        df: "pl.DataFrame",
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        """
        Collect the underlying arrow arrays in an Arrow Table.

        This operation is mostly zero copy.

        Data types that do copy:
            * CategoricalType

        Args:
            df (`polars.DataFrame`): DataFrame to convert to Arrow Table
            features (`Features`, optional): Dataset features.
            info (`DatasetInfo`, optional): Dataset information, like description, citation, etc.
            split (`NamedSplit`, optional): Name of the dataset split.

        Examples:
        ```py
        >>> ds = Dataset.from_polars(df)
        ```
        """
        if info is not None and features is not None and info.features != features:
            raise ValueError(
                f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}"
            )
        features = features if features is not None else info.features if info is not None else None
        if info is None:
            info = DatasetInfo()
        info.features = features
        table = InMemoryTable(df.to_arrow())
        if features is not None:
            # more expensive cast than InMemoryTable.from_polars(..., schema=features.arrow_schema)
            # needed to support the str to Audio conversion for instance
            table = table.cast(features.arrow_schema)
        return cls(table, info=info, split=split)

    @classmethod
    def from_dict(
        cls,
        mapping: dict,
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        """
        Convert `dict` to a `pyarrow.Table` to create a [`Dataset`].

        Args:
            mapping (`Mapping`):
                Mapping of strings to Arrays or Python lists.
            features ([`Features`], *optional*):
                Dataset features.
            info (`DatasetInfo`, *optional*):
                Dataset information, like description, citation, etc.
            split (`NamedSplit`, *optional*):
                Name of the dataset split.

        Returns:
            [`Dataset`]
        """
        if info is not None and features is not None and info.features != features:
            raise ValueError(
                f"Features specified in `features` and `info.features` can't be different:\n{features}\n{info.features}"
            )
        features = features if features is not None else info.features if info is not None else None
        arrow_typed_mapping = {}
        for col, data in mapping.items():
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                data = cast_array_to_feature(data, features[col]) if features is not None else data
            else:
                data = OptimizedTypedSequence(
                    features.encode_column(data, col) if features is not None else data,
                    type=features[col] if features is not None else None,
                    col=col,
                )
            arrow_typed_mapping[col] = data
        mapping = arrow_typed_mapping
        pa_table = InMemoryTable.from_pydict(mapping=mapping)
        if info is None:
            info = DatasetInfo()
        info.features = features
        if info.features is None:
            info.features = Features(
                {
                    col: generate_from_arrow_type(data.type)
                    if isinstance(data, (pa.Array, pa.ChunkedArray))
                    else data.get_inferred_type()
                    for col, data in mapping.items()
                }
            )
        return cls(pa_table, info=info, split=split)

    @classmethod
    def from_list(
        cls,
        mapping: List[dict],
        features: Optional[Features] = None,
        info: Optional[DatasetInfo] = None,
        split: Optional[NamedSplit] = None,
    ) -> "Dataset":
        """
        Convert a list of dicts to a `pyarrow.Table` to create a [`Dataset`]`.

        Note that the keys of the first entry will be used to determine the dataset columns,
        regardless of what is passed to features.

        Args:
            mapping (`List[dict]`): A list of mappings of strings to row values.
            features (`Features`, optional): Dataset features.
            info (`DatasetInfo`, optional): Dataset information, like description, citation, etc.
            split (`NamedSplit`, optional): Name of the dataset split.

        Returns:
            [`Dataset`]
        """
        # for simplicity and consistency wrt OptimizedTypedSequence we do not use InMemoryTable.from_pylist here
        mapping = {k: [r.get(k) for r in mapping] for k in mapping[0]} if mapping else {}
        return cls.from_dict(mapping, features, info, split)

    @staticmethod
    def from_csv(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Create Dataset from CSV file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the CSV file(s).
            split ([`NamedSplit`], *optional*):
                Split name to be assigned to the dataset.
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`pandas.read_csv`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> ds = Dataset.from_csv('path/to/dataset.csv')
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.csv import CsvDatasetReader

        return CsvDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
            **kwargs,
        ).read()

    @staticmethod
    def from_generator(
        generator: Callable,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        gen_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        split: NamedSplit = Split.TRAIN,
        **kwargs,
    ):
        """Create a Dataset from a generator.

        Args:
            generator (:`Callable`):
                A generator function that `yields` examples.
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            gen_kwargs(`dict`, *optional*):
                Keyword arguments to be passed to the `generator` callable.
                You can define a sharded dataset by passing the list of shards in `gen_kwargs` and setting `num_proc` greater than 1.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.
                If `num_proc` is greater than one, then all list values in `gen_kwargs` must be the same length. These values will be split between calls to the generator. The number of shards will be the minimum of the shortest list in `gen_kwargs` and `num_proc`.

                <Added version="2.7.0"/>
            split ([`NamedSplit`], defaults to `Split.TRAIN`):
                Split name to be assigned to the dataset.

                <Added version="2.21.0"/>
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to :[`GeneratorConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> def gen():
        ...     yield {"text": "Good", "label": 0}
        ...     yield {"text": "Bad", "label": 1}
        ...
        >>> ds = Dataset.from_generator(gen)
        ```

        ```py
        >>> def gen(shards):
        ...     for shard in shards:
        ...         with open(shard) as f:
        ...             for line in f:
        ...                 yield {"line": line}
        ...
        >>> shards = [f"data{i}.txt" for i in range(32)]
        >>> ds = Dataset.from_generator(gen, gen_kwargs={"shards": shards})
        ```
        """
        from .io.generator import GeneratorDatasetInputStream

        return GeneratorDatasetInputStream(
            generator=generator,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            gen_kwargs=gen_kwargs,
            num_proc=num_proc,
            split=split,
            **kwargs,
        ).read()

    @staticmethod
    def from_json(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        field: Optional[str] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Create Dataset from JSON or JSON Lines file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the JSON or JSON Lines file(s).
            split ([`NamedSplit`], *optional*):
                Split name to be assigned to the dataset.
            features ([`Features`], *optional*):
                 Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            field (`str`, *optional*):
                Field name of the JSON file where the dataset is contained in.
            num_proc (`int`, *optional* defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`JsonConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> ds = Dataset.from_json('path/to/dataset.json')
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.json import JsonDatasetReader

        return JsonDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            field=field,
            num_proc=num_proc,
            **kwargs,
        ).read()

    @staticmethod
    def from_parquet(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Create Dataset from Parquet file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the Parquet file(s).
            split (`NamedSplit`, *optional*):
                Split name to be assigned to the dataset.
            features (`Features`, *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            columns (`List[str]`, *optional*):
                If not `None`, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`ParquetConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> ds = Dataset.from_parquet('path/to/dataset.parquet')
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.parquet import ParquetDatasetReader

        return ParquetDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            columns=columns,
            num_proc=num_proc,
            **kwargs,
        ).read()

    @staticmethod
    def from_text(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Create Dataset from text file(s).

        Args:
            path_or_paths (`path-like` or list of `path-like`):
                Path(s) of the text file(s).
            split (`NamedSplit`, *optional*):
                Split name to be assigned to the dataset.
            features (`Features`, *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                This is helpful if the dataset is made of multiple files. Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`TextConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> ds = Dataset.from_text('path/to/dataset.txt')
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.text import TextDatasetReader

        return TextDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            num_proc=num_proc,
            **kwargs,
        ).read()

    @staticmethod
    def from_spark(
        df: "pyspark.sql.DataFrame",
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        keep_in_memory: bool = False,
        cache_dir: str = None,
        working_dir: str = None,
        load_from_cache_file: bool = True,
        **kwargs,
    ):
        """Create a Dataset from Spark DataFrame. Dataset downloading is distributed over Spark workers.

        Args:
            df (`pyspark.sql.DataFrame`):
                The DataFrame containing the desired data.
            split (`NamedSplit`, *optional*):
                Split name to be assigned to the dataset.
            features (`Features`, *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data. When using a multi-node Spark cluster, the cache_dir must be accessible to both
                workers and the driver.
            keep_in_memory (`bool`):
                Whether to copy the data in-memory.
            working_dir (`str`, *optional*)
                Intermediate directory for each Spark worker to write data to before moving it to `cache_dir`. Setting
                a non-NFS intermediate directory may improve performance.
            load_from_cache_file (`bool`):
                Whether to load the dataset from the cache if possible.

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> df = spark.createDataFrame(
        >>>     data=[[1, "Elia"], [2, "Teo"], [3, "Fang"]],
        >>>     columns=["id", "name"],
        >>> )
        >>> ds = Dataset.from_spark(df)
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.spark import SparkDatasetReader

        if sys.platform == "win32":
            raise EnvironmentError("Dataset.from_spark is not currently supported on Windows")

        return SparkDatasetReader(
            df,
            split=split,
            features=features,
            streaming=False,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            working_dir=working_dir,
            load_from_cache_file=load_from_cache_file,
            **kwargs,
        ).read()

    @staticmethod
    def from_sql(
        sql: Union[str, "sqlalchemy.sql.Selectable"],
        con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"],
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ):
        """Create Dataset from SQL query or database table.

        Args:
            sql (`str` or `sqlalchemy.sql.Selectable`):
                SQL query to be executed or a table name.
            con (`str` or `sqlite3.Connection` or `sqlalchemy.engine.Connection` or `sqlalchemy.engine.Connection`):
                A [URI string](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls) used to instantiate a database connection or a SQLite3/SQLAlchemy connection object.
            features ([`Features`], *optional*):
                Dataset features.
            cache_dir (`str`, *optional*, defaults to `"~/.cache/huggingface/datasets"`):
                Directory to cache data.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            **kwargs (additional keyword arguments):
                Keyword arguments to be passed to [`SqlConfig`].

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> # Fetch a database table
        >>> ds = Dataset.from_sql("test_data", "postgres:///db_name")
        >>> # Execute a SQL query on the table
        >>> ds = Dataset.from_sql("SELECT sentence FROM test_data", "postgres:///db_name")
        >>> # Use a Selectable object to specify the query
        >>> from sqlalchemy import select, text
        >>> stmt = select([text("sentence")]).select_from(text("test_data"))
        >>> ds = Dataset.from_sql(stmt, "postgres:///db_name")
        ```

        <Tip>

        The returned dataset can only be cached if `con` is specified as URI string.

        </Tip>
        """
        from .io.sql import SqlDatasetReader

        return SqlDatasetReader(
            sql,
            con,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            **kwargs,
        ).read()

    def __setstate__(self, state):
        self.__dict__.update(state)
        maybe_register_dataset_for_temp_dir_deletion(self)
        return self

    def __del__(self):
        if hasattr(self, "_data"):
            del self._data
        if hasattr(self, "_indices"):
            del self._indices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Here `del` is used to del the pyarrow tables. This properly closes the files used for memory mapped tables
        self.__del__()

    def save_to_disk(
        self,
        dataset_path: PathLike,
        fs="deprecated",
        max_shard_size: Optional[Union[str, int]] = None,
        num_shards: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
    ):
        """
        Saves a dataset to a dataset directory, or in a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

        For [`Image`] and [`Audio`] data:

        All the Image() and Audio() data are stored in the arrow files.
        If you want to store paths or urls, please use the Value("string") type.

        Args:
            dataset_path (`path-like`):
                Path (e.g. `dataset/train`) or remote URI (e.g. `s3://my-bucket/dataset/train`)
                of the dataset directory where the dataset will be saved to.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
                (like `"50MB"`).
            num_shards (`int`, *optional*):
                Number of shards to write. By default the number of shards depends on `max_shard_size` and `num_proc`.

                <Added version="2.8.0"/>
            num_proc (`int`, *optional*):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.8.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Example:

        ```py
        >>> ds.save_to_disk("path/to/dataset/directory")
        >>> ds.save_to_disk("path/to/dataset/directory", max_shard_size="1GB")
        >>> ds.save_to_disk("path/to/dataset/directory", num_shards=1024)
        ```
        """
        if max_shard_size is not None and num_shards is not None:
            raise ValueError(
                "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
            )
        if fs != "deprecated":
            warnings.warn(
                "'fs' was deprecated in favor of 'storage_options' in version 2.8.0 and will be removed in 3.0.0.\n"
                "You can remove this warning by passing 'storage_options=fs.storage_options' instead.",
                FutureWarning,
            )
            storage_options = fs.storage_options

        if self.list_indexes():
            raise ValueError("please remove all the indexes using `dataset.drop_index` before saving a dataset")

        if num_shards is None:
            dataset_nbytes = self._estimate_nbytes()
            max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)
            num_shards = int(dataset_nbytes / max_shard_size) + 1
            num_shards = max(num_shards, num_proc or 1)

        num_proc = num_proc if num_proc is not None else 1
        num_shards = num_shards if num_shards is not None else num_proc

        fs: fsspec.AbstractFileSystem
        fs, _ = url_to_fs(dataset_path, **(storage_options or {}))

        if not is_remote_filesystem(fs):
            parent_cache_files_paths = {
                Path(cache_filename["filename"]).resolve().parent for cache_filename in self.cache_files
            }
            # Check that the dataset doesn't overwrite iself. It can cause a permission error on Windows and a segfault on linux.
            if Path(dataset_path).expanduser().resolve() in parent_cache_files_paths:
                raise PermissionError(
                    f"Tried to overwrite {Path(dataset_path).expanduser().resolve()} but a dataset can't overwrite itself."
                )

        fs.makedirs(dataset_path, exist_ok=True)

        # Get json serializable state
        state = {
            key: self.__dict__[key]
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_output_all_columns",
            ]
        }
        state["_split"] = str(self.split) if self.split is not None else self.split
        state["_data_files"] = [
            {"filename": f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow"} for shard_idx in range(num_shards)
        ]
        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(
                    str(e) + f"\nThe format kwargs must be JSON serializable, but key '{k}' isn't."
                ) from None
        # Get json serializable dataset info
        dataset_info = asdict(self._info)

        shards_done = 0
        pbar = hf_tqdm(
            unit=" examples",
            total=len(self),
            desc=f"Saving the dataset ({shards_done}/{num_shards} shards)",
        )
        kwargs_per_job = (
            {
                "job_id": shard_idx,
                "shard": self.shard(num_shards=num_shards, index=shard_idx, contiguous=True),
                "fpath": posixpath.join(dataset_path, f"data-{shard_idx:05d}-of-{num_shards:05d}.arrow"),
                "storage_options": storage_options,
            }
            for shard_idx in range(num_shards)
        )
        shard_lengths = [None] * num_shards
        shard_sizes = [None] * num_shards
        if num_proc > 1:
            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, Dataset._save_to_disk_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            shards_done += 1
                            pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                            logger.debug(f"Finished writing shard number {job_id} of {num_shards}.")
                            shard_lengths[job_id], shard_sizes[job_id] = content
                        else:
                            pbar.update(content)
        else:
            with pbar:
                for kwargs in kwargs_per_job:
                    for job_id, done, content in Dataset._save_to_disk_single(**kwargs):
                        if done:
                            shards_done += 1
                            pbar.set_description(f"Saving the dataset ({shards_done}/{num_shards} shards)")
                            logger.debug(f"Finished writing shard number {job_id} of {num_shards}.")
                            shard_lengths[job_id], shard_sizes[job_id] = content
                        else:
                            pbar.update(content)
        with fs.open(
            posixpath.join(dataset_path, config.DATASET_STATE_JSON_FILENAME), "w", encoding="utf-8"
        ) as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with fs.open(
            posixpath.join(dataset_path, config.DATASET_INFO_FILENAME), "w", encoding="utf-8"
        ) as dataset_info_file:
            # Sort only the first level of keys, or we might shuffle fields of nested features if we use sort_keys=True
            sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    @staticmethod
    def _save_to_disk_single(job_id: int, shard: "Dataset", fpath: str, storage_options: Optional[dict]):
        batch_size = config.DEFAULT_MAX_BATCH_SIZE

        num_examples_progress_update = 0
        writer = ArrowWriter(
            features=shard.features,
            path=fpath,
            storage_options=storage_options,
            embed_local_files=True,
        )
        try:
            _time = time.time()
            for pa_table in shard.with_format("arrow").iter(batch_size):
                writer.write_table(pa_table)
                num_examples_progress_update += len(pa_table)
                if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                    _time = time.time()
                    yield job_id, False, num_examples_progress_update
                    num_examples_progress_update = 0
        finally:
            yield job_id, False, num_examples_progress_update
            num_examples, num_bytes = writer.finalize()
            writer.close()

        yield job_id, True, (num_examples, num_bytes)

    @staticmethod
    def _build_local_temp_path(uri_or_path: str) -> Path:
        """
        Builds and returns a Path concatenating a local temporary dir with the dir path (or absolute/relative
        path extracted from the uri) passed.

        Args:
            uri_or_path (`str`): Path (e.g. `"dataset/train"`) or remote URI (e.g.
                `"s3://my-bucket/dataset/train"`) to concatenate.

        Returns:
            :class:`Path`: the concatenated path (temp dir + path)
        """
        src_dataset_path = Path(uri_or_path)
        tmp_dir = get_temporary_cache_files_directory()
        return Path(tmp_dir, src_dataset_path.relative_to(src_dataset_path.anchor))

    @staticmethod
    def load_from_disk(
        dataset_path: PathLike,
        fs="deprecated",
        keep_in_memory: Optional[bool] = None,
        storage_options: Optional[dict] = None,
    ) -> "Dataset":
        """
        Loads a dataset that was previously saved using [`save_to_disk`] from a dataset directory, or from a
        filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

        Args:
            dataset_path (`path-like`):
                Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3//my-bucket/dataset/train"`)
                of the dataset directory where the dataset will be loaded from.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem where the dataset will be saved to.

                <Deprecated version="2.8.0">

                `fs` was deprecated in version 2.8.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`

                </Deprecated>

            keep_in_memory (`bool`, defaults to `None`):
                Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                [improve performance](../cache#improve-performance) section.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.8.0"/>

        Returns:
            [`Dataset`] or [`DatasetDict`]:
            - If `dataset_path` is a path of a dataset directory, the dataset requested.
            - If `dataset_path` is a path of a dataset dict directory, a `datasets.DatasetDict` with each split.

        Example:

        ```py
        >>> ds = load_from_disk("path/to/dataset/directory")
        ```
        """
        if fs != "deprecated":
            warnings.warn(
                "'fs' was deprecated in favor of 'storage_options' in version 2.8.0 and will be removed in 3.0.0.\n"
                "You can remove this warning by passing 'storage_options=fs.storage_options' instead.",
                FutureWarning,
            )
            storage_options = fs.storage_options

        fs: fsspec.AbstractFileSystem
        fs, dataset_path = url_to_fs(dataset_path, **(storage_options or {}))

        dest_dataset_path = dataset_path
        dataset_dict_json_path = posixpath.join(dest_dataset_path, config.DATASETDICT_JSON_FILENAME)
        dataset_state_json_path = posixpath.join(dest_dataset_path, config.DATASET_STATE_JSON_FILENAME)
        dataset_info_path = posixpath.join(dest_dataset_path, config.DATASET_INFO_FILENAME)

        dataset_dict_is_file = fs.isfile(dataset_dict_json_path)
        dataset_info_is_file = fs.isfile(dataset_info_path)
        dataset_state_is_file = fs.isfile(dataset_state_json_path)
        if not dataset_info_is_file and not dataset_state_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(
                    f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."
                )
            raise FileNotFoundError(
                f"No such files: '{dataset_info_path}', nor '{dataset_state_json_path}' found. Expected to load a `Dataset` object but provided path is not a `Dataset`."
            )
        if not dataset_info_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(
                    f"No such file: '{dataset_info_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."
                )
            raise FileNotFoundError(
                f"No such file: '{dataset_info_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."
            )
        if not dataset_state_is_file:
            if dataset_dict_is_file:
                raise FileNotFoundError(
                    f"No such file: '{dataset_state_json_path}' found. Expected to load a `Dataset` object, but got a `DatasetDict`. Please use either `datasets.load_from_disk` or `DatasetDict.load_from_disk` instead."
                )
            raise FileNotFoundError(
                f"No such file: '{dataset_state_json_path}'. Expected to load a `Dataset` object but provided path is not a `Dataset`."
            )

        # copies file from filesystem if it is remote filesystem to local filesystem and modifies dataset_path to temp directory containing local copies
        if is_remote_filesystem(fs):
            src_dataset_path = dest_dataset_path
            dest_dataset_path = Dataset._build_local_temp_path(src_dataset_path)
            fs.download(src_dataset_path, dest_dataset_path.as_posix(), recursive=True)
            dataset_state_json_path = posixpath.join(dest_dataset_path, config.DATASET_STATE_JSON_FILENAME)
            dataset_info_path = posixpath.join(dest_dataset_path, config.DATASET_INFO_FILENAME)

        with open(dataset_state_json_path, encoding="utf-8") as state_file:
            state = json.load(state_file)
        with open(dataset_info_path, encoding="utf-8") as dataset_info_file:
            dataset_info = DatasetInfo.from_dict(json.load(dataset_info_file))

        dataset_size = estimate_dataset_size(
            Path(dest_dataset_path, data_file["filename"]) for data_file in state["_data_files"]
        )
        keep_in_memory = keep_in_memory if keep_in_memory is not None else is_small_dataset(dataset_size)
        table_cls = InMemoryTable if keep_in_memory else MemoryMappedTable

        arrow_table = concat_tables(
            thread_map(
                table_cls.from_file,
                [posixpath.join(dest_dataset_path, data_file["filename"]) for data_file in state["_data_files"]],
                tqdm_class=hf_tqdm,
                desc="Loading dataset from disk",
                # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
                disable=len(state["_data_files"]) <= 16 or None,
            )
        )

        split = state["_split"]
        split = Split(split) if split is not None else split

        dataset = Dataset(
            arrow_table=arrow_table,
            info=dataset_info,
            split=split,
            fingerprint=state["_fingerprint"],
        )

        format = {
            "type": state["_format_type"],
            "format_kwargs": state["_format_kwargs"],
            "columns": state["_format_columns"],
            "output_all_columns": state["_output_all_columns"],
        }
        dataset = dataset.with_format(**format)

        return dataset

    @property
    def data(self) -> Table:
        """The Apache Arrow table backing the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.data
        MemoryMappedTable
        text: string
        label: int64
        ----
        text: [["compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .","the soundtrack alone is worth the price of admission .","rodriguez does a splendid job of racial profiling hollywood style--casting excellent latin actors of all ages--a trend long overdue .","beneath the film's obvious determination to shock at any cost lies considerable skill and determination , backed by sheer nerve .","bielinsky is a filmmaker of impressive talent .","so beautifully acted and directed , it's clear that washington most certainly has a new career ahead of him if he so chooses .","a visual spectacle full of stunning images and effects .","a gentle and engrossing character study .","it's enough to watch huppert scheming , with her small , intelligent eyes as steady as any noir villain , and to enjoy the perfectly pitched web of tension that chabrol spins .","an engrossing portrait of uncompromising artists trying to create something original against the backdrop of a corporate music industry that only seems to care about the bottom line .",...,"ultimately , jane learns her place as a girl , softens up and loses some of the intensity that made her an interesting character to begin with .","ah-nuld's action hero days might be over .","it's clear why deuces wild , which was shot two years ago , has been gathering dust on mgm's shelf .","feels like nothing quite so much as a middle-aged moviemaker's attempt to surround himself with beautiful , half-naked women .","when the precise nature of matthew's predicament finally comes into sharp focus , the revelation fails to justify the build-up .","this picture is murder by numbers , and as easy to be bored by as your abc's , despite a few whopping shootouts .","hilarious musical comedy though stymied by accents thick as mud .","if you are into splatter movies , then you will probably have a reasonably good time with the salton sea .","a dull , simple-minded and stereotypical tale of drugs , death and mind-numbing indifference on the inner-city streets .","the feature-length stretch . . . strains the show's concept ."]]
        label: [[1,1,1,1,1,1,1,1,1,1,...,0,0,0,0,0,0,0,0,0,0]]
        ```
        """
        return self._data

    @property
    def cache_files(self) -> List[dict]:
        """The cache files containing the Apache Arrow table backing the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.cache_files
        [{'filename': '/root/.cache/huggingface/datasets/rotten_tomatoes_movie_review/default/1.0.0/40d411e45a6ce3484deed7cc15b82a53dad9a72aafd9f86f8f227134bec5ca46/rotten_tomatoes_movie_review-validation.arrow'}]
        ```
        """
        cache_files = list_table_cache_files(self._data)
        if self._indices is not None:
            cache_files += list_table_cache_files(self._indices)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    @property
    def num_columns(self) -> int:
        """Number of columns in the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.num_columns
        2
        ```
        """
        return self._data.num_columns

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset (same as [`Dataset.__len__`]).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.num_rows
        1066
        ```
        """
        if self._indices is not None:
            return self._indices.num_rows
        return self._data.num_rows

    @property
    def column_names(self) -> List[str]:
        """Names of the columns in the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.column_names
        ['text', 'label']
        ```
        """
        return self._data.column_names

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the dataset (number of columns, number of rows).

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.shape
        (1066, 2)
        ```
        """
        if self._indices is not None:
            return (self._indices.num_rows, self._data.num_columns)
        return self._data.shape

    def unique(self, column: str) -> List:
        """Return a list of the unique elements in a column.

        This is implemented in the low-level backend and as such, very fast.

        Args:
            column (`str`):
                Column name (list all the column names with [`~datasets.Dataset.column_names`]).

        Returns:
            `list`: List of unique elements in the given column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.unique('label')
        [1, 0]
        ```
        """
        if column not in self._data.column_names:
            raise ValueError(f"Column ({column}) not in table columns ({self._data.column_names}).")

        if self._indices is not None and self._indices.num_rows != self._data.num_rows:
            dataset = self.flatten_indices()
        else:
            dataset = self

        return dataset._data.column(column).unique().to_pylist()

    def class_encode_column(self, column: str, include_nulls: bool = False) -> "Dataset":
        """Casts the given column as [`~datasets.features.ClassLabel`] and updates the table.

        Args:
            column (`str`):
                The name of the column to cast (list all the column names with [`~datasets.Dataset.column_names`])
            include_nulls (`bool`, defaults to `False`):
                Whether to include null values in the class labels. If `True`, the null values will be encoded as the `"None"` class label.

                <Added version="1.14.2"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("boolq", split="validation")
        >>> ds.features
        {'answer': Value(dtype='bool', id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        >>> ds = ds.class_encode_column('answer')
        >>> ds.features
        {'answer': ClassLabel(num_classes=2, names=['False', 'True'], id=None),
         'passage': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None)}
        ```
        """
        # Sanity checks
        if column not in self._data.column_names:
            raise ValueError(f"Column ({column}) not in table columns ({self._data.column_names}).")
        src_feat = self._info.features[column]
        if not isinstance(src_feat, Value):
            raise ValueError(
                f"Class encoding is only supported for {Value.__name__} column, and column {column} is {type(src_feat).__name__}."
            )

        if src_feat.dtype != "string" or (include_nulls and None in self.unique(column)):

            def stringify_column(batch):
                batch[column] = [
                    str(sample) if include_nulls or sample is not None else None for sample in batch[column]
                ]
                return batch

            dset = self.map(
                stringify_column,
                batched=True,
                desc="Stringifying the column",
            )
        else:
            dset = self

        # Create the new feature
        class_names = sorted(str(sample) for sample in dset.unique(column) if include_nulls or sample is not None)
        dst_feat = ClassLabel(names=class_names)

        def cast_to_class_labels(batch):
            batch[column] = [
                dst_feat.str2int(str(sample)) if include_nulls or sample is not None else None
                for sample in batch[column]
            ]
            return batch

        new_features = dset.features.copy()
        new_features[column] = dst_feat

        dset = dset.map(
            cast_to_class_labels,
            batched=True,
            features=new_features,
            desc="Casting to class labels",
        )

        return dset

    @fingerprint_transform(inplace=False)
    def flatten(self, new_fingerprint: Optional[str] = None, max_depth=16) -> "Dataset":
        """Flatten the table.
        Each column with a struct type is flattened into one column per struct field.
        Other columns are left unchanged.

        Args:
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A copy of the dataset with flattened columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("squad", split="train")
        >>> ds.features
        {'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None),
         'context': Value(dtype='string', id=None),
         'id': Value(dtype='string', id=None),
         'question': Value(dtype='string', id=None),
         'title': Value(dtype='string', id=None)}
        >>> ds.flatten()
        Dataset({
            features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
            num_rows: 87599
        })
        ```
        """
        dataset = copy.deepcopy(self)
        for depth in range(1, max_depth):
            if any(isinstance(field.type, pa.StructType) for field in dataset._data.schema):
                dataset._data = dataset._data.flatten()
            else:
                break
        dataset.info.features = self._info.features.flatten(max_depth=max_depth)
        dataset.info.features = Features({col: dataset.info.features[col] for col in dataset.data.column_names})
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        logger.info(f'Flattened dataset from depth {depth} to depth {1 if depth + 1 < max_depth else "unknown"}.')
        dataset._fingerprint = new_fingerprint
        return dataset

    def cast(
        self,
        features: Features,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        num_proc: Optional[int] = None,
    ) -> "Dataset":
        """
        Cast the dataset to a new set of features.

        Args:
            features ([`Features`]):
                New features to cast the dataset to.
                The name of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `str` <-> `ClassLabel` you should use [`~datasets.Dataset.map`] to update the Dataset.
            batch_size (`int`, defaults to `1000`):
                Number of examples per batch provided to cast.
                If `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to cast.
            keep_in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.
            load_from_cache_file (`bool`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_name (`str`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running [`~datasets.Dataset.map`].
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes for multiprocessing. By default it doesn't
                use multiprocessing.

        Returns:
            [`Dataset`]: A copy of the dataset with casted features.

        Example:

        ```py
        >>> from datasets import load_dataset, ClassLabel, Value
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> new_features = ds.features.copy()
        >>> new_features['label'] = ClassLabel(names=['bad', 'good'])
        >>> new_features['text'] = Value('large_string')
        >>> ds = ds.cast(new_features)
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='large_string', id=None)}
        ```
        """
        if sorted(features) != sorted(self._data.column_names):
            raise ValueError(
                f"The columns in features ({list(features)}) must be identical "
                f"as the columns in the dataset: {self._data.column_names}"
            )

        schema = features.arrow_schema
        format = self.format
        dataset = self.with_format("arrow")
        # capture the PyArrow version here to make the lambda serializable on Windows
        dataset = dataset.map(
            partial(table_cast, schema=schema),
            batched=True,
            batch_size=batch_size,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
            features=features,
            desc="Casting the dataset",
        )
        dataset = dataset.with_format(**format)
        return dataset

    @fingerprint_transform(inplace=False)
    def cast_column(self, column: str, feature: FeatureType, new_fingerprint: Optional[str] = None) -> "Dataset":
        """Cast column to feature for decoding.

        Args:
            column (`str`):
                Column name.
            feature (`FeatureType`):
                Target feature.
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> ds = ds.cast_column('label', ClassLabel(names=['bad', 'good']))
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='string', id=None)}
        ```
        """
        if hasattr(feature, "decode_example"):
            dataset = copy.deepcopy(self)
            dataset._info.features[column] = feature
            dataset._fingerprint = new_fingerprint
            dataset._data = dataset._data.cast(dataset.features.arrow_schema)
            dataset._data = update_metadata_with_features(dataset._data, dataset.features)
            return dataset
        else:
            features = self.features
            features[column] = feature
            return self.cast(features)

    @transmit_tasks
    @transmit_format
    @fingerprint_transform(inplace=False)
    def remove_columns(self, column_names: Union[str, List[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        """
        Remove one or several column(s) in the dataset and the features associated to them.

        You can also remove a column using [`~datasets.Dataset.map`] with `remove_columns` but the present method
        doesn't copy the data of the remaining columns and is thus faster.

        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to remove.
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A copy of the dataset object without the columns to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds = ds.remove_columns('label')
        Dataset({
            features: ['text'],
            num_rows: 1066
        })
        >>> ds = ds.remove_columns(column_names=ds.column_names) # Removing all the columns returns an empty dataset with the `num_rows` property set to 0
        Dataset({
            features: [],
            num_rows: 0
        })
        ```
        """
        dataset = copy.deepcopy(self)
        if isinstance(column_names, str):
            column_names = [column_names]

        missing_columns = set(column_names) - set(self._data.column_names)
        if missing_columns:
            raise ValueError(
                f"Column name {list(missing_columns)} not in the dataset. "
                f"Current columns in the dataset: {dataset._data.column_names}"
            )

        for column_name in column_names:
            del dataset._info.features[column_name]

        dataset._data = dataset._data.drop(column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @transmit_tasks
    @fingerprint_transform(inplace=False)
    def rename_column(
        self, original_column_name: str, new_column_name: str, new_fingerprint: Optional[str] = None
    ) -> "Dataset":
        """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A copy of the dataset with a renamed column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds = ds.rename_column('label', 'label_new')
        Dataset({
            features: ['text', 'label_new'],
            num_rows: 1066
        })
        ```
        """
        dataset = copy.deepcopy(self)
        if original_column_name not in dataset._data.column_names:
            raise ValueError(
                f"Original column name {original_column_name} not in the dataset. "
                f"Current columns in the dataset: {dataset._data.column_names}"
            )
        if new_column_name in dataset._data.column_names:
            raise ValueError(
                f"New column name {new_column_name} already in the dataset. "
                f"Please choose a column name which is not already in the dataset. "
                f"Current columns in the dataset: {dataset._data.column_names}"
            )
        if not new_column_name:
            raise ValueError("New column name is empty.")

        def rename(columns):
            return [new_column_name if col == original_column_name else col for col in columns]

        new_column_names = rename(self._data.column_names)
        if self._format_columns is not None:
            dataset._format_columns = rename(self._format_columns)

        dataset._info.features = Features(
            {
                new_column_name if col == original_column_name else col: feature
                for col, feature in self._info.features.items()
            }
        )

        dataset._data = dataset._data.rename_columns(new_column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @transmit_tasks
    @fingerprint_transform(inplace=False)
    def rename_columns(self, column_mapping: Dict[str, str], new_fingerprint: Optional[str] = None) -> "Dataset":
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A copy of the dataset with renamed columns

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds = ds.rename_columns({'text': 'text_new', 'label': 'label_new'})
        Dataset({
            features: ['text_new', 'label_new'],
            num_rows: 1066
        })
        ```
        """
        dataset = copy.deepcopy(self)

        extra_columns = set(column_mapping.keys()) - set(dataset.column_names)
        if extra_columns:
            raise ValueError(
                f"Original column names {extra_columns} not in the dataset. "
                f"Current columns in the dataset: {dataset._data.column_names}"
            )

        number_of_duplicates_in_new_columns = len(column_mapping.values()) - len(set(column_mapping.values()))
        if number_of_duplicates_in_new_columns != 0:
            raise ValueError(
                "New column names must all be different, but this column mapping "
                f"has {number_of_duplicates_in_new_columns} duplicates"
            )

        empty_new_columns = [new_col for new_col in column_mapping.values() if not new_col]
        if empty_new_columns:
            raise ValueError(f"New column names {empty_new_columns} are empty.")

        def rename(columns):
            return [column_mapping[col] if col in column_mapping else col for col in columns]

        new_column_names = rename(self._data.column_names)
        if self._format_columns is not None:
            dataset._format_columns = rename(self._format_columns)

        dataset._info.features = Features(
            {
                column_mapping[col] if col in column_mapping else col: feature
                for col, feature in (self._info.features or {}).items()
            }
        )

        dataset._data = dataset._data.rename_columns(new_column_names)
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    @transmit_tasks
    @transmit_format
    @fingerprint_transform(inplace=False)
    def select_columns(self, column_names: Union[str, List[str]], new_fingerprint: Optional[str] = None) -> "Dataset":
        """Select one or several column(s) in the dataset and the features
        associated to them.

        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to keep.
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform. If `None`,
                the new fingerprint is computed using a hash of the previous
                fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A copy of the dataset object which only consists of
            selected columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.select_columns(['text'])
        Dataset({
            features: ['text'],
            num_rows: 1066
        })
        ```
        """
        if isinstance(column_names, str):
            column_names = [column_names]

        missing_columns = set(column_names) - set(self._data.column_names)
        if missing_columns:
            raise ValueError(
                f"Column name {list(missing_columns)} not in the "
                "dataset. Current columns in the dataset: "
                f"{self._data.column_names}."
            )

        dataset = copy.deepcopy(self)
        dataset._data = dataset._data.select(column_names)
        dataset._info.features = Features({col: self._info.features[col] for col in dataset._data.column_names})
        dataset._data = update_metadata_with_features(dataset._data, dataset.features)
        dataset._fingerprint = new_fingerprint
        return dataset

    def __len__(self):
        """Number of rows in the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.__len__
        <bound method Dataset.__len__ of Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })>
        ```
        """
        return self.num_rows

    def __iter__(self):
        """Iterate through the examples.

        If a formatting is set with [`Dataset.set_format`] rows will be returned with the
        selected format.
        """
        if self._indices is None:
            # Fast iteration
            # Benchmark: https://gist.github.com/mariosasko/0248288a2e3a7556873969717c1fe52b (fast_iter_batch)
            format_kwargs = self._format_kwargs if self._format_kwargs is not None else {}
            formatter = get_formatter(self._format_type, features=self._info.features, **format_kwargs)
            batch_size = config.ARROW_READER_BATCH_SIZE_IN_DATASET_ITER
            for pa_subtable in table_iter(self.data, batch_size=batch_size):
                for i in range(pa_subtable.num_rows):
                    pa_subtable_ex = pa_subtable.slice(i, 1)
                    formatted_output = format_table(
                        pa_subtable_ex,
                        0,
                        formatter=formatter,
                        format_columns=self._format_columns,
                        output_all_columns=self._output_all_columns,
                    )
                    yield formatted_output
        else:
            for i in range(self.num_rows):
                yield self._getitem(
                    i,
                )

    def iter(self, batch_size: int, drop_last_batch: bool = False):
        """Iterate through the batches of size `batch_size`.

        If a formatting is set with [`~datasets.Dataset.set_format`] rows will be returned with the
        selected format.

        Args:
            batch_size (:obj:`int`): size of each batch to yield.
            drop_last_batch (:obj:`bool`, default `False`): Whether a last batch smaller than the batch_size should be
                dropped
        """
        if self._indices is None:
            # Fast iteration
            # Benchmark: https://gist.github.com/mariosasko/0248288a2e3a7556873969717c1fe52b (fast_iter_batch)
            format_kwargs = self._format_kwargs if self._format_kwargs is not None else {}
            formatter = get_formatter(self._format_type, features=self._info.features, **format_kwargs)
            for pa_subtable in table_iter(self.data, batch_size=batch_size, drop_last_batch=drop_last_batch):
                formatted_batch = format_table(
                    pa_subtable,
                    range(pa_subtable.num_rows),
                    formatter=formatter,
                    format_columns=self._format_columns,
                    output_all_columns=self._output_all_columns,
                )
                yield formatted_batch
        else:
            num_rows = self.num_rows if not drop_last_batch else self.num_rows // batch_size * batch_size
            for i in range(0, num_rows, batch_size):
                yield self._getitem(
                    slice(i, i + batch_size),
                )

    def __repr__(self):
        return f"Dataset({{\n    features: {list(self._info.features.keys())},\n    num_rows: {self.num_rows}\n}})"

    @property
    def format(self):
        return {
            "type": self._format_type,
            "format_kwargs": self._format_kwargs,
            "columns": self.column_names if self._format_columns is None else self._format_columns,
            "output_all_columns": self._output_all_columns,
        }

    @contextlib.contextmanager
    def formatted_as(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """To be used in a `with` statement. Set `__getitem__` return format (type and columns).

        Args:
            type (`str`, *optional*):
                Output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__`` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """
        old_format_type = self._format_type
        old_format_kwargs = self._format_kwargs
        old_format_columns = self._format_columns
        old_output_all_columns = self._output_all_columns
        try:
            self.set_format(type, columns, output_all_columns, **format_kwargs)
            yield
        finally:
            self.set_format(old_format_type, old_format_columns, old_output_all_columns, **old_format_kwargs)

    @fingerprint_transform(inplace=True)
    def set_format(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """Set `__getitem__` return format (type and columns). The data formatting is applied on-the-fly.
        The format `type` (for example "numpy") is used to format batches when using `__getitem__`.
        It's also possible to use custom transforms for formatting using [`~datasets.Dataset.set_transform`].

        Args:
            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        It is possible to call [`~datasets.Dataset.map`] after calling `set_format`. Since `map` may add new columns, then the list of formatted columns
        gets updated. In this case, if you apply `map` on a dataset to add a new column, then this column will be formatted as:

            ```
            new formatted columns = (all columns - previously unformatted columns)
            ```

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type='numpy', columns=['text', 'label'])
        >>> ds.format
        {'type': 'numpy',
        'format_kwargs': {},
        'columns': ['text', 'label'],
        'output_all_columns': False}
        ```
        """
        format_kwargs.update(format_kwargs.pop("format_kwargs", {}))  # allow to use self.set_format(**self.format)

        # Check that the format_type and format_kwargs are valid and make it possible to have a Formatter
        type = get_format_type_from_alias(type)
        get_formatter(type, features=self._info.features, **format_kwargs)

        # Check filter column
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, tuple):
            columns = list(columns)
        if columns is not None:
            missing_columns = set(columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"Columns {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}"
                )
        if columns is not None:
            columns = columns.copy()  # Ensures modifications made to the list after this call don't cause bugs

        self._format_type = type
        self._format_kwargs = format_kwargs
        self._format_columns = columns
        self._output_all_columns = output_all_columns
        logger.debug(
            "Set __getitem__(key) output type to %s for %s columns "
            " (when key is int or slice) and %s output other (un-formatted) columns.",
            "python objects" if type is None else type,
            "no" if columns is None else str(columns),
            "do" if output_all_columns else "don't",
        )

    def reset_format(self):
        """Reset `__getitem__` return format to python objects and all columns.

        Same as `self.set_format()`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds.set_format(type='numpy', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds.format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'numpy'}
        >>> ds.reset_format()
        >>> ds.format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        ```
        """
        self.set_format()

    def set_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[List] = None,
        output_all_columns: bool = False,
    ):
        """Set `__getitem__` return format using this transform. The transform is applied on-the-fly on batches when `__getitem__` is called.
        As [`~datasets.Dataset.set_format`], this can be reset using [`~datasets.Dataset.reset_format`].

        Args:
            transform (`Callable`, *optional*):
                User-defined formatting transform, replaces the format defined by [`~datasets.Dataset.set_format`].
                A formatting function is a callable that takes a batch (as a `dict`) as input and returns a batch.
                This function is applied right before returning the objects in `__getitem__`.
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
                If set to True, then the other un-formatted columns are kept with the output of the transform.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> def encode(batch):
        ...     return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
        >>> ds.set_transform(encode)
        >>> ds[0]
        {'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1]),
         'input_ids': tensor([  101, 29353,  2135, 15102,  1996,  9428, 20868,  2890,  8663,  6895,
                 20470,  2571,  3663,  2090,  4603,  3017,  3008,  1998,  2037, 24211,
                 5637,  1998, 11690,  2336,  1012,   102]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0])}
        ```
        """
        self.set_format("custom", columns=columns, output_all_columns=output_all_columns, transform=transform)

    def with_format(
        self,
        type: Optional[str] = None,
        columns: Optional[List] = None,
        output_all_columns: bool = False,
        **format_kwargs,
    ):
        """Set `__getitem__` return format (type and columns). The data formatting is applied on-the-fly.
        The format `type` (for example "numpy") is used to format batches when using `__getitem__`.

        It's also possible to use custom transforms for formatting using [`~datasets.Dataset.with_transform`].

        Contrary to [`~datasets.Dataset.set_format`], `with_format` returns a new [`Dataset`] object.

        Args:
            type (`str`, *optional*):
                Either output type selected in `[None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']`.
                `None` means `__getitem__` returns python objects (default).
            columns (`List[str]`, *optional*):
                Columns to format in the output.
                `None` means `__getitem__` returns all columns (default).
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
            **format_kwargs (additional keyword arguments):
                Keywords arguments passed to the convert function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, padding=True), batched=True)
        >>> ds.format
        {'columns': ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': None}
        >>> ds = ds.with_format(type='tensorflow', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        >>> ds.format
        {'columns': ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
         'format_kwargs': {},
         'output_all_columns': False,
         'type': 'tensorflow'}
        ```
        """
        dataset = copy.deepcopy(self)
        dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **format_kwargs)
        return dataset

    def with_transform(
        self,
        transform: Optional[Callable],
        columns: Optional[List] = None,
        output_all_columns: bool = False,
    ):
        """Set `__getitem__` return format using this transform. The transform is applied on-the-fly on batches when `__getitem__` is called.

        As [`~datasets.Dataset.set_format`], this can be reset using [`~datasets.Dataset.reset_format`].

        Contrary to [`~datasets.Dataset.set_transform`], `with_transform` returns a new [`Dataset`] object.

        Args:
            transform (`Callable`, `optional`):
                User-defined formatting transform, replaces the format defined by [`~datasets.Dataset.set_format`].
                A formatting function is a callable that takes a batch (as a `dict`) as input and returns a batch.
                This function is applied right before returning the objects in `__getitem__`.
            columns (`List[str]`, `optional`):
                Columns to format in the output.
                If specified, then the input batch of the transform only contains those columns.
            output_all_columns (`bool`, defaults to `False`):
                Keep un-formatted columns as well in the output (as python objects).
                If set to `True`, then the other un-formatted columns are kept with the output of the transform.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> from transformers import AutoTokenizer
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> def encode(example):
        ...     return tokenizer(example["text"], padding=True, truncation=True, return_tensors='pt')
        >>> ds = ds.with_transform(encode)
        >>> ds[0]
        {'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]),
         'input_ids': tensor([  101, 18027, 16310, 16001,  1103,  9321,   178, 11604,  7235,  6617,
                 1742,  2165,  2820,  1206,  6588, 22572, 12937,  1811,  2153,  1105,
                 1147, 12890, 19587,  6463,  1105, 15026,  1482,   119,   102]),
         'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0])}
        ```
        """
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    @deprecated()
    def prepare_for_task(self, task: Union[str, TaskTemplate], id: int = 0) -> "Dataset":
        """
        Prepare a dataset for the given task by casting the dataset's [`Features`] to standardized column names and types as detailed in [`datasets.tasks`](./task_templates).

        Casts [`datasets.DatasetInfo.features`] according to a task-specific schema. Intended for single-use only, so all task templates are removed from [`datasets.DatasetInfo.task_templates`] after casting.

        Args:
            task (`Union[str, TaskTemplate]`):
                The task to prepare the dataset for during training and evaluation. If `str`, supported tasks include:

                - `"text-classification"`
                - `"question-answering"`

                If [`TaskTemplate`], must be one of the task templates in [`datasets.tasks`](./task_templates).
            id (`int`, defaults to `0`):
                The id required to unambiguously identify the task template when multiple task templates of the same type are supported.
        """
        # TODO(lewtun): Add support for casting nested features like answers.text and answers.answer_start in SQuAD
        if isinstance(task, str):
            tasks = [template.task for template in (self.info.task_templates or [])]
            compatible_templates = [template for template in (self.info.task_templates or []) if template.task == task]
            if not compatible_templates:
                raise ValueError(
                    f"Task {task} is not compatible with this dataset! Available tasks: {list(unique_values(tasks))}"
                )

            if not 0 <= id < len(compatible_templates):
                templates_list_str = "\n".join(
                    f"- `{idx}` for task {template}" for idx, template in enumerate(compatible_templates)
                )
                raise ValueError(
                    f"Id {id} for task {task} is not in a valid range. Supported ids:\n{templates_list_str}"
                )
            template = compatible_templates[id]
        elif isinstance(task, TaskTemplate):
            template = task
        else:
            raise ValueError(
                f"Expected a `str` or `datasets.TaskTemplate` object but got task {task} with type {type(task)}."
            )
        template = template.align_with_features(self.info.features)
        column_mapping = template.column_mapping
        columns_to_drop = [column for column in self.column_names if column not in column_mapping]
        dataset = self.remove_columns(columns_to_drop)
        dataset = dataset.rename_columns(column_mapping)
        # We found a template so now flush `DatasetInfo` to skip the template update in `DatasetInfo.__post_init__`
        dataset.info.task_templates = None
        dataset = dataset.cast(features=template.features)
        return dataset

    def _getitem(self, key: Union[int, slice, str, ListLike[int]], **kwargs) -> Union[Dict, List]:
        """
        Can be used to index columns (by string names) or rows (by integer, slice, or list-like of integer indices)
        """
        if isinstance(key, bool):
            raise TypeError("dataset index must be int, str, slice or collection of int, not bool")
        format_type = kwargs["format_type"] if "format_type" in kwargs else self._format_type
        format_columns = kwargs["format_columns"] if "format_columns" in kwargs else self._format_columns
        output_all_columns = (
            kwargs["output_all_columns"] if "output_all_columns" in kwargs else self._output_all_columns
        )
        format_kwargs = kwargs["format_kwargs"] if "format_kwargs" in kwargs else self._format_kwargs
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        formatter = get_formatter(format_type, features=self._info.features, **format_kwargs)
        pa_subtable = query_table(self._data, key, indices=self._indices)
        formatted_output = format_table(
            pa_subtable, key, formatter=formatter, format_columns=format_columns, output_all_columns=output_all_columns
        )
        return formatted_output

    @overload
    def __getitem__(self, key: Union[int, slice, Iterable[int]]) -> Dict:  # noqa: F811
        ...

    @overload
    def __getitem__(self, key: str) -> List:  # noqa: F811
        ...

    def __getitem__(self, key):  # noqa: F811
        """Can be used to index columns (by string names) or rows (by integer index or iterable of indices or bools)."""
        return self._getitem(key)

    def __getitems__(self, keys: List) -> List:
        """Can be used to get a batch using a list of integers indices."""
        batch = self.__getitem__(keys)
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)]

    def cleanup_cache_files(self) -> int:
        """Clean up all cache files in the dataset cache directory, excepted the currently used cache file if there is
        one.

        Be careful when running this command that no other process is currently using other cache files.

        Returns:
            `int`: Number of removed files.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.cleanup_cache_files()
        10
        ```
        """
        current_cache_files = [os.path.abspath(cache_file["filename"]) for cache_file in self.cache_files]
        if not current_cache_files:
            return 0
        cache_directory = os.path.dirname(current_cache_files[0])
        logger.info(f"Listing files in {cache_directory}")
        files: List[str] = os.listdir(cache_directory)
        files_to_remove = []
        for f_name in files:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            if f_name.startswith("cache-") and f_name.endswith(".arrow"):
                if full_name in current_cache_files:
                    logger.info(f"Keeping currently used cache file at {full_name}")
                    continue
                files_to_remove.append(full_name)
        for file_path in files_to_remove:
            logger.info(f"Removing {file_path}")
            os.remove(file_path)
        return len(files_to_remove)

    def _get_cache_file_path(self, fingerprint):
        if is_caching_enabled() and self.cache_files:
            cache_file_name = "cache-" + fingerprint + ".arrow"
            cache_directory = os.path.dirname(self.cache_files[0]["filename"])
        else:
            cache_file_name = "cache-" + generate_random_fingerprint() + ".arrow"
            cache_directory = get_temporary_cache_files_directory()
        cache_file_path = os.path.join(cache_directory, cache_file_name)
        return cache_file_path

    @transmit_tasks
    @transmit_format
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":
        """
        Apply a function to all the examples in the table (individually or in batches) and update the table.
        If your function returns a column that already exists, then it overwrites it.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
        - If batched is `True` and `batch_size` is `n > 1`, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.

        Args:
            function (`Callable`): Function with one of the following signatures:

                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: `lambda x: x`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the
                signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
                The columns to be passed into `function`
                as positional arguments. If `None`, a `dict` mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
                If `batch_size <= 0` or `batch_size == None`, provide the full dataset as a single batch to `function`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_name (`str`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific Features to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Disallow null values in the table.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`.
            num_proc (`int`, *optional*, defaults to `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially.
            suffix_template (`str`):
                If `cache_file_name` is specified, then this suffix
                will be added at the end of the base name of each. Defaults to `"_{rank:05d}_of_{num_proc:05d}"`. For example, if `cache_file_name` is "processed.arrow", then for
                `rank=1` and `num_proc=4`, the resulting file would be `"processed_00001_of_00004.arrow"` for the default suffix.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while mapping examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> ds[0:3]["text"]
        ['Review: compassionately explores the seemingly irreconcilable situation between conservative christian parents and their estranged gay and lesbian children .',
         'Review: the soundtrack alone is worth the price of admission .',
         'Review: rodriguez does a splendid job of racial profiling hollywood style--casting excellent latin actors of all ages--a trend long overdue .']

        # process a batch of examples
        >>> ds = ds.map(lambda example: tokenizer(example["text"]), batched=True)
        # set number of processors
        >>> ds = ds.map(add_prefix, num_proc=4)
        ```
        """
        if keep_in_memory and cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `cache_file_name` but not both.")

        if num_proc is not None and num_proc <= 0:
            raise ValueError("num_proc must be an integer > 0.")

        # If the array is empty we do nothing (but we make sure to handle an empty indices mapping and remove the requested columns anyway)
        if len(self) == 0:
            if self._indices is not None:  # empty indices mapping
                self = Dataset(
                    self.data.slice(0, 0),
                    info=self.info.copy(),
                    split=self.split,
                    fingerprint=new_fingerprint,
                )
            if remove_columns:
                return self.remove_columns(remove_columns)
            else:
                return self

        if function is None:
            function = lambda x: x  # noqa: E731

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        if input_columns is not None:
            missing_columns = set(input_columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"Input column {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}"
                )

        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]

        if remove_columns is not None:
            missing_columns = set(remove_columns) - set(self._data.column_names)
            if missing_columns:
                raise ValueError(
                    f"Column to remove {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}"
                )

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if fn_kwargs is None:
            fn_kwargs = {}

        if num_proc is not None and num_proc > len(self):
            num_proc = len(self)
            logger.warning(
                f"num_proc must be <= {len(self)}. Reducing num_proc to {num_proc} for dataset of size {len(self)}."
            )

        dataset_kwargs = {
            "shard": self,
            "function": function,
            "with_indices": with_indices,
            "with_rank": with_rank,
            "input_columns": input_columns,
            "batched": batched,
            "batch_size": batch_size,
            "drop_last_batch": drop_last_batch,
            "remove_columns": remove_columns,
            "keep_in_memory": keep_in_memory,
            "writer_batch_size": writer_batch_size,
            "features": features,
            "disable_nullable": disable_nullable,
            "fn_kwargs": fn_kwargs,
        }

        if new_fingerprint is None:
            # we create a unique hash from the function,
            # current dataset file and the mapping args
            transform = format_transform_for_fingerprint(Dataset._map_single)
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(Dataset._map_single, (), dataset_kwargs)
            kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
            new_fingerprint = update_fingerprint(self._fingerprint, transform, kwargs_for_fingerprint)
        else:
            validate_fingerprint(new_fingerprint)
        dataset_kwargs["new_fingerprint"] = new_fingerprint

        if self.cache_files:
            if cache_file_name is None:
                cache_file_name = self._get_cache_file_path(new_fingerprint)
        dataset_kwargs["cache_file_name"] = cache_file_name

        def load_processed_shard_from_cache(shard_kwargs):
            """Load a processed shard from cache if it exists, otherwise throw an error."""
            shard = shard_kwargs["shard"]
            # Check if we've already cached this computation (indexed by a hash)
            if shard_kwargs["cache_file_name"] is not None:
                if os.path.exists(shard_kwargs["cache_file_name"]) and load_from_cache_file:
                    info = shard.info.copy()
                    info.features = features
                    info.task_templates = None
                    return Dataset.from_file(shard_kwargs["cache_file_name"], info=info, split=shard.split)
            raise NonExistentDatasetError

        num_shards = num_proc if num_proc is not None else 1
        if batched and drop_last_batch:
            pbar_total = len(self) // num_shards // batch_size * num_shards * batch_size
        else:
            pbar_total = len(self)

        shards_done = 0
        if num_proc is None or num_proc == 1:
            transformed_dataset = None
            try:
                transformed_dataset = load_processed_shard_from_cache(dataset_kwargs)
                logger.info(f"Loading cached processed dataset at {dataset_kwargs['cache_file_name']}")
            except NonExistentDatasetError:
                pass
            if transformed_dataset is None:
                with hf_tqdm(
                    unit=" examples",
                    total=pbar_total,
                    desc=desc or "Map",
                ) as pbar:
                    for rank, done, content in Dataset._map_single(**dataset_kwargs):
                        if done:
                            shards_done += 1
                            logger.debug(f"Finished processing shard number {rank} of {num_shards}.")
                            transformed_dataset = content
                        else:
                            pbar.update(content)
            assert transformed_dataset is not None, "Failed to retrieve the result from map"
            # update fingerprint if the dataset changed
            if transformed_dataset._fingerprint != self._fingerprint:
                transformed_dataset._fingerprint = new_fingerprint
            return transformed_dataset
        else:

            def format_cache_file_name(
                cache_file_name: Optional[str],
                rank: Union[int, Literal["*"]],  # noqa: F722
            ) -> Optional[str]:
                if not cache_file_name:
                    return cache_file_name
                sep = cache_file_name.rindex(".")
                base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
                if isinstance(rank, int):
                    cache_file_name = base_name + suffix_template.format(rank=rank, num_proc=num_proc) + extension
                    logger.info(f"Process #{rank} will write at {cache_file_name}")
                else:
                    cache_file_name = (
                        base_name
                        + suffix_template.replace("{rank:05d}", "{rank}").format(rank=rank, num_proc=num_proc)
                        + extension
                    )
                return cache_file_name

            def format_new_fingerprint(new_fingerprint: str, rank: int) -> str:
                new_fingerprint = new_fingerprint + suffix_template.format(rank=rank, num_proc=num_proc)
                validate_fingerprint(new_fingerprint)
                return new_fingerprint

            prev_env = deepcopy(os.environ)
            # check if parallelism if off
            # from https://github.com/huggingface/tokenizers/blob/bb668bc439dc34389b71dbb8ce0c597f15707b53/tokenizers/src/utils/parallelism.rs#L22
            if prev_env.get("TOKENIZERS_PARALLELISM", "false").lower() not in (
                "",
                "off",
                "false",
                "f",
                "no",
                "n",
                "0",
            ):
                logger.warning("Setting TOKENIZERS_PARALLELISM=false for forked processes.")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            shards = [
                self.shard(num_shards=num_proc, index=rank, contiguous=True, keep_in_memory=keep_in_memory)
                for rank in range(num_proc)
            ]
            kwargs_per_job = [
                {
                    **dataset_kwargs,
                    "shard": shards[rank],
                    "cache_file_name": format_cache_file_name(cache_file_name, rank),
                    "rank": rank,
                    "offset": sum(len(s) for s in shards[:rank]),
                    "new_fingerprint": format_new_fingerprint(new_fingerprint, rank),
                }
                for rank in range(num_shards)
            ]

            transformed_shards = [None] * num_shards
            for rank in range(num_shards):
                try:
                    transformed_shards[rank] = load_processed_shard_from_cache(kwargs_per_job[rank])
                    kwargs_per_job[rank] = None
                except NonExistentDatasetError:
                    pass

            kwargs_per_job = [kwargs for kwargs in kwargs_per_job if kwargs is not None]

            # We try to create a pool with as many workers as dataset not yet cached.
            if kwargs_per_job:
                if len(kwargs_per_job) < num_shards:
                    logger.info(
                        f"Reprocessing {len(kwargs_per_job)}/{num_shards} shards because some of them were missing from the cache."
                    )
                with Pool(len(kwargs_per_job)) as pool:
                    os.environ = prev_env
                    logger.info(f"Spawning {num_proc} processes")
                    with hf_tqdm(
                        unit=" examples",
                        total=pbar_total,
                        desc=(desc or "Map") + f" (num_proc={num_proc})",
                    ) as pbar:
                        for rank, done, content in iflatmap_unordered(
                            pool, Dataset._map_single, kwargs_iterable=kwargs_per_job
                        ):
                            if done:
                                shards_done += 1
                                logger.debug(f"Finished processing shard number {rank} of {num_shards}.")
                                transformed_shards[rank] = content
                            else:
                                pbar.update(content)
                # Avoids PermissionError on Windows (the error: https://github.com/huggingface/datasets/actions/runs/4026734820/jobs/6921621805)
                for kwargs in kwargs_per_job:
                    del kwargs["shard"]
            else:
                logger.info(f"Loading cached processed dataset at {format_cache_file_name(cache_file_name, '*')}")
            assert (
                None not in transformed_shards
            ), f"Failed to retrieve results from map: result list {transformed_shards} still contains None - at least one worker failed to return its results"
            logger.info(f"Concatenating {num_proc} shards")
            result = _concatenate_map_style_datasets(transformed_shards)
            # update fingerprint if the dataset changed
            if any(
                transformed_shard._fingerprint != shard._fingerprint
                for transformed_shard, shard in zip(transformed_shards, shards)
            ):
                result._fingerprint = new_fingerprint
            else:
                result._fingerprint = self._fingerprint
            return result

    @staticmethod
    def _map_single(
        shard: "Dataset",
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[List[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        new_fingerprint: Optional[str] = None,
        rank: Optional[int] = None,
        offset: int = 0,
    ) -> Iterable[Tuple[int, bool, Union[int, "Dataset"]]]:
        """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does update examples).

        Args:
            shard (`datasets.Dataset`): Dataset to map the transform on.
            function (`Callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> Dict[str, Any]` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> Dict[str, List]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: lambda x: x
            with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, default `False`): Provide process rank to `function`. Note that in this case the signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[List[str]]`, defaults to `None`): The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (`int`, optional, defaults to `1000`): Number of examples per batch provided to `function` if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            drop_last_batch (`bool`, default: `False`): Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[List[str]]`, defaults to `None`): Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a cache file.
            cache_file_name (`str`, optional, defaults to `None`): Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
            features (`Optional[datasets.Features]`, defaults to `None`): Use a specific Features to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`): Disallow null values in the table.
            fn_kwargs (`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            new_fingerprint (`str`, optional, defaults to `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
            rank: (`int`, optional, defaults to `None`): If specified, this is the process rank when doing multiprocessing
            offset: (`int`, defaults to 0): If specified, this is an offset applied to the indices passed to `function` if `with_indices=True`.
        """
        if fn_kwargs is None:
            fn_kwargs = {}

        # If we do batch computation but no batch size is provided, default to the full dataset
        if batched and (batch_size is None or batch_size <= 0):
            batch_size = shard.num_rows

        # We set this variable to True after processing the first example/batch in
        # `apply_function_on_filtered_inputs` if the map function returns a dict.
        # If set to False, no new arrow table will be created

        update_data = None

        format_kwargs = shard._format_kwargs.copy()
        # Lazy formatting is only available for the default format (None/python)
        if not input_columns and shard._format_type is None:
            format_kwargs["lazy"] = True
        input_formatter = get_formatter(
            shard._format_type,
            features=shard.features,
            **format_kwargs,
        )

        class NumExamplesMismatchError(Exception):
            pass

        def validate_function_output(processed_inputs, indices):
            """Validate output of the map function."""
            if processed_inputs is not None and not isinstance(processed_inputs, (Mapping, pa.Table, pd.DataFrame)):
                raise TypeError(
                    f"Provided `function` which is applied to all elements of table returns a variable of type {type(processed_inputs)}. Make sure provided `function` returns a variable of type `dict` (or a pyarrow table) to update the dataset or `None` if you are only interested in side effects."
                )
            elif isinstance(indices, list) and isinstance(processed_inputs, Mapping):
                allowed_batch_return_types = (list, np.ndarray, pd.Series)
                if config.POLARS_AVAILABLE and "polars" in sys.modules:
                    import polars as pl

                    allowed_batch_return_types += (pl.Series, pl.DataFrame)
                if config.TF_AVAILABLE and "tensorflow" in sys.modules:
                    import tensorflow as tf

                    allowed_batch_return_types += (tf.Tensor,)
                if config.TORCH_AVAILABLE and "torch" in sys.modules:
                    import torch

                    allowed_batch_return_types += (torch.Tensor,)
                if config.JAX_AVAILABLE and "jax" in sys.modules:
                    import jax.numpy as jnp

                    allowed_batch_return_types += (jnp.ndarray,)
                all_dict_values_are_lists = all(
                    isinstance(value, allowed_batch_return_types) for value in processed_inputs.values()
                )
                if all_dict_values_are_lists is False:
                    raise TypeError(
                        f"Provided `function` which is applied to all elements of table returns a `dict` of types {[type(x) for x in processed_inputs.values()]}. When using `batched=True`, make sure provided `function` returns a `dict` of types like `{allowed_batch_return_types}`."
                    )

        def apply_function_on_filtered_inputs(pa_inputs, indices, check_same_num_examples=False, offset=0):
            """Utility to apply the function on a selection of columns."""
            nonlocal update_data
            inputs = format_table(
                pa_inputs,
                0 if not batched else range(pa_inputs.num_rows),
                format_columns=input_columns,
                formatter=input_formatter,
            )
            fn_args = [inputs] if input_columns is None else [inputs[col] for col in input_columns]
            if offset == 0:
                effective_indices = indices
            else:
                effective_indices = [i + offset for i in indices] if isinstance(indices, list) else indices + offset
            additional_args = ()
            if with_indices:
                additional_args += (effective_indices,)
            if with_rank:
                additional_args += (rank,)
            processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
            if isinstance(processed_inputs, LazyDict):
                processed_inputs = {
                    k: v for k, v in processed_inputs.data.items() if k not in processed_inputs.keys_to_format
                }
                returned_lazy_dict = True
            else:
                returned_lazy_dict = False
            if update_data is None:
                # Check if the function returns updated examples
                update_data = isinstance(processed_inputs, (Mapping, pa.Table, pd.DataFrame))
                validate_function_output(processed_inputs, indices)
            if not update_data:
                return None  # Nothing to update, let's move on
            if shard._format_type or input_columns:
                # TODO(QL, MS): ideally the behavior should be the same even if the dataset is formatted (may require major release)
                inputs_to_merge = dict(zip(pa_inputs.column_names, pa_inputs.itercolumns()))
            elif isinstance(inputs, LazyDict):
                inputs_to_merge = {
                    k: (v if k not in inputs.keys_to_format else pa_inputs[k]) for k, v in inputs.data.items()
                }
            else:
                inputs_to_merge = inputs
            if remove_columns is not None:
                for column in remove_columns:
                    # `function` can modify input in-place causing column to be already removed.
                    if column in inputs_to_merge:
                        inputs_to_merge.pop(column)
                    if returned_lazy_dict and column in processed_inputs:
                        processed_inputs.pop(column)
            if check_same_num_examples:
                input_num_examples = len(pa_inputs)
                processed_inputs_num_examples = len(processed_inputs[next(iter(processed_inputs.keys()))])
                if input_num_examples != processed_inputs_num_examples:
                    raise NumExamplesMismatchError()
            if isinstance(inputs, Mapping) and isinstance(processed_inputs, Mapping):
                # The .map() transform *updates* the dataset:
                # the output dictionary contains both the the input data and the output data.
                # The output dictionary may contain Arrow values from `inputs_to_merge` so that we can re-write them efficiently.
                return {**inputs_to_merge, **processed_inputs}
            else:
                return processed_inputs

        def init_buffer_and_writer():
            # Prepare output buffer and batched writer in memory or on file if we update the table
            writer_features = features
            if writer_features is None:
                writer_features = shard.features
                update_features = True
            else:
                update_features = False
            if keep_in_memory or cache_file_name is None:
                buf_writer = pa.BufferOutputStream()
                tmp_file = None
                writer = ArrowWriter(
                    features=writer_features,
                    stream=buf_writer,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            else:
                buf_writer = None
                logger.info(f"Caching processed dataset at {cache_file_name}")
                tmp_file = tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(cache_file_name), delete=False)
                writer = ArrowWriter(
                    features=writer_features,
                    path=tmp_file.name,
                    writer_batch_size=writer_batch_size,
                    update_features=update_features,
                    fingerprint=new_fingerprint,
                    disable_nullable=disable_nullable,
                )
            return buf_writer, writer, tmp_file

        num_examples_progress_update = 0
        # If `update_data` is True after processing the first example/batch, initalize these resources with `init_buffer_and_writer`
        buf_writer, writer, tmp_file = None, None, None

        # Check if Polars is available and import it if so
        if config.POLARS_AVAILABLE and "polars" in sys.modules:
            import polars as pl

        # Optionally initialize the writer as a context manager
        with contextlib.ExitStack() as stack:
            try:
                arrow_formatted_shard = shard.with_format("arrow")

                # Loop over single examples or batches and write to buffer/file if examples are to be updated
                if not batched:
                    shard_iterable = enumerate(arrow_formatted_shard)
                else:
                    num_rows = len(shard) if not drop_last_batch else len(shard) // batch_size * batch_size
                    shard_iterable = zip(
                        range(0, num_rows, batch_size),
                        arrow_formatted_shard.iter(batch_size, drop_last_batch=drop_last_batch),
                    )
                if not batched:
                    _time = time.time()
                    for i, example in shard_iterable:
                        example = apply_function_on_filtered_inputs(example, i, offset=offset)
                        if update_data:
                            if i == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(example, pa.Table):
                                writer.write_row(example)
                            elif isinstance(example, pd.DataFrame):
                                writer.write_row(pa.Table.from_pandas(example))
                            elif (
                                config.POLARS_AVAILABLE
                                and "polars" in sys.modules
                                and isinstance(example, pl.DataFrame)
                            ):
                                writer.write_row(example.to_arrow())
                            else:
                                writer.write(example)
                        num_examples_progress_update += 1
                        if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                            _time = time.time()
                            yield rank, False, num_examples_progress_update
                            num_examples_progress_update = 0
                else:
                    _time = time.time()
                    for i, batch in shard_iterable:
                        num_examples_in_batch = len(batch)
                        indices = list(
                            range(*(slice(i, i + batch_size).indices(shard.num_rows)))
                        )  # Something simpler?
                        try:
                            batch = apply_function_on_filtered_inputs(
                                batch,
                                indices,
                                check_same_num_examples=len(shard.list_indexes()) > 0,
                                offset=offset,
                            )
                        except NumExamplesMismatchError:
                            raise DatasetTransformationNotAllowedError(
                                "Using `.map` in batched mode on a dataset with attached indexes is allowed only if it doesn't create or remove existing examples. You can first run `.drop_index() to remove your index and then re-add it."
                            ) from None
                        if update_data:
                            if i == 0:
                                buf_writer, writer, tmp_file = init_buffer_and_writer()
                                stack.enter_context(writer)
                            if isinstance(batch, pa.Table):
                                writer.write_table(batch)
                            elif isinstance(batch, pd.DataFrame):
                                writer.write_table(pa.Table.from_pandas(batch))
                            elif (
                                config.POLARS_AVAILABLE and "polars" in sys.modules and isinstance(batch, pl.DataFrame)
                            ):
                                writer.write_table(batch.to_arrow())
                            else:
                                writer.write_batch(batch)
                        num_examples_progress_update += num_examples_in_batch
                        if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                            _time = time.time()
                            yield rank, False, num_examples_progress_update
                            num_examples_progress_update = 0
                if update_data and writer is not None:
                    writer.finalize()  # close_stream=bool(buf_writer is None))  # We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                yield rank, False, num_examples_progress_update
                if update_data:
                    if writer is not None:
                        writer.finalize()
                    if tmp_file is not None:
                        tmp_file.close()
                        if os.path.exists(tmp_file.name):
                            os.remove(tmp_file.name)
                raise

        yield rank, False, num_examples_progress_update
        if update_data and tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(cache_file_name, 0o666 & ~umask)

        if update_data:
            # Create new Dataset from buffer or file
            info = shard.info.copy()
            info.features = writer._features
            info.task_templates = None
            if buf_writer is None:
                yield rank, True, Dataset.from_file(cache_file_name, info=info, split=shard.split)
            else:
                yield rank, True, Dataset.from_buffer(buf_writer.getvalue(), info=info, split=shard.split)
        else:
            yield rank, True, shard

    @transmit_format
    @fingerprint_transform(inplace=False)
    def batch(
        self,
        batch_size: int,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """
        Group samples from the dataset into batches.

        Args:
            batch_size (`int`):
                The number of samples in each batch.
            drop_last_batch (`bool`, defaults to `False`):
                Whether to drop the last incomplete batch.
            num_proc (`int`, *optional*, defaults to `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Returns:
            [`Dataset`]: A new Dataset where each item is a batch of multiple samples from the original dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> batched_ds = ds.batch(batch_size=4)
        >>> batched_ds[0]
        {'text': ['compassionately explores the seemingly irreconcilable situation...', ...],  # 4 items
        'label': [1, 1, 1, 1]}
        ```
        """

        def batch_fn(example):
            return {k: [v] for k, v in example.items()}

        return self.map(
            batch_fn,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
            new_fingerprint=new_fingerprint,
            desc="Batching examples",
        )

    @transmit_format
    @fingerprint_transform(
        inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name", "desc"], version="2.0.1"
    )
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        with_rank: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> "Dataset":
        """Apply a filter function to all the elements in the table in batches
        and update the table so that the dataset only includes examples according to the filter function.

        Args:
            function (`Callable`): Callable with one of the following signatures:

                - `function(example: Dict[str, Any]) -> bool` if `batched=False` and `with_indices=False` and `with_rank=False`
                - `function(example: Dict[str, Any], *extra_args) -> bool` if `batched=False` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)
                - `function(batch: Dict[str, List]) -> List[bool]` if `batched=True` and `with_indices=False` and `with_rank=False`
                - `function(batch: Dict[str, List], *extra_args) -> List[bool]` if `batched=True` and `with_indices=True` and/or `with_rank=True` (one extra arg for each)

                If no function is provided, defaults to an always `True` function: `lambda x: True`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the
                signature of `function` should be `def function(example, idx[, rank]): ...`.
            with_rank (`bool`, defaults to `False`):
                Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`str` or `List[str]`, *optional*):
                The columns to be passed into `function` as
                positional arguments. If `None`, a `dict` mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if
                `batched = True`. If `batched = False`, one example per batch is passed to `function`.
                If `batch_size <= 0` or `batch_size == None`, provide the full dataset as a single batch to `function`.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the current computation from `function`
                can be identified, use it instead of recomputing.
            cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            fn_kwargs (`dict`, *optional*):
                Keyword arguments to be passed to `function`.
            num_proc (`int`, *optional*):
                Number of processes for multiprocessing. By default it doesn't
                use multiprocessing.
            suffix_template (`str`):
                If `cache_file_name` is specified, then this suffix will be added at the end of the base name of each.
                For example, if `cache_file_name` is `"processed.arrow"`, then for `rank = 1` and `num_proc = 4`,
                the resulting file would be `"processed_00001_of_00004.arrow"` for the default suffix (default
                `_{rank:05d}_of_{num_proc:05d}`).
            new_fingerprint (`str`, *optional*):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.
            desc (`str`, *optional*, defaults to `None`):
                Meaningful description to be displayed alongside with the progress bar while filtering examples.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.filter(lambda x: x["label"] == 1)
        Dataset({
            features: ['text', 'label'],
            num_rows: 533
        })
        ```
        """
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.filter` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.`"
            )

        if function is None:
            function = lambda x: True  # noqa: E731

        if len(self) == 0:
            return self

        indices = self.map(
            function=partial(
                get_indices_from_mask_function,
                function,
                batched,
                with_indices,
                with_rank,
                input_columns,
                self._indices,
            ),
            with_indices=True,
            with_rank=True,
            features=Features({"indices": Value("uint64")}),
            batched=True,
            batch_size=batch_size,
            remove_columns=self.column_names,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint,
            input_columns=input_columns,
            desc=desc or "Filter",
        )
        new_dataset = copy.deepcopy(self)
        new_dataset._indices = indices.data
        new_dataset._fingerprint = new_fingerprint
        return new_dataset

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["cache_file_name"])
    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        num_proc: Optional[int] = None,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create and cache a new Dataset by flattening the indices mapping.

        Args:
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            cache_file_name (`str`, *optional*, default `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Allow null values in the table.
            num_proc (`int`, optional, default `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
        """

        return self.map(
            batched=True,  # for speed
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            new_fingerprint=new_fingerprint,
            desc="Flattening the indices",
            num_proc=num_proc,
        )

    def _new_dataset_with_indices(
        self,
        indices_cache_file_name: Optional[str] = None,
        indices_buffer: Optional[pa.Buffer] = None,
        fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Return a new Dataset obtained by adding indices (provided in indices_cache_file_name or in a buffer) to the
        current Dataset.
        """

        if indices_cache_file_name is None and indices_buffer is None:
            raise ValueError("At least one of indices_cache_file_name or indices_buffer must be provided.")

        if fingerprint is None:
            raise ValueError("please specify a fingerprint for the dataset with indices")

        if indices_cache_file_name is not None:
            indices_table = MemoryMappedTable.from_file(indices_cache_file_name)
        else:
            indices_table = InMemoryTable.from_buffer(indices_buffer)

        # Return new Dataset object
        # don't forget to copy the objects
        return Dataset(
            self._data,
            info=self.info.copy(),
            split=self.split,
            indices_table=indices_table,
            fingerprint=fingerprint,
        )

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create a new dataset with rows selected following the list/array of indices.

        Args:
            indices (`range`, `list`, `iterable`, `ndarray` or `Series`):
                Range, list or 1D-array of integer indices for indexing.
                If the indices correspond to a contiguous range, the Arrow table is simply sliced.
                However passing a list of indices that are not contiguous creates indices mapping, which is much less efficient,
                but still faster than recreating an Arrow table made of the requested rows.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the indices mapping in memory instead of writing it to a cache file.
            indices_cache_file_name (`str`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.select(range(4))
        Dataset({
            features: ['text', 'label'],
            num_rows: 4
        })
        ```
        """
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )

        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        # If indices is a PyArrow array, we convert to NumPy
        if isinstance(indices, (pa.Array, pa.ChunkedArray)):
            indices = indices.to_numpy().astype(np.int64)

        # Convert generator objects to lists
        if isinstance(indices, Iterator):
            indices = list(indices)

        # If the indices are contiguous, simply slice the arrow table
        if isinstance(indices, range):
            if _is_range_contiguous(indices) and indices.start >= 0:
                start, length = indices.start, indices.stop - indices.start
                return self._select_contiguous(start, length, new_fingerprint=new_fingerprint)
        else:
            try:
                start = next(iter(indices))
            except StopIteration:
                # if `indices` is an empty iterable, we return an empty dataset
                return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)
            if start >= 0:
                counter_from_start = itertools.count(start=start)
                if all(i == j for i, j in zip(indices, counter_from_start)):
                    length = next(counter_from_start) - start
                    return self._select_contiguous(start, length, new_fingerprint=new_fingerprint)

        # If not contiguous, we need to create a new indices mapping
        return self._select_with_indices_mapping(
            indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(inplace=False)
    def _select_contiguous(
        self,
        start: int,
        length: int,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create a new dataset with rows from a contiguous slice of data.
        The slice is defined by that start index and its length.

        Args:
            start (`int`): start index.
            length (`int`): length of the slice to select.
            new_fingerprint (`str`, optional, default `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds._select_contiguous(0, 4)
        Dataset({
            features: ['text', 'label'],
            num_rows: 4
        })
        ```
        """
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )

        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        _check_valid_indices_value(start, len(self))
        _check_valid_indices_value(start + length - 1, len(self))
        if self._indices is None or length == 0:
            return Dataset(
                self.data.slice(start, length),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint,
            )
        else:
            return Dataset(
                self.data,
                info=self.info.copy(),
                split=self.split,
                indices_table=self._indices.slice(start, length),
                fingerprint=new_fingerprint,
            )

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def _select_with_indices_mapping(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create a new dataset with rows selected following the list/array of indices.
        The new dataset is made by creating a new indices mapping on top of the main arrow table.

        Args:
            indices (sequence, iterable, range, ndarray or Series): List or 1D-array of integer indices for indexing.
            keep_in_memory (`bool`, default `False`): Keep the indices mapping in memory instead of writing it to a cache file.
            indices_cache_file_name (`str`, optional, default `None`): Provide the name of a path for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
            writer_batch_size (`int`, default `1000`): Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
            new_fingerprint (`str`, optional, default `None`): the new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds._select_with_indices_mapping(range(4))
        Dataset({
            features: ['text', 'label'],
            num_rows: 4
        })
        ```
        """
        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.select` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )

        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        # Prepare the writer for our indices arrow table
        if keep_in_memory or indices_cache_file_name is None:
            buf_writer = pa.BufferOutputStream()
            tmp_file = None
            writer = ArrowWriter(
                stream=buf_writer, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices"
            )
        else:
            buf_writer = None
            logger.info(f"Caching indices mapping at {indices_cache_file_name}")
            tmp_file = tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(indices_cache_file_name), delete=False)
            writer = ArrowWriter(
                path=tmp_file.name, writer_batch_size=writer_batch_size, fingerprint=new_fingerprint, unit="indices"
            )

        indices = indices if isinstance(indices, list) else list(indices)

        size = len(self)
        if indices:
            _check_valid_indices_value(int(max(indices)), size=size)
            _check_valid_indices_value(int(min(indices)), size=size)
        else:
            return self._select_contiguous(0, 0, new_fingerprint=new_fingerprint)

        indices_array = pa.array(indices, type=pa.uint64())
        # Check if we need to convert indices
        if self._indices is not None:
            indices_array = self._indices.column(0).take(indices_array)

        indices_table = pa.Table.from_arrays([indices_array], names=["indices"])

        with writer:
            try:
                writer.write_table(indices_table)
                writer.finalize()  # close_stream=bool(buf_writer is None))  We only close if we are writing in a file
            except (Exception, KeyboardInterrupt):
                if tmp_file is not None:
                    tmp_file.close()
                    if os.path.exists(tmp_file.name):
                        os.remove(tmp_file.name)
                raise

        if tmp_file is not None:
            tmp_file.close()
            shutil.move(tmp_file.name, indices_cache_file_name)
            umask = os.umask(0o666)
            os.umask(umask)
            os.chmod(indices_cache_file_name, 0o666 & ~umask)

        # Return new Dataset object
        if buf_writer is None:
            return self._new_dataset_with_indices(
                indices_cache_file_name=indices_cache_file_name, fingerprint=new_fingerprint
            )
        else:
            return self._new_dataset_with_indices(indices_buffer=buf_writer.getvalue(), fingerprint=new_fingerprint)

    def skip(self, n: int) -> "Dataset":
        """
        Create a new [`Dataset`] that skips the first `n` elements.

        Args:
            n (`int`):
                Number of elements to skip.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'}]
        >>> ds = ds.skip(1)
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'},
         {'label': 1,
         'text': 'if you sometimes like to go to the movies to have fun , wasabi is a good place to start .'}]
        ```
        """
        return self.select(range(n, len(self)))

    def take(self, n: int) -> "Dataset":
        """
        Create a new [`Dataset`] with only the first `n` elements.

        Args:
            n (`int`):
                Number of elements to take.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train")
        >>> small_ds = ds.take(2)
        >>> list(small_ds)
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .'}]
        ```
        """
        return self.select(range(n))

    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"])
    def sort(
        self,
        column_names: Union[str, Sequence_[str]],
        reverse: Union[bool, Sequence_[bool]] = False,
        kind="deprecated",
        null_placement: str = "at_end",
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create a new dataset sorted according to a single or multiple columns.

        Args:
            column_names (`Union[str, Sequence[str]]`):
                Column name(s) to sort by.
            reverse (`Union[bool, Sequence[bool]]`, defaults to `False`):
                If `True`, sort by descending order rather than ascending. If a single bool is provided,
                the value is applied to the sorting of all column names. Otherwise a list of bools with the
                same length and order as column_names must be provided.
            kind (`str`, *optional*):
                Pandas algorithm for sorting selected in `{quicksort, mergesort, heapsort, stable}`,
                The default is `quicksort`. Note that both `stable` and `mergesort` use `timsort` under the covers and, in general,
                the actual implementation will vary with data type. The `mergesort` option is retained for backwards compatibility.
                <Deprecated version="2.8.0">

                `kind` was deprecated in version 2.10.0 and will be removed in 3.0.0.

                </Deprecated>
            null_placement (`str`, defaults to `at_end`):
                Put `None` values at the beginning if `at_start` or `first` or at the end if `at_end` or `last`

                <Added version="1.14.2"/>
            keep_in_memory (`bool`, defaults to `False`):
                Keep the sorted indices in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the sorted indices
                can be identified, use it instead of recomputing.
            indices_cache_file_name (`str`, *optional*, defaults to `None`):
                Provide the name of a path for the cache file. It is used to store the
                sorted indices instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                Higher value gives smaller cache files, lower value consume less temporary memory.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset('rotten_tomatoes', split='validation')
        >>> ds['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> sorted_ds = ds.sort('label')
        >>> sorted_ds['label'][:10]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> another_sorted_ds = ds.sort(['label', 'text'], reverse=[True, False])
        >>> another_sorted_ds['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ```
        """
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.sort` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )
        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        # Deprecation warning
        if kind != "deprecated":
            warnings.warn(
                "'kind' was deprecated in version 2.10.0 and will be removed in 3.0.0.",
                category=FutureWarning,
            )

        # Check proper format of and for duplicates in column_names
        if isinstance(column_names, str):
            column_names = [column_names]

        # Check proper format and length of reverse
        if not isinstance(reverse, bool):
            if len(reverse) != len(column_names):
                raise ValueError(
                    "Parameter 'reverse' should be either a boolean or a list of booleans with the same length as 'column_names'."
                )
        else:
            reverse = [reverse] * len(column_names)

        # Check whether column name(s) exist in dataset
        for column in column_names:
            if not isinstance(column, str) or column not in self._data.column_names:
                raise ValueError(
                    f"Column '{column}' not found in the dataset. Please provide a column selected in: {self._data.column_names}"
                )

        # Change null_placement to conform to pyarrow's sort_indices() while ensuring backwards compatability
        if null_placement not in ["at_start", "at_end"]:
            if null_placement == "first":
                null_placement = "at_start"
            elif null_placement == "last":
                null_placement = "at_end"
            else:
                raise ValueError(
                    f"null_placement '{null_placement}' is an invalid parameter value. Must be either 'last', 'at_end', 'first' or 'at_start'."
                )

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        # Check if we've already cached this computation (indexed by a hash)
        if self.cache_files:
            if indices_cache_file_name is None:
                # we create a unique hash from the function, current dataset file and the mapping args
                indices_cache_file_name = self._get_cache_file_path(new_fingerprint)
            if os.path.exists(indices_cache_file_name) and load_from_cache_file:
                logger.info(f"Loading cached sorted indices for dataset at {indices_cache_file_name}")
                return self._new_dataset_with_indices(
                    fingerprint=new_fingerprint, indices_cache_file_name=indices_cache_file_name
                )

        sort_table = query_table(
            table=self._data,
            key=slice(0, len(self)),
            indices=self._indices,
        )

        sort_keys = [
            (col, "ascending" if not col_reverse else "descending") for col, col_reverse in zip(column_names, reverse)
        ]

        indices = pc.sort_indices(sort_table, sort_keys=sort_keys, null_placement=null_placement)

        return self.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(
        inplace=False, randomized_function=True, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"]
    )
    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Dataset":
        """Create a new Dataset where the rows are shuffled.

        Currently shuffling uses numpy random generators.
        You can either supply a NumPy BitGenerator to use, or a seed to initiate NumPy's default random generator (PCG64).

        Shuffling takes the list of indices `[0:len(my_dataset)]` and shuffles it to create an indices mapping.
        However as soon as your [`Dataset`] has an indices mapping, the speed can become 10x slower.
        This is because there is an extra step to get the row index to read using the indices mapping, and most importantly, you aren't reading contiguous chunks of data anymore.
        To restore the speed, you'd need to rewrite the entire dataset on your disk again using [`Dataset.flatten_indices`], which removes the indices mapping.
        This may take a lot of time depending of the size of your dataset though:

        ```python
        my_dataset[0]  # fast
        my_dataset = my_dataset.shuffle(seed=42)
        my_dataset[0]  # up to 10x slower
        my_dataset = my_dataset.flatten_indices()  # rewrite the shuffled dataset on disk as contiguous chunks of data
        my_dataset[0]  # fast again
        ```

        In this case, we recommend switching to an [`IterableDataset`] and leveraging its fast approximate shuffling method [`IterableDataset.shuffle`].
        It only shuffles the shards order and adds a shuffle buffer to your dataset, which keeps the speed of your dataset optimal:

        ```python
        my_iterable_dataset = my_dataset.to_iterable_dataset(num_shards=128)
        for example in enumerate(my_iterable_dataset):  # fast
            pass

        shuffled_iterable_dataset = my_iterable_dataset.shuffle(seed=42, buffer_size=100)

        for example in enumerate(shuffled_iterable_dataset):  # as fast as before
            pass
        ```

        Args:
            seed (`int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`.
                If `None`, then fresh, unpredictable entropy will be pulled from the OS.
                If an `int` or `array_like[ints]` is passed, then it will be passed to SeedSequence to derive the initial BitGenerator state.
            generator (`numpy.random.Generator`, *optional*):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
            keep_in_memory (`bool`, default `False`):
                Keep the shuffled indices in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the shuffled indices
                can be identified, use it instead of recomputing.
            indices_cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                shuffled indices instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds['label'][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # set a seed
        >>> shuffled_ds = ds.shuffle(seed=42)
        >>> shuffled_ds['label'][:10]
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        ```
        """
        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.shuffle` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )
        # If the array is empty we do nothing
        if len(self) == 0:
            return self

        if keep_in_memory and indices_cache_file_name is not None:
            raise ValueError("Please use either `keep_in_memory` or `indices_cache_file_name` but not both.")

        if seed is not None and generator is not None:
            raise ValueError("Both `seed` and `generator` were provided. Please specify just one of them.")

        if generator is not None and not isinstance(generator, np.random.Generator):
            raise ValueError("The provided generator must be an instance of numpy.random.Generator")

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if generator is None:
            if seed is None:
                _, seed, pos, *_ = np.random.get_state()
                seed = seed[pos] if pos < 624 else seed[0]
                _ = np.random.random()  # do 1 step of rng
            generator = np.random.default_rng(seed)

        # Check if we've already cached this computation (indexed by a hash)
        if self.cache_files:
            if indices_cache_file_name is None:
                # we create a unique hash from the function, current dataset file and the mapping args
                indices_cache_file_name = self._get_cache_file_path(new_fingerprint)
            if os.path.exists(indices_cache_file_name) and load_from_cache_file:
                logger.info(f"Loading cached shuffled indices for dataset at {indices_cache_file_name}")
                return self._new_dataset_with_indices(
                    fingerprint=new_fingerprint, indices_cache_file_name=indices_cache_file_name
                )

        permutation = generator.permutation(len(self))

        return self.select(
            indices=permutation,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name if not keep_in_memory else None,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )

    @transmit_format
    @fingerprint_transform(
        inplace=False,
        randomized_function=True,
        fingerprint_names=["train_new_fingerprint", "test_new_fingerprint"],
        ignore_kwargs=["load_from_cache_file", "train_indices_cache_file_name", "test_indices_cache_file_name"],
    )
    def train_test_split(
        self,
        test_size: Union[float, int, None] = None,
        train_size: Union[float, int, None] = None,
        shuffle: bool = True,
        stratify_by_column: Optional[str] = None,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        train_indices_cache_file_name: Optional[str] = None,
        test_indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        train_new_fingerprint: Optional[str] = None,
        test_new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        """Return a dictionary ([`datasets.DatasetDict`]) with two random train and test subsets (`train` and `test` `Dataset` splits).
        Splits are created from the dataset according to `test_size`, `train_size` and `shuffle`.

        This method is similar to scikit-learn `train_test_split`.

        Args:
            test_size (`numpy.random.Generator`, *optional*):
                Size of the test split
                If `float`, should be between `0.0` and `1.0` and represent the proportion of the dataset to include in the test split.
                If `int`, represents the absolute number of test samples.
                If `None`, the value is set to the complement of the train size.
                If `train_size` is also `None`, it will be set to `0.25`.
            train_size (`numpy.random.Generator`, *optional*):
                Size of the train split
                If `float`, should be between `0.0` and `1.0` and represent the proportion of the dataset to include in the train split.
                If `int`, represents the absolute number of train samples.
                If `None`, the value is automatically set to the complement of the test size.
            shuffle (`bool`, *optional*, defaults to `True`):
                Whether or not to shuffle the data before splitting.
            stratify_by_column (`str`, *optional*, defaults to `None`):
                The column name of labels to be used to perform stratified split of data.
            seed (`int`, *optional*):
                A seed to initialize the default BitGenerator if `generator=None`.
                If `None`, then fresh, unpredictable entropy will be pulled from the OS.
                If an `int` or `array_like[ints]` is passed, then it will be passed to SeedSequence to derive the initial BitGenerator state.
            generator (`numpy.random.Generator`, *optional*):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
            keep_in_memory (`bool`, defaults to `False`):
                Keep the splits indices in memory instead of writing it to a cache file.
            load_from_cache_file (`Optional[bool]`, defaults to `True` if caching is enabled):
                If a cache file storing the splits indices
                can be identified, use it instead of recomputing.
            train_cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                train split indices instead of the automatically generated cache file name.
            test_cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                test split indices instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            train_new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the train set after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
            test_new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the test set after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds = ds.train_test_split(test_size=0.2, shuffle=True)
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 852
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 214
            })
        })

        # set a seed
        >>> ds = ds.train_test_split(test_size=0.2, seed=42)

        # stratified split
        >>> ds = load_dataset("imdb",split="train")
        Dataset({
            features: ['text', 'label'],
            num_rows: 25000
        })
        >>> ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 20000
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 5000
            })
        })
        ```
        """
        from .dataset_dict import DatasetDict  # import here because of circular dependency

        if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.train_test_split` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it."
            )
        # If the array is empty we do nothing
        if len(self) == 0:
            return DatasetDict({"train": self, "test": self})

        if test_size is None and train_size is None:
            test_size = 0.25

        # Safety checks similar to scikit-learn's ones.
        # (adapted from https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/model_selection/_split.py#L1750)
        n_samples = len(self)
        if (
            isinstance(test_size, int)
            and (test_size >= n_samples or test_size <= 0)
            or isinstance(test_size, float)
            and (test_size <= 0 or test_size >= 1)
        ):
            raise ValueError(
                f"test_size={test_size} should be either positive and smaller "
                f"than the number of samples {n_samples} or a float in the (0, 1) range"
            )

        if (
            isinstance(train_size, int)
            and (train_size >= n_samples or train_size <= 0)
            or isinstance(train_size, float)
            and (train_size <= 0 or train_size >= 1)
        ):
            raise ValueError(
                f"train_size={train_size} should be either positive and smaller "
                f"than the number of samples {n_samples} or a float in the (0, 1) range"
            )

        if train_size is not None and not isinstance(train_size, (int, float)):
            raise ValueError(f"Invalid value for train_size: {train_size} of type {type(train_size)}")
        if test_size is not None and not isinstance(test_size, (int, float)):
            raise ValueError(f"Invalid value for test_size: {test_size} of type {type(test_size)}")

        if isinstance(train_size, float) and isinstance(test_size, float) and train_size + test_size > 1:
            raise ValueError(
                f"The sum of test_size and train_size = {train_size + test_size}, should be in the (0, 1)"
                " range. Reduce test_size and/or train_size."
            )

        if isinstance(test_size, float):
            n_test = ceil(test_size * n_samples)
        elif isinstance(test_size, int):
            n_test = float(test_size)

        if isinstance(train_size, float):
            n_train = floor(train_size * n_samples)
        elif isinstance(train_size, int):
            n_train = float(train_size)

        if train_size is None:
            n_train = n_samples - n_test
        elif test_size is None:
            n_test = n_samples - n_train

        if n_train + n_test > n_samples:
            raise ValueError(
                f"The sum of train_size and test_size = {n_train + n_test}, "
                "should be smaller than the number of "
                f"samples {n_samples}. Reduce test_size and/or "
                "train_size."
            )

        n_train, n_test = int(n_train), int(n_test)

        if n_train == 0:
            raise ValueError(
                f"With n_samples={n_samples}, test_size={test_size} and train_size={train_size}, the "
                "resulting train set will be empty. Adjust any of the "
                "aforementioned parameters."
            )

        load_from_cache_file = load_from_cache_file if load_from_cache_file is not None else is_caching_enabled()

        if generator is None and shuffle is True:
            if seed is None:
                _, seed, pos, *_ = np.random.get_state()
                seed = seed[pos] if pos < 624 else seed[0]
                _ = np.random.random()  # do 1 step of rng
            generator = np.random.default_rng(seed)

        # Check if we've already cached this computation (indexed by a hash)
        if self.cache_files:
            if train_indices_cache_file_name is None or test_indices_cache_file_name is None:
                # we create a unique hash from the function, current dataset file and the mapping args

                if train_indices_cache_file_name is None:
                    train_indices_cache_file_name = self._get_cache_file_path(train_new_fingerprint)
                if test_indices_cache_file_name is None:
                    test_indices_cache_file_name = self._get_cache_file_path(test_new_fingerprint)
            if (
                os.path.exists(train_indices_cache_file_name)
                and os.path.exists(test_indices_cache_file_name)
                and load_from_cache_file
            ):
                logger.info(
                    f"Loading cached split indices for dataset at {train_indices_cache_file_name} and {test_indices_cache_file_name}"
                )
                return DatasetDict(
                    {
                        "train": self._new_dataset_with_indices(
                            fingerprint=train_new_fingerprint, indices_cache_file_name=train_indices_cache_file_name
                        ),
                        "test": self._new_dataset_with_indices(
                            fingerprint=test_new_fingerprint, indices_cache_file_name=test_indices_cache_file_name
                        ),
                    }
                )
        if not shuffle:
            if stratify_by_column is not None:
                raise ValueError("Stratified train/test split is not implemented for `shuffle=False`")
            train_indices = np.arange(n_train)
            test_indices = np.arange(n_train, n_train + n_test)
        else:
            # stratified partition
            if stratify_by_column is not None:
                if stratify_by_column not in self._info.features.keys():
                    raise ValueError(f"Key {stratify_by_column} not found in {self._info.features.keys()}")
                if not isinstance(self._info.features[stratify_by_column], ClassLabel):
                    raise ValueError(
                        f"Stratifying by column is only supported for {ClassLabel.__name__} column, and column {stratify_by_column} is {type(self._info.features[stratify_by_column]).__name__}."
                    )
                try:
                    train_indices, test_indices = next(
                        stratified_shuffle_split_generate_indices(
                            self.with_format("numpy")[stratify_by_column], n_train, n_test, rng=generator
                        )
                    )
                except Exception as error:
                    if str(error) == "Minimum class count error":
                        raise ValueError(
                            f"The least populated class in {stratify_by_column} column has only 1"
                            " member, which is too few. The minimum"
                            " number of groups for any class cannot"
                            " be less than 2."
                        )
                    else:
                        raise error

            # random partition
            else:
                permutation = generator.permutation(len(self))
                test_indices = permutation[:n_test]
                train_indices = permutation[n_test : (n_test + n_train)]

        train_split = self.select(
            indices=train_indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=train_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=train_new_fingerprint,
        )
        test_split = self.select(
            indices=test_indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=test_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=test_new_fingerprint,
        )

        return DatasetDict({"train": train_split, "test": test_split})

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "Dataset":
        """Return the `index`-nth shard from dataset split into `num_shards` pieces.

        This shards deterministically. `dset.shard(n, i)` will contain all elements of dset whose
        index mod `n = i`.

        `dset.shard(n, i, contiguous=True)` will instead split dset into contiguous chunks,
        so it can be easily concatenated back together after processing. If `n % i == l`, then the
        first `l` shards will have length `(n // i) + 1`, and the remaining shards will have length `(n // i)`.
        `datasets.concatenate([dset.shard(n, i, contiguous=True) for i in range(n)])` will return
        a dataset with the same order as the original.

        Be sure to shard before using any randomizing operator (such as `shuffle`).
        It is best if the shard operator is used early in the dataset pipeline.


        Args:
            num_shards (`int`):
                How many shards to split the dataset into.
            index (`int`):
                Which shard to select and return.
            contiguous: (`bool`, defaults to `False`):
                Whether to select contiguous blocks of indices for shards.
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            indices_cache_file_name (`str`, *optional*):
                Provide the name of a path for the cache file. It is used to store the
                indices of each shard instead of the automatically generated cache file name.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds
        Dataset({
            features: ['text', 'label'],
            num_rows: 1066
        })
        >>> ds.shard(num_shards=2, index=0)
        Dataset({
            features: ['text', 'label'],
            num_rows: 533
        })
        ```
        """
        if not 0 <= index < num_shards:
            raise ValueError("index should be in [0, num_shards-1]")
        if contiguous:
            div = len(self) // num_shards
            mod = len(self) % num_shards
            start = div * index + min(index, mod)
            end = start + div + (1 if index < mod else 0)
            indices = range(start, end)
        else:
            indices = np.arange(index, len(self), num_shards)

        return self.select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
        )

    @deprecated()
    def export(
        self,
        filename: str,
        format: str = "tfrecord",
    ):
        """Writes the Arrow dataset to a TFRecord file.

        The dataset must already be in tensorflow format. The records will be written with
        keys from `dataset._format_columns`.

        Args:
            filename (`str`): The filename, including the `.tfrecord` extension, to write to.
            format (`str`, optional, default `"tfrecord"`): The type of output file. Currently this is a no-op, as
                TFRecords are the only option. This enables a more flexible function signature later.
        """
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            logger.error("Tensorflow needs to be installed to be able to return Tensorflow tensors.")

        # From https://www.tensorflow.org/tutorials/load_data/tfrecord
        def _bytes_feature(values):
            """Returns a bytes_list from a list of string / byte."""
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

        def _float_feature(values):
            """Returns a float_list from a list of float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=values))

        def _int64_feature(values):
            """Returns an int64_list from a list of bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        def _feature(values: Union[float, int, str, np.ndarray, list]) -> "tf.train.Feature":
            """Typechecks `values` and returns the corresponding tf.train.Feature."""
            if isinstance(values, list):
                if values and isinstance(values[0], str):
                    return _bytes_feature([v.encode() for v in values])
                else:
                    raise ValueError(f"values={values} is empty or contains items that cannot be serialized")
            elif isinstance(values, np.ndarray):
                if values.dtype == np.dtype(float):
                    return _float_feature(values)
                elif values.dtype == np.int64:
                    return _int64_feature(values)
                elif values.dtype == np.dtype(str) or (
                    values.dtype == np.dtype(object) and len(values) > 0 and isinstance(values[0], str)
                ):
                    return _bytes_feature([v.encode() for v in values])
                else:
                    raise ValueError(
                        f"values={values} is empty or is an np.ndarray with items of dtype {values[0].dtype}, which cannot be serialized"
                    )
            elif hasattr(values, "dtype"):
                if np.issubdtype(values.dtype, np.floating):
                    return _float_feature([values.item()])
                elif np.issubdtype(values.dtype, np.integer):
                    return _int64_feature([values.item()])
                elif np.issubdtype(values.dtype, str):
                    return _bytes_feature([values.item().encode()])
                else:
                    raise ValueError(f"values={values} has dtype {values.dtype}, which cannot be serialized")
            else:
                raise ValueError(f"values={values} are not numpy objects or strings, and so cannot be serialized")

        def serialize_example(ex):
            feature = {key: _feature(value) for key, value in ex.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            return example_proto.SerializeToString()

        def tf_serialize_example(ex):
            tf_string = tf.py_function(serialize_example, (ex,), tf.string)
            return tf.reshape(tf_string, ())

        def generator():
            for ex in self:
                yield serialize_example(ex)

        if self._format_type != "numpy":
            raise ValueError("Dataset format must be numpy before exporting")
        if not filename.endswith(".tfrecord"):
            raise ValueError("filename {filename} must end with .tfrecord")
        tf_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
        writer = tf.data.experimental.TFRecordWriter(filename)
        logger.info(f"Writing TFRecord to {filename}")
        writer.write(tf_dataset)
        logger.info(f"Finished writing TFRecord to {filename}")
        self = None  # delete the dataset reference used by tf_dataset

    def to_csv(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_csv_kwargs,
    ) -> int:
        """Exports the dataset to csv

        Args:
            path_or_buf (`PathLike` or `FileOrBuffer`):
                Either a path to a file (e.g. `file.csv`), a remote URI (e.g. `hf://datasets/username/my_dataset_name/data.csv`),
                or a BinaryIO, where the dataset will be saved to in the specified format.
            batch_size (`int`, *optional*):
                Size of the batch to load in memory and write at once.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            num_proc (`int`, *optional*):
                Number of processes for multiprocessing. By default it doesn't
                use multiprocessing. `batch_size` in this case defaults to
                `datasets.config.DEFAULT_MAX_BATCH_SIZE` but feel free to make it 5x or 10x of the default
                value if you have sufficient compute power.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.19.0"/>
            **to_csv_kwargs (additional keyword arguments):
                Parameters to pass to pandas's [`pandas.DataFrame.to_csv`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html).

                <Changed version="2.10.0">

                Now, `index` defaults to `False` if not specified.

                If you would like to write the index, pass `index=True` and also set a name for the index column by
                passing `index_label`.

                </Changed>

        Returns:
            `int`: The number of characters or bytes written.

        Example:

        ```py
        >>> ds.to_csv("path/to/dataset/directory")
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.csv import CsvDatasetWriter

        return CsvDatasetWriter(
            self,
            path_or_buf,
            batch_size=batch_size,
            num_proc=num_proc,
            storage_options=storage_options,
            **to_csv_kwargs,
        ).write()

    def to_dict(self, batch_size: Optional[int] = None, batched="deprecated") -> Union[dict, Iterator[dict]]:
        """Returns the dataset as a Python dict. Can also return a generator for large datasets.

        Args:
            batched (`bool`):
                Set to `True` to return a generator that yields the dataset as batches
                of `batch_size` rows. Defaults to `False` (returns the whole datasets once).

                <Deprecated version="2.11.0">

                Use `.iter(batch_size=batch_size)` followed by `.to_dict()` on the individual batches instead.

                </Deprecated>

            batch_size (`int`, *optional*): The size (number of rows) of the batches if `batched` is `True`.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.

        Returns:
            `dict` or `Iterator[dict]`

        Example:

        ```py
        >>> ds.to_dict()
        ```
        """
        if batched != "deprecated":
            warnings.warn(
                "'batched' was deprecated in version 2.11.0 and will be removed in version 3.0.0. Use `.iter(batch_size=batch_size)` followed by `.to_dict()` on the individual batches instead.",
                FutureWarning,
            )
        else:
            batched = False

        if not batched:
            return query_table(
                table=self._data,
                key=slice(0, len(self)),
                indices=self._indices,
            ).to_pydict()
        else:
            batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
            return (
                query_table(
                    table=self._data,
                    key=slice(offset, offset + batch_size),
                    indices=self._indices,
                ).to_pydict()
                for offset in range(0, len(self), batch_size)
            )

    def to_list(self) -> list:
        """Returns the dataset as a Python list.

        Returns:
            `list`

        Example:

        ```py
        >>> ds.to_list()
        ```
        """
        return query_table(
            table=self._data,
            key=slice(0, len(self)),
            indices=self._indices,
        ).to_pylist()

    def to_json(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ) -> int:
        """Export the dataset to JSON Lines or JSON.

        The default output format is [JSON Lines](https://jsonlines.org/).
        To export to [JSON](https://www.json.org), pass `lines=False` argument and the desired `orient`.

        Args:
            path_or_buf (`PathLike` or `FileOrBuffer`):
                Either a path to a file (e.g. `file.json`), a remote URI (e.g. `hf://datasets/username/my_dataset_name/data.json`),
                or a BinaryIO, where the dataset will be saved to in the specified format.
            batch_size (`int`, *optional*):
                Size of the batch to load in memory and write at once.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            num_proc (`int`, *optional*):
                Number of processes for multiprocessing. By default, it doesn't
                use multiprocessing. `batch_size` in this case defaults to
                `datasets.config.DEFAULT_MAX_BATCH_SIZE` but feel free to make it 5x or 10x of the default
                value if you have sufficient compute power.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.19.0"/>
            **to_json_kwargs (additional keyword arguments):
                Parameters to pass to pandas's [`pandas.DataFrame.to_json`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html).
                Default arguments are `lines=True` and `orient="records".

                <Changed version="2.11.0">

                The parameter `index` defaults to `False` if `orient` is `"split"` or `"table"`.

                If you would like to write the index, pass `index=True`.

                </Changed>

        Returns:
            `int`: The number of characters or bytes written.

        Example:

        ```py
        >>> ds.to_json("path/to/dataset/directory/filename.jsonl")
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.json import JsonDatasetWriter

        return JsonDatasetWriter(
            self,
            path_or_buf,
            batch_size=batch_size,
            num_proc=num_proc,
            storage_options=storage_options,
            **to_json_kwargs,
        ).write()

    def to_pandas(
        self, batch_size: Optional[int] = None, batched: bool = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """Returns the dataset as a `pandas.DataFrame`. Can also return a generator for large datasets.

        Args:
            batched (`bool`):
                Set to `True` to return a generator that yields the dataset as batches
                of `batch_size` rows. Defaults to `False` (returns the whole datasets once).
            batch_size (`int`, *optional*):
                The size (number of rows) of the batches if `batched` is `True`.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.

        Returns:
            `pandas.DataFrame` or `Iterator[pandas.DataFrame]`

        Example:

        ```py
        >>> ds.to_pandas()
        ```
        """
        if not batched:
            return query_table(
                table=self._data,
                key=slice(0, len(self)),
                indices=self._indices,
            ).to_pandas(types_mapper=pandas_types_mapper)
        else:
            batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
            return (
                query_table(
                    table=self._data,
                    key=slice(offset, offset + batch_size),
                    indices=self._indices,
                ).to_pandas(types_mapper=pandas_types_mapper)
                for offset in range(0, len(self), batch_size)
            )

    def to_polars(
        self,
        batch_size: Optional[int] = None,
        batched: bool = False,
        schema_overrides: Optional[dict] = None,
        rechunk: bool = True,
    ) -> Union["pl.DataFrame", Iterator["pl.DataFrame"]]:
        """Returns the dataset as a `polars.DataFrame`. Can also return a generator for large datasets.

        Args:
            batched (`bool`):
                Set to `True` to return a generator that yields the dataset as batches
                of `batch_size` rows. Defaults to `False` (returns the whole datasets once).
            batch_size (`int`, *optional*):
                The size (number of rows) of the batches if `batched` is `True`.
                Defaults to `genomicsml.datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            schema_overrides (`dict`, *optional*):
                Support type specification or override of one or more columns; note that
                any dtypes inferred from the schema param will be overridden.
            rechunk (`bool`):
                Make sure that all data is in contiguous memory. Defaults to `True`.
        Returns:
            `polars.DataFrame` or `Iterator[polars.DataFrame]`

        Example:

        ```py
        >>> ds.to_polars()
        ```
        """
        if config.POLARS_AVAILABLE:
            import polars as pl

            if not batched:
                return pl.from_arrow(
                    query_table(
                        table=self._data,
                        key=slice(0, len(self)),
                        indices=self._indices if self._indices is not None else None,
                    ),
                    schema_overrides=schema_overrides,
                    rechunk=rechunk,
                )
            else:
                batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
                return (
                    pl.from_arrow(
                        query_table(
                            table=self._data,
                            key=slice(offset, offset + batch_size),
                            indices=self._indices if self._indices is not None else None,
                        ),
                        schema_overrides=schema_overrides,
                        rechunk=rechunk,
                    )
                    for offset in range(0, len(self), batch_size)
                )
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")

    def to_parquet(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **parquet_writer_kwargs,
    ) -> int:
        """Exports the dataset to parquet

        Args:
            path_or_buf (`PathLike` or `FileOrBuffer`):
                Either a path to a file (e.g. `file.parquet`), a remote URI (e.g. `hf://datasets/username/my_dataset_name/data.parquet`),
                or a BinaryIO, where the dataset will be saved to in the specified format.
            batch_size (`int`, *optional*):
                Size of the batch to load in memory and write at once.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.19.0"/>
            **parquet_writer_kwargs (additional keyword arguments):
                Parameters to pass to PyArrow's `pyarrow.parquet.ParquetWriter`.

        Returns:
            `int`: The number of characters or bytes written.

        Example:

        ```py
        >>> ds.to_parquet("path/to/dataset/directory")
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.parquet import ParquetDatasetWriter

        return ParquetDatasetWriter(
            self, path_or_buf, batch_size=batch_size, storage_options=storage_options, **parquet_writer_kwargs
        ).write()

    def to_sql(
        self,
        name: str,
        con: Union[str, "sqlalchemy.engine.Connection", "sqlalchemy.engine.Engine", "sqlite3.Connection"],
        batch_size: Optional[int] = None,
        **sql_writer_kwargs,
    ) -> int:
        """Exports the dataset to a SQL database.

        Args:
            name (`str`):
                Name of SQL table.
            con (`str` or `sqlite3.Connection` or `sqlalchemy.engine.Connection` or `sqlalchemy.engine.Connection`):
                A [URI string](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls) or a SQLite3/SQLAlchemy connection object used to write to a database.
            batch_size (`int`, *optional*):
                Size of the batch to load in memory and write at once.
                Defaults to `datasets.config.DEFAULT_MAX_BATCH_SIZE`.
            **sql_writer_kwargs (additional keyword arguments):
                Parameters to pass to pandas's [`pandas.DataFrame.to_sql`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html).

                <Changed version="2.11.0">

                Now, `index` defaults to `False` if not specified.

                If you would like to write the index, pass `index=True` and also set a name for the index column by
                passing `index_label`.

                </Changed>

        Returns:
            `int`: The number of records written.

        Example:

        ```py
        >>> # con provided as a connection URI string
        >>> ds.to_sql("data", "sqlite:///my_own_db.sql")
        >>> # con provided as a sqlite3 connection object
        >>> import sqlite3
        >>> con = sqlite3.connect("my_own_db.sql")
        >>> with con:
        ...     ds.to_sql("data", con)
        ```
        """
        # Dynamic import to avoid circular dependency
        from .io.sql import SqlDatasetWriter

        return SqlDatasetWriter(self, name, con, batch_size=batch_size, **sql_writer_kwargs).write()

    def _estimate_nbytes(self) -> int:
        dataset_nbytes = self.data.nbytes

        # Find decodable columns, because if there are any, we need to
        # adjust the dataset size computation (needed for sharding) to account for possible external files
        decodable_columns = [
            k for k, v in self._info.features.items() if require_decoding(v, ignore_decode_attribute=True)
        ]

        if decodable_columns:
            # Approximate the space needed to store the bytes from the external files by analyzing the first 1000 examples
            extra_nbytes = 0

            def extra_nbytes_visitor(array, feature):
                nonlocal extra_nbytes
                if isinstance(feature, (Audio, Image)):
                    for x in array.to_pylist():
                        if x is not None and x["bytes"] is None and x["path"] is not None:
                            size = xgetsize(x["path"])
                            extra_nbytes += size
                    extra_nbytes -= array.field("path").nbytes

            table = self.with_format("arrow")[:1000]
            table_visitor(table, extra_nbytes_visitor)

            extra_nbytes = extra_nbytes * len(self.data) / len(table)
            dataset_nbytes = dataset_nbytes + extra_nbytes

        if self._indices is not None:
            dataset_nbytes = dataset_nbytes * len(self._indices) / len(self.data)
        return dataset_nbytes

    @staticmethod
    def _generate_tables_from_shards(shards: List["Dataset"], batch_size: int):
        for shard_idx, shard in enumerate(shards):
            for pa_table in shard.with_format("arrow").iter(batch_size):
                yield shard_idx, pa_table

    @staticmethod
    def _generate_tables_from_cache_file(filename: str):
        for batch_idx, batch in enumerate(_memory_mapped_record_batch_reader_from_file(filename)):
            yield batch_idx, pa.Table.from_batches([batch])

    def to_iterable_dataset(self, num_shards: Optional[int] = 1) -> "IterableDataset":
        """Get an [`datasets.IterableDataset`] from a map-style [`datasets.Dataset`].
        This is equivalent to loading a dataset in streaming mode with [`datasets.load_dataset`], but much faster since the data is streamed from local files.

        Contrary to map-style datasets, iterable datasets are lazy and can only be iterated over (e.g. using a for loop).
        Since they are read sequentially in training loops, iterable datasets are much faster than map-style datasets.
        All the transformations applied to iterable datasets like filtering or processing are done on-the-fly when you start iterating over the dataset.

        Still, it is possible to shuffle an iterable dataset using [`datasets.IterableDataset.shuffle`].
        This is a fast approximate shuffling that works best if you have multiple shards and if you specify a buffer size that is big enough.

        To get the best speed performance, make sure your dataset doesn't have an indices mapping.
        If this is the case, the data are not read contiguously, which can be slow sometimes.
        You can use `ds = ds.flatten_indices()` to write your dataset in contiguous chunks of data and have optimal speed before switching to an iterable dataset.

        Args:
            num_shards (`int`, default to `1`):
                Number of shards to define when instantiating the iterable dataset. This is especially useful for big datasets to be able to shuffle properly,
                and also to enable fast parallel loading using a PyTorch DataLoader or in distributed setups for example.
                Shards are defined using [`datasets.Dataset.shard`]: it simply slices the data without writing anything on disk.

        Returns:
            [`datasets.IterableDataset`]

        Example:

        Basic usage:
        ```python
        >>> ids = ds.to_iterable_dataset()
        >>> for example in ids:
        ...     pass
        ```

        With lazy filtering and processing:
        ```python
        >>> ids = ds.to_iterable_dataset()
        >>> ids = ids.filter(filter_fn).map(process_fn)  # will filter and process on-the-fly when you start iterating over the iterable dataset
        >>> for example in ids:
        ...     pass
        ```

        With sharding to enable efficient shuffling:
        ```python
        >>> ids = ds.to_iterable_dataset(num_shards=64)  # the dataset is split into 64 shards to be iterated over
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer for fast approximate shuffling when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With a PyTorch DataLoader:
        ```python
        >>> import torch
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.filter(filter_fn).map(process_fn)
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards to each worker to load, filter and process when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With a PyTorch DataLoader and shuffling:
        ```python
        >>> import torch
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from the shuffled list of shards to each worker when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        In a distributed setup like PyTorch DDP with a PyTorch DataLoader and shuffling
        ```python
        >>> from datasets.distributed import split_dataset_by_node
        >>> ids = ds.to_iterable_dataset(num_shards=512)
        >>> ids = ids.shuffle(buffer_size=10_000)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> ids = split_dataset_by_node(ds, world_size=8, rank=0)  # will keep only 512 / 8 = 64 shards from the shuffled lists of shards when you start iterating
        >>> dataloader = torch.utils.data.DataLoader(ids, num_workers=4)  # will assign 64 / 4 = 16 shards from this node's list of shards to each worker when you start iterating
        >>> for example in ids:
        ...     pass
        ```

        With shuffling and multiple epochs:
        ```python
        >>> ids = ds.to_iterable_dataset(num_shards=64)
        >>> ids = ids.shuffle(buffer_size=10_000, seed=42)  # will shuffle the shards order and use a shuffle buffer when you start iterating
        >>> for epoch in range(n_epochs):
        ...     ids.set_epoch(epoch)  # will use effective_seed = seed + epoch to shuffle the shards and for the shuffle buffer when you start iterating
        ...     for example in ids:
        ...         pass
        ```
        Feel free to also use [`IterableDataset.set_epoch`] when using a PyTorch DataLoader or in distributed setups.
        """
        from .iterable_dataset import ArrowExamplesIterable, IterableDataset

        if self._format_type is not None:
            raise NotImplementedError(
                "Converting a formatted dataset to a formatted iterable dataset is not implemented yet. Please run `my_dataset = my_dataset.with_format(None)` before calling to_iterable_dataset"
            )
        if num_shards > len(self):
            raise ValueError(
                f"Unable to shard a dataset of size {len(self)} into {num_shards} shards (the number of shards exceeds the number of samples)."
            )
        if self._indices is not None:
            logger.info(
                "Converting an Arrow dataset to iterable but it has an indices mapping that can make it slower. "
                "You can use `ds = ds.flatten_indices()` to write your dataset in contiguous chunks of data and have optimal speed."
            )
        shards = (
            [copy.deepcopy(self)]
            if num_shards == 1
            else [
                self.shard(num_shards=num_shards, index=shard_idx, contiguous=True) for shard_idx in range(num_shards)
            ]
        )
        ex_iterable = ArrowExamplesIterable(
            Dataset._generate_tables_from_shards,
            kwargs={"shards": shards, "batch_size": config.DEFAULT_MAX_BATCH_SIZE},
        )
        return IterableDataset(ex_iterable, info=DatasetInfo(features=self.features))

    def _push_parquet_shards_to_hub(
        self,
        repo_id: str,
        data_dir: str = "data",
        split: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[int] = None,
        embed_external_files: bool = True,
    ) -> Tuple[str, str, int, int, List[str], int]:
        """Pushes the dataset shards as Parquet files to the hub.

        Returns:
            additions (`List[CommitOperation]`): list of the `CommitOperationAdd` of the uploaded shards
            uploaded_size (`int`): number of uploaded bytes to the repository
            dataset_nbytes (`int`): approximate size in bytes of the uploaded dataset afer uncompression
        """
        # Find decodable columns, because if there are any, we need to:
        # embed the bytes from the files in the shards
        decodable_columns = (
            [k for k, v in self._info.features.items() if require_decoding(v, ignore_decode_attribute=True)]
            if embed_external_files
            else []
        )

        dataset_nbytes = self._estimate_nbytes()

        if num_shards is None:
            max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)
            num_shards = int(dataset_nbytes / max_shard_size) + 1
            num_shards = max(num_shards, 1)

        shards = (self.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))

        if decodable_columns:

            def shards_with_embedded_external_files(shards):
                for shard in shards:
                    format = shard.format
                    shard = shard.with_format("arrow")
                    shard = shard.map(
                        embed_table_storage,
                        batched=True,
                        batch_size=1000,
                        keep_in_memory=True,
                    )
                    shard = shard.with_format(**format)
                    yield shard

            shards = shards_with_embedded_external_files(shards)

        api = HfApi(endpoint=config.HF_ENDPOINT, token=token)

        uploaded_size = 0
        additions = []
        for index, shard in hf_tqdm(
            enumerate(shards),
            desc="Uploading the dataset shards",
            total=num_shards,
        ):
            shard_path_in_repo = f"{data_dir}/{split}-{index:05d}-of-{num_shards:05d}.parquet"
            buffer = BytesIO()
            shard.to_parquet(buffer)
            uploaded_size += buffer.tell()
            shard_addition = CommitOperationAdd(path_in_repo=shard_path_in_repo, path_or_fileobj=buffer)
            api.preupload_lfs_files(
                repo_id=repo_id,
                additions=[shard_addition],
                repo_type="dataset",
                revision=revision,
                create_pr=create_pr,
            )
            additions.append(shard_addition)

        return additions, uploaded_size, dataset_nbytes

    def push_to_hub(
        self,
        repo_id: str,
        config_name: str = "default",
        set_default: Optional[bool] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        branch="deprecated",
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[int] = None,
        embed_external_files: bool = True,
    ) -> CommitInfo:
        """Pushes the dataset to the hub as a Parquet dataset.
        The dataset is pushed using HTTP requests and does not need to have neither git or git-lfs installed.

        The resulting Parquet files are self-contained by default. If your dataset contains [`Image`] or [`Audio`]
        data, the Parquet files will store the bytes of your images or audio files.
        You can disable this by setting `embed_external_files` to `False`.

        Args:
            repo_id (`str`):
                The ID of the repository to push to in the following format: `<user>/<dataset_name>` or
                `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace
                of the logged-in user.
            config_name (`str`, defaults to "default"):
                The configuration name (or subset) of a dataset. Defaults to "default".
            set_default (`bool`, *optional*):
                Whether to set this configuration as the default one. Otherwise, the default configuration is the one
                named "default".
            split (`str`, *optional*):
                The name of the split that will be given to that dataset. Defaults to `self.split`.
            data_dir (`str`, *optional*):
                Directory name that will contain the uploaded data files. Defaults to the `config_name` if different
                from "default", else "data".

                <Added version="2.17.0"/>
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload dataset"`.
            commit_description (`str`, *optional*):
                Description of the commit that will be created.
                Additionally, description of the PR if a PR is created (`create_pr` is True).

                <Added version="2.16.0"/>
            private (`bool`, *optional*, defaults to `False`):
                Whether the dataset repository should be set to private or not. Only affects repository creation:
                a repository that already exists will not be affected by that parameter.
            token (`str`, *optional*):
                An optional authentication token for the Hugging Face Hub. If no token is passed, will default
                to the token saved locally when logging in with `huggingface-cli login`. Will raise an error
                if no token is passed and the user is not logged-in.
            revision (`str`, *optional*):
                Branch to push the uploaded files to. Defaults to the `"main"` branch.

                <Added version="2.15.0"/>
            branch (`str`, *optional*):
                The git branch on which to push the dataset. This defaults to the default branch as specified
                in your repository, which defaults to `"main"`.

                <Deprecated version="2.15.0">

                `branch` was deprecated in favor of `revision` in version 2.15.0 and will be removed in 3.0.0.

                </Deprecated>
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.

                <Added version="2.15.0"/>
            max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
                The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by
                a unit (like `"5MB"`).
            num_shards (`int`, *optional*):
                Number of shards to write. By default, the number of shards depends on `max_shard_size`.

                <Added version="2.8.0"/>
            embed_external_files (`bool`, defaults to `True`):
                Whether to embed file bytes in the shards.
                In particular, this will do the following before the push for the fields of type:

                - [`Audio`] and [`Image`]: remove local path information and embed file content in the Parquet files.

        Return:
            huggingface_hub.CommitInfo

        Example:

        ```python
        >>> dataset.push_to_hub("<organization>/<dataset_id>")
        >>> dataset_dict.push_to_hub("<organization>/<dataset_id>", private=True)
        >>> dataset.push_to_hub("<organization>/<dataset_id>", max_shard_size="1GB")
        >>> dataset.push_to_hub("<organization>/<dataset_id>", num_shards=1024)
        ```

        If your dataset has multiple splits (e.g. train/validation/test):

        ```python
        >>> train_dataset.push_to_hub("<organization>/<dataset_id>", split="train")
        >>> val_dataset.push_to_hub("<organization>/<dataset_id>", split="validation")
        >>> # later
        >>> dataset = load_dataset("<organization>/<dataset_id>")
        >>> train_dataset = dataset["train"]
        >>> val_dataset = dataset["validation"]
        ```

        If you want to add a new configuration (or subset) to a dataset (e.g. if the dataset has multiple tasks/versions/languages):

        ```python
        >>> english_dataset.push_to_hub("<organization>/<dataset_id>", "en")
        >>> french_dataset.push_to_hub("<organization>/<dataset_id>", "fr")
        >>> # later
        >>> english_dataset = load_dataset("<organization>/<dataset_id>", "en")
        >>> french_dataset = load_dataset("<organization>/<dataset_id>", "fr")
        ```
        """
        if config_name == "data":
            raise ValueError("`config_name` cannot be 'data'. Please, choose another name for configuration.")

        if max_shard_size is not None and num_shards is not None:
            raise ValueError(
                "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
            )

        if split is None:
            split = str(self.split) if self.split is not None else "train"

        if not re.match(_split_re, split):
            raise ValueError(f"Split name should match '{_split_re}' but got '{split}'.")

        if branch != "deprecated":
            warnings.warn(
                "'branch' was deprecated in favor of 'revision' in version 2.15.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'revision={branch}' instead.",
                FutureWarning,
            )
            revision = branch

        api = HfApi(endpoint=config.HF_ENDPOINT, token=token)

        repo_url = api.create_repo(
            repo_id,
            token=token,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        if revision is not None and not revision.startswith("refs/pr/"):
            # We do not call create_branch for a PR reference: 400 Bad Request
            api.create_branch(repo_id, branch=revision, token=token, repo_type="dataset", exist_ok=True)

        if not data_dir:
            data_dir = config_name if config_name != "default" else "data"  # for backward compatibility

        additions, uploaded_size, dataset_nbytes = self._push_parquet_shards_to_hub(
            repo_id=repo_id,
            data_dir=data_dir,
            split=split,
            token=token,
            revision=revision,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            create_pr=create_pr,
            embed_external_files=embed_external_files,
        )

        # Check if the repo already has a README.md and/or a dataset_infos.json to update them with the new split info (size and pattern)
        # and delete old split shards (if they exist)
        repo_with_dataset_card, repo_with_dataset_infos = False, False
        deletions, deleted_size = [], 0
        repo_splits = []  # use a list to keep the order of the splits
        repo_files_to_add = [addition.path_in_repo for addition in additions]
        for repo_file in api.list_repo_tree(
            repo_id=repo_id, revision=revision, repo_type="dataset", token=token, recursive=True
        ):
            if not isinstance(repo_file, RepoFile):
                continue
            if repo_file.rfilename == config.REPOCARD_FILENAME:
                repo_with_dataset_card = True
            elif repo_file.rfilename == config.DATASETDICT_INFOS_FILENAME:
                repo_with_dataset_infos = True
            elif (
                repo_file.rfilename.startswith(f"{data_dir}/{split}-") and repo_file.rfilename not in repo_files_to_add
            ):
                deletions.append(CommitOperationDelete(path_in_repo=repo_file.rfilename))
                deleted_size += repo_file.size
            elif fnmatch.fnmatch(
                repo_file.rfilename, PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED.replace("{split}", "*")
            ):
                repo_split = string_to_dict(
                    repo_file.rfilename,
                    glob_pattern_to_regex(PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED),
                )["split"]
                if repo_split not in repo_splits:
                    repo_splits.append(repo_split)

        organization, dataset_name = repo_id.split("/") if "/" in repo_id else (None, repo_id)
        info_to_dump = self.info.copy()
        info_to_dump.download_checksums = None
        info_to_dump.download_size = uploaded_size
        info_to_dump.dataset_size = dataset_nbytes
        info_to_dump.size_in_bytes = uploaded_size + dataset_nbytes
        info_to_dump.config_name = config_name
        info_to_dump.splits = SplitDict(
            {split: SplitInfo(split, num_bytes=dataset_nbytes, num_examples=len(self), dataset_name=dataset_name)}
        )
        # get the info from the README to update them
        if repo_with_dataset_card:
            dataset_card_path = api.hf_hub_download(
                repo_id, config.REPOCARD_FILENAME, repo_type="dataset", revision=revision
            )
            dataset_card = DatasetCard.load(Path(dataset_card_path))
            dataset_card_data = dataset_card.data
            metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
            dataset_infos: DatasetInfosDict = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
            if dataset_infos and config_name in dataset_infos:
                repo_info = dataset_infos[config_name]
            else:
                repo_info = None
        # get the deprecated dataset_infos.json to update them
        elif repo_with_dataset_infos:
            dataset_card = None
            dataset_card_data = DatasetCardData()
            metadata_configs = MetadataConfigs()
            dataset_infos_path = api.hf_hub_download(
                repo_id, config.DATASETDICT_INFOS_FILENAME, repo_type="dataset", revision=revision
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                dataset_infos: dict = json.load(f)
                dataset_info = dataset_infos.get(config_name, None) if dataset_infos else None
                repo_info = DatasetInfo.from_dict(dataset_info) if dataset_info else None
        else:
            dataset_card = None
            dataset_card_data = DatasetCardData()
            metadata_configs = MetadataConfigs()
            repo_info = None
        # update the total info to dump from existing info
        if repo_info is not None:
            logger.info("Updating downloaded metadata with the new split.")
            if repo_info.splits and list(repo_info.splits) != [split]:
                if self._info.features != repo_info.features:
                    raise ValueError(
                        f"Features of the new split don't match the features of the existing splits on the hub: {self._info.features} != {repo_info.features}"
                    )

                if split in repo_info.splits:
                    repo_info.download_size -= deleted_size
                    repo_info.dataset_size -= repo_info.splits.get(split, SplitInfo()).num_bytes or 0

                repo_info.download_checksums = None
                repo_info.download_size = (repo_info.download_size or 0) + uploaded_size
                repo_info.dataset_size = (repo_info.dataset_size or 0) + dataset_nbytes
                repo_info.size_in_bytes = repo_info.download_size + repo_info.dataset_size
                repo_info.splits.pop(split, None)
                repo_info.splits[split] = SplitInfo(
                    split, num_bytes=dataset_nbytes, num_examples=len(self), dataset_name=dataset_name
                )
                info_to_dump = repo_info
        # create the metadata configs if it was uploaded with push_to_hub before metadata configs existed
        if not metadata_configs and repo_splits:
            default_metadata_configs_to_dump = {
                "data_files": [{"split": split, "path": f"data/{split}-*"} for split in repo_splits]
            }
            MetadataConfigs({"default": default_metadata_configs_to_dump}).to_dataset_card_data(dataset_card_data)
        # update the metadata configs
        if config_name in metadata_configs:
            metadata_config = metadata_configs[config_name]
            if "data_files" in metadata_config:
                data_files_to_dump = sanitize_patterns(metadata_config["data_files"])
            else:
                data_files_to_dump = {}
            # add the new split
            data_files_to_dump[split] = [f"{data_dir}/{split}-*"]
            metadata_config_to_dump = {
                "data_files": [
                    {
                        "split": _split,
                        "path": _pattern[0] if len(_pattern) == 1 else _pattern,
                    }
                    for _split, _pattern in data_files_to_dump.items()
                ]
            }
        else:
            metadata_config_to_dump = {"data_files": [{"split": split, "path": f"{data_dir}/{split}-*"}]}
        if set_default and config_name != "default":
            if metadata_configs:
                default_config_name = metadata_configs.get_default_config_name()
                if default_config_name == "default":
                    raise ValueError(
                        "There exists a configuration named 'default'. To set a different configuration as default, "
                        "rename the 'default' one first."
                    )
                else:
                    _ = metadata_configs[default_config_name].pop("default")
            metadata_config_to_dump["default"] = True
        # push to the deprecated dataset_infos.json
        if repo_with_dataset_infos:
            dataset_infos_path = api.hf_hub_download(
                repo_id, config.DATASETDICT_INFOS_FILENAME, repo_type="dataset", revision=revision
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                dataset_infos: dict = json.load(f)
            dataset_infos[config_name] = asdict(info_to_dump)
            buffer = BytesIO()
            buffer.write(json.dumps(dataset_infos, indent=4).encode("utf-8"))
            additions.append(
                CommitOperationAdd(path_in_repo=config.DATASETDICT_INFOS_FILENAME, path_or_fileobj=buffer)
            )
        # push to README
        DatasetInfosDict({config_name: info_to_dump}).to_dataset_card_data(dataset_card_data)
        MetadataConfigs({config_name: metadata_config_to_dump}).to_dataset_card_data(dataset_card_data)
        dataset_card = DatasetCard(f"---\n{dataset_card_data}\n---\n") if dataset_card is None else dataset_card
        additions.append(
            CommitOperationAdd(path_in_repo=config.REPOCARD_FILENAME, path_or_fileobj=str(dataset_card).encode())
        )

        commit_message = commit_message if commit_message is not None else "Upload dataset"
        if len(additions) <= config.UPLOADS_MAX_NUMBER_PER_COMMIT:
            commit_info = api.create_commit(
                repo_id,
                operations=additions + deletions,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                repo_type="dataset",
                revision=revision,
                create_pr=create_pr,
            )
        else:
            logger.info(
                f"Number of files to upload is larger than {config.UPLOADS_MAX_NUMBER_PER_COMMIT}. Splitting the push into multiple commits."
            )
            num_commits = math.ceil(len(additions) / config.UPLOADS_MAX_NUMBER_PER_COMMIT)
            for i in range(0, num_commits):
                operations = additions[
                    i * config.UPLOADS_MAX_NUMBER_PER_COMMIT : (i + 1) * config.UPLOADS_MAX_NUMBER_PER_COMMIT
                ] + (deletions if i == 0 else [])
                commit_info = api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=commit_message + f" (part {i:05d}-of-{num_commits:05d})",
                    commit_description=commit_description,
                    token=token,
                    repo_type="dataset",
                    revision=revision,
                    create_pr=create_pr,
                )
                logger.info(
                    f"Commit #{i+1} completed"
                    + (f" (still {num_commits - i - 1} to go)" if num_commits - i - 1 else "")
                    + "."
                )
        return commit_info

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_column(self, name: str, column: Union[list, np.array], new_fingerprint: str):
        """Add column to Dataset.

        <Added version="1.7"/>

        Args:
            name (`str`):
                Column name.
            column (`list` or `np.array`):
                Column data to be added.

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> more_text = ds["text"]
        >>> ds.add_column(name="text_2", column=more_text)
        Dataset({
            features: ['text', 'label', 'text_2'],
            num_rows: 1066
        })
        ```
        """
        column_table = InMemoryTable.from_pydict({name: column})
        _check_column_names(self._data.column_names + column_table.column_names)
        dataset = self.flatten_indices() if self._indices is not None else self
        # Concatenate tables horizontally
        table = concat_tables([dataset._data, column_table], axis=1)
        # Update features
        info = dataset.info.copy()
        info.features.update(Features.from_arrow_schema(column_table.schema))
        table = update_metadata_with_features(table, info.features)
        return Dataset(table, info=info, split=self.split, indices_table=None, fingerprint=new_fingerprint)

    def add_faiss_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        device: Optional[int] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,  # noqa: F821
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
        dtype=np.float32,
    ):
        """Add a dense index using Faiss for fast retrieval.
        By default the index is done over the vectors of the specified column.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:

        - For [string factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

        Args:
            column (`str`):
                The column of the vectors to add to the index.
            index_name (`str`, *optional*):
                The `index_name`/identifier of the index.
                This is the `index_name` that is used to call [`~datasets.Dataset.get_nearest_examples`] or [`~datasets.Dataset.search`].
                By default it corresponds to `column`.
            device (`Union[int, List[int]]`, *optional*):
                If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (`str`, *optional*):
                This is passed to the index factory of Faiss to create the index.
                Default index class is `IndexFlat`.
            metric_type (`int`, *optional*):
                Type of metric. Ex: `faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (`faiss.Index`, *optional*):
                Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (`int`):
                Size of the batch to use while adding vectors to the `FaissIndex`. Default value is `1000`.
                <Added version="2.4.0"/>
            train_size (`int`, *optional*):
                If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to `False`):
                Enable the verbosity of the Faiss index.
            dtype (`data-type`):
                The dtype of the numpy arrays that are indexed.
                Default is `np.float32`.

        Example:

        ```python
        >>> ds = datasets.load_dataset('crime_and_punish', split='train')
        >>> ds_with_embeddings = ds.map(lambda example: {'embeddings': embed(example['line']}))
        >>> ds_with_embeddings.add_faiss_index(column='embeddings')
        >>> # query
        >>> scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', embed('my new query'), k=10)
        >>> # save index
        >>> ds_with_embeddings.save_faiss_index('embeddings', 'my_index.faiss')

        >>> ds = datasets.load_dataset('crime_and_punish', split='train')
        >>> # load index
        >>> ds.load_faiss_index('embeddings', 'my_index.faiss')
        >>> # query
        >>> scores, retrieved_examples = ds.get_nearest_examples('embeddings', embed('my new query'), k=10)
        ```
        """
        with self.formatted_as(type="numpy", columns=[column], dtype=dtype):
            super().add_faiss_index(
                column=column,
                index_name=index_name,
                device=device,
                string_factory=string_factory,
                metric_type=metric_type,
                custom_index=custom_index,
                batch_size=batch_size,
                train_size=train_size,
                faiss_verbose=faiss_verbose,
            )
        return self

    def add_faiss_index_from_external_arrays(
        self,
        external_arrays: np.array,
        index_name: str,
        device: Optional[int] = None,
        string_factory: Optional[str] = None,
        metric_type: Optional[int] = None,
        custom_index: Optional["faiss.Index"] = None,  # noqa: F821
        batch_size: int = 1000,
        train_size: Optional[int] = None,
        faiss_verbose: bool = False,
        dtype=np.float32,
    ):
        """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of `external_arrays`.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:

        - For [string factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

        Args:
            external_arrays (`np.array`):
                If you want to use arrays from outside the lib for the index, you can set `external_arrays`.
                It will use `external_arrays` to create the Faiss index instead of the arrays in the given `column`.
            index_name (`str`):
                The `index_name`/identifier of the index.
                This is the `index_name` that is used to call [`~datasets.Dataset.get_nearest_examples`] or [`~datasets.Dataset.search`].
            device (Optional `Union[int, List[int]]`, *optional*):
                If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (`str`, *optional*):
                This is passed to the index factory of Faiss to create the index.
                Default index class is `IndexFlat`.
            metric_type (`int`, *optional*):
                Type of metric. Ex: `faiss.faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (`faiss.Index`, *optional*):
                Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (`int`, *optional*):
                Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (`int`, *optional*):
                If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False):
                Enable the verbosity of the Faiss index.
            dtype (`numpy.dtype`):
                The dtype of the numpy arrays that are indexed. Default is np.float32.
        """
        super().add_faiss_index_from_external_arrays(
            external_arrays=external_arrays.astype(dtype),
            index_name=index_name,
            device=device,
            string_factory=string_factory,
            metric_type=metric_type,
            custom_index=custom_index,
            batch_size=batch_size,
            train_size=train_size,
            faiss_verbose=faiss_verbose,
        )

    def add_elasticsearch_index(
        self,
        column: str,
        index_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        es_client: Optional["elasticsearch.Elasticsearch"] = None,  # noqa: F821
        es_index_name: Optional[str] = None,
        es_index_config: Optional[dict] = None,
    ):
        """Add a text index using ElasticSearch for fast retrieval. This is done in-place.

        Args:
            column (`str`):
                The column of the documents to add to the index.
            index_name (`str`, *optional*):
                The `index_name`/identifier of the index.
                This is the index name that is used to call [`~Dataset.get_nearest_examples`] or [`~Dataset.search`].
                By default it corresponds to `column`.
            host (`str`, *optional*, defaults to `localhost`):
                Host of where ElasticSearch is running.
            port (`str`, *optional*, defaults to `9200`):
                Port of where ElasticSearch is running.
            es_client (`elasticsearch.Elasticsearch`, *optional*):
                The elasticsearch client used to create the index if host and port are `None`.
            es_index_name (`str`, *optional*):
                The elasticsearch index name used to create the index.
            es_index_config (`dict`, *optional*):
                The configuration of the elasticsearch index.
                Default config is:
                    ```
                    {
                        "settings": {
                            "number_of_shards": 1,
                            "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
                        },
                        "mappings": {
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "standard",
                                    "similarity": "BM25"
                                },
                            }
                        },
                    }
                    ```
        Example:

        ```python
        >>> es_client = elasticsearch.Elasticsearch()
        >>> ds = datasets.load_dataset('crime_and_punish', split='train')
        >>> ds.add_elasticsearch_index(column='line', es_client=es_client, es_index_name="my_es_index")
        >>> scores, retrieved_examples = ds.get_nearest_examples('line', 'my new query', k=10)
        ```
        """
        with self.formatted_as(type=None, columns=[column]):
            super().add_elasticsearch_index(
                column=column,
                index_name=index_name,
                host=host,
                port=port,
                es_client=es_client,
                es_index_name=es_index_name,
                es_index_config=es_index_config,
            )
        return self

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_item(self, item: dict, new_fingerprint: str):
        """Add item to Dataset.

        <Added version="1.7"/>

        Args:
            item (`dict`):
                Item data to be added.

        Returns:
            [`Dataset`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> new_review = {'label': 0, 'text': 'this movie is the absolute worst thing I have ever seen'}
        >>> ds = ds.add_item(new_review)
        >>> ds[-1]
        {'label': 0, 'text': 'this movie is the absolute worst thing I have ever seen'}
        ```
        """
        item_table = InMemoryTable.from_pydict({k: [v] for k, v in item.items()})
        # We don't call _check_if_features_can_be_aligned here so this cast is "unsafe"
        dset_features, item_features = _align_features(
            [self._info.features, Features.from_arrow_schema(item_table.schema)]
        )
        # Cast to align the schemas of the tables and concatenate the tables
        table = concat_tables(
            [
                self._data.cast(dset_features.arrow_schema) if self._info.features != dset_features else self._data,
                item_table.cast(item_features.arrow_schema),
            ]
        )
        if self._indices is None:
            indices_table = None
        else:
            item_indices_array = pa.array([len(self._data)], type=pa.uint64())
            item_indices_table = InMemoryTable.from_arrays([item_indices_array], names=["indices"])
            indices_table = concat_tables([self._indices, item_indices_table])
        info = self.info.copy()
        info.features.update(item_features)
        table = update_metadata_with_features(table, info.features)
        return Dataset(
            table,
            info=info,
            split=self.split,
            indices_table=indices_table,
            fingerprint=new_fingerprint,
        )

    def align_labels_with_mapping(self, label2id: Dict, label_column: str) -> "Dataset":
        """Align the dataset's label ID and label name mapping to match an input `label2id` mapping.
        This is useful when you want to ensure that a model's predicted labels are aligned with the dataset.
        The alignment in done using the lowercase label names.

        Args:
            label2id (`dict`):
                The label name to ID mapping to align the dataset with.
            label_column (`str`):
                The column name of labels to align on.

        Example:

        ```python
        >>> # dataset with mapping {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        >>> ds = load_dataset("glue", "mnli", split="train")
        >>> # mapping to align with
        >>> label2id = {'CONTRADICTION': 0, 'NEUTRAL': 1, 'ENTAILMENT': 2}
        >>> ds_aligned = ds.align_labels_with_mapping(label2id, "label")
        ```

        """
        # Sanity checks
        if label_column not in self._data.column_names:
            raise ValueError(f"Column ({label_column}) not in table columns ({self._data.column_names}).")

        label_feature = self._info.features[label_column]
        if not (
            isinstance(label_feature, ClassLabel)
            or (isinstance(label_feature, Sequence) and isinstance(label_feature.feature, ClassLabel))
        ):
            raise ValueError(
                f"Aligning labels with a mapping is only supported for {ClassLabel.__name__} column or {Sequence.__name__} column with the inner type {ClassLabel.__name__}, and column {label_feature} is of type {type(label_feature).__name__}."
            )

        # Sort input mapping by ID value to ensure the label names are aligned
        label2id = dict(sorted(label2id.items(), key=lambda item: item[1]))
        label_names = list(label2id.keys())
        # Some label mappings use uppercase label names so we lowercase them during alignment
        label2id = {k.lower(): v for k, v in label2id.items()}
        int2str_function = (
            label_feature.int2str if isinstance(label_feature, ClassLabel) else label_feature.feature.int2str
        )

        if isinstance(label_feature, ClassLabel):

            def process_label_ids(batch):
                dset_label_names = [
                    int2str_function(label_id).lower() if label_id is not None else None
                    for label_id in batch[label_column]
                ]
                batch[label_column] = [
                    label2id[label_name] if label_name is not None else None for label_name in dset_label_names
                ]
                return batch

        else:

            def process_label_ids(batch):
                dset_label_names = [
                    [int2str_function(label_id).lower() if label_id is not None else None for label_id in seq]
                    for seq in batch[label_column]
                ]
                batch[label_column] = [
                    [label2id[label_name] if label_name is not None else None for label_name in seq]
                    for seq in dset_label_names
                ]
                return batch

        features = self.features
        features[label_column] = (
            ClassLabel(num_classes=len(label_names), names=label_names)
            if isinstance(label_feature, ClassLabel)
            else Sequence(ClassLabel(num_classes=len(label_names), names=label_names))
        )
        return self.map(process_label_ids, features=features, batched=True, desc="Aligning the labels")


def _concatenate_map_style_datasets(
    dsets: List[Dataset],
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    axis: int = 0,
):
    """
    Converts a list of :class:`Dataset` with the same schema into a single :class:`Dataset`.
    When you concatenate on axis 0, missing data are filled with None values.

    Args:
        dsets (`List[datasets.Dataset]`): List of Datasets to concatenate.
        info (:class:`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (:class:`NamedSplit`, optional): Name of the dataset split.
        axis (``{0, 1}``, default ``0``, meaning over rows):
            Axis to concatenate over, where ``0`` means over rows (vertically) and ``1`` means over columns
            (horizontally).

            *New in version 1.6.0*

    Example:

    ```py
    >>> ds3 = _concatenate_map_style_datasets([ds1, ds2])
    ```
    """
    # Ignore datasets with no rows
    if any(dset.num_rows > 0 for dset in dsets):
        dsets = [dset for dset in dsets if dset.num_rows > 0]
    else:
        # Return first dataset if all datasets are empty
        return dsets[0]

    # Perform checks (and a potentional cast if axis=0)
    if axis == 0:
        _check_if_features_can_be_aligned([dset.features for dset in dsets])
    else:
        if not all(dset.num_rows == dsets[0].num_rows for dset in dsets):
            raise ValueError("Number of rows must match for all datasets")
        _check_column_names([col_name for dset in dsets for col_name in dset._data.column_names])

    # Find common format or reset format
    format = dsets[0].format
    if any(dset.format != format for dset in dsets):
        format = {}
        logger.info("Some of the datasets have disparate format. Resetting the format of the concatenated dataset.")

    def apply_offset_to_indices_table(table, offset):
        if offset == 0:
            return table
        else:
            array = table["indices"]
            new_array = pc.add(array, pa.scalar(offset, type=pa.uint64()))
            return InMemoryTable.from_arrays([new_array], names=["indices"])

    # Concatenate indices if they exist
    if any(dset._indices is not None for dset in dsets):
        if axis == 0:
            # Datasets with no indices tables are replaced with a dataset with an indices table in memory.
            # Applying an offset to an indices table also brings the table in memory.
            indices_tables = []
            for i in range(len(dsets)):
                if dsets[i]._indices is None:
                    dsets[i] = dsets[i]._select_with_indices_mapping(range(len(dsets[i])))
                indices_tables.append(dsets[i]._indices)

            # An offset needs to be applied to the indices before concatenating
            offset = 0
            for i in range(len(dsets)):
                indices_tables[i] = apply_offset_to_indices_table(indices_tables[i], offset)
                offset += len(dsets[i]._data)

            # Concatenate indices
            indices_tables = [t for t in indices_tables if len(t) > 0]
            if indices_tables:
                indices_table = concat_tables(indices_tables)
            else:
                indices_table = InMemoryTable.from_batches([], schema=pa.schema({"indices": pa.int64()}))
        else:
            if len(dsets) == 1:
                indices_table = dsets[0]._indices
            else:
                for i in range(len(dsets)):
                    dsets[i] = dsets[i].flatten_indices()
                indices_table = None
    else:
        indices_table = None

    table = concat_tables([dset._data for dset in dsets], axis=axis)
    if axis == 0:
        features_list = _align_features([dset.features for dset in dsets])
    else:
        features_list = [dset.features for dset in dsets]
    table = update_metadata_with_features(table, {k: v for features in features_list for k, v in features.items()})

    # Concatenate infos
    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint(
        "".join(dset._fingerprint for dset in dsets), _concatenate_map_style_datasets, {"info": info, "split": split}
    )

    # Make final concatenated dataset
    concatenated_dataset = Dataset(
        table,
        info=info,
        split=split,
        indices_table=indices_table,
        fingerprint=fingerprint,
    )
    concatenated_dataset.set_format(**format)
    return concatenated_dataset


def _interleave_map_style_datasets(
    datasets: List["Dataset"],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    **kwargs,
) -> "Dataset":
    """
    Interleave several map-style datasets (sources) into a single map-style dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    If `probabilities = None` (default) the new dataset is constructed by cycling between each source to get the examples.
    If `probabilities` is not `None, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    Args:
        datasets (`List[Dataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        info (:class:`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (:class:`NamedSplit`, optional): Name of the dataset split.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
        **kwargs (additional keyword arguments): Keyword arguments to be passed to :meth:`datasets.Datasets.select` when selecting the indices used to interleave the datasets.

    Output:
        :class:`datasets.Dataset`
    """
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(
            f"{stopping_strategy} stopping strategy in `interleave_datasets` is not implemented yet with a list of {type(datasets[0])}"
        )

    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = _concatenate_map_style_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])

    # if stopping_strategy is "first_exhausted", it is an undersampling situation whereas it is an oversampling situation if it is "all_exhausted"
    oversampling = stopping_strategy == "all_exhausted"

    if probabilities is None and not oversampling:
        # Undersampling situation with cycling between each sources
        # Example:: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples

        # Reasoning behind the following operation: keeping the min_length first indices of each dataset
        # while offsetting in order to correspond to the right indices of the concatenated dataset
        # and flattening to effectively interleave the datasets
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    elif probabilities is None:
        # Oversampling situation with cycling between each sources
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 5, 9, 0, 6, 10, 1, 3, 11]
        # Note that we have 5 examples per dataset with a rolling window since the longest dataset has 5 samples

        # Reasoning behind the following operation: for each dataset indices (i.e column) repeat the indices to have max_length indices per dataset
        # For example, if the max_length is 5 and the i-th dataset has 3 samples, the i-th column will be [0,1,2,0,1]
        indices = np.mod(np.arange(max(lengths)).reshape(-1, 1), np.array(lengths).reshape(1, -1))

        # We have to keep the indices to their respective dataset offsets and to flatten to effectively interleave the datasets
        indices = (indices + offsets).flatten().tolist()

    else:
        # boolean array indicating if at index i if the dataset_i has been fully exhausted
        is_exhausted = np.full(len(lengths), False)

        # if undersampling ("first_exhausted"), we stop as soon as one dataset is exhausted
        # if oversampling ("all_exhausted"), we stop as soons as every dataset is exhausted, i.e as soon as every samples of every dataset has been visited at least once
        bool_strategy_func = np.all if oversampling else np.any

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))

        current_index = [0] * len(datasets)
        indices = []
        for source_idx in iter_random_indices():
            # If no oversampling, we stop as soon as a dataset has ran out of examples (np.any)
            # Otherwise, we stop as soon as every dataset has ran out of examples (np.all)
            if bool_strategy_func(is_exhausted):
                # the stopping condition was reached, let's stop
                break

            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1

            # we've ran out of examples for the current dataset, let's update our boolean array and bring the current_index back to 0
            if current_index[source_idx] >= lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0

    return concatenated_datasets.select(indices, **kwargs)


def _split_by_node_map_style_dataset(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    """
    Split a dataset for the node at rank `rank` in a pool of nodes of size `world_size`.
    Each node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset.
    To maximize data loading throughput, chunks are made of contiguous data on disk if possible.

    Args:
        dataset ([`Dataset`]):
            The dataset to split by node.
        rank (`int`):
            Rank of the current node.
        world_size (`int`):
            Total number of nodes.

    Returns:
        [`Dataset`]: The dataset to be used on the node at rank `rank`.
    """
    return dataset.shard(num_shards=world_size, index=rank, contiguous=True)


# This is outside Dataset.filter as it needs to be picklable for multiprocessing


def get_indices_from_mask_function(
    function: Callable,
    batched: bool,
    with_indices: bool,
    with_rank: bool,
    input_columns: Optional[Union[str, List[str]]],
    indices_mapping: Optional[Table] = None,
    *args,
    **fn_kwargs,
):
    if batched:
        # we extract indices and rank from args
        *inputs, indices, rank = args
        additional_args = ()
        if with_indices:
            additional_args += (indices,)
        if with_rank:
            additional_args += (rank,)
        mask = function(*inputs, *additional_args, **fn_kwargs)
    else:
        # we get batched data (to do less look-ups) but `function` only accepts one example
        # therefore we need to call `function` on each example of the batch to get the mask
        *inputs, indices, rank = args
        mask = []
        if input_columns is None:
            # inputs only contains a batch of examples
            batch: dict = inputs[0]
            num_examples = len(batch[next(iter(batch.keys()))])
            for i in range(num_examples):
                example = {key: batch[key][i] for key in batch}
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(function(example, *additional_args, **fn_kwargs))
        else:
            # inputs is a list of columns
            columns: List[List] = inputs
            num_examples = len(columns[0])
            for i in range(num_examples):
                input = [column[i] for column in columns]
                additional_args = ()
                if with_indices:
                    additional_args += (indices[i],)
                if with_rank:
                    additional_args += (rank,)
                mask.append(function(*input, *additional_args, **fn_kwargs))
    indices_array = [i for i, to_keep in zip(indices, mask) if to_keep]
    if indices_mapping is not None:
        indices_array = pa.array(indices_array, type=pa.uint64())
        indices_array = indices_mapping.column(0).take(indices_array)
        indices_array = indices_array.to_pylist()
    return {"indices": indices_array}
