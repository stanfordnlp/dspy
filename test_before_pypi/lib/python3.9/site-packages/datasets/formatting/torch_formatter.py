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
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter


if TYPE_CHECKING:
    import torch


class TorchFormatter(TensorFormatter[Mapping, "torch.Tensor", Mapping]):
    def __init__(self, features=None, **torch_tensor_kwargs):
        super().__init__(features=features)
        self.torch_tensor_kwargs = torch_tensor_kwargs
        import torch  # noqa import torch at initialization

    def _consolidate(self, column):
        import torch

        if isinstance(column, list) and column:
            if all(
                isinstance(x, torch.Tensor) and x.shape == column[0].shape and x.dtype == column[0].dtype
                for x in column
            ):
                return torch.stack(column)
        return column

    def _tensorize(self, value):
        import torch

        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()

        default_dtype = {}

        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": torch.int64}

            # Convert dtype to np.int64 if it's either np.uint16 or np.uint32 to ensure compatibility.
            # np.uint64 is excluded from this conversion as there is no compatible PyTorch dtype that can handle it without loss.
            if value.dtype in [np.uint16, np.uint32]:
                value = value.astype(np.int64)

        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": torch.float32}
        elif config.PIL_AVAILABLE and "PIL" in sys.modules:
            import PIL.Image

            if isinstance(value, PIL.Image.Image):
                value = np.asarray(value)
                if value.ndim == 2:
                    value = value[:, :, np.newaxis]

                value = value.transpose((2, 0, 1))
        return torch.tensor(value, **{**default_dtype, **self.torch_tensor_kwargs})

    def _recursive_tensorize(self, data_struct):
        import torch

        # support for torch, tf, jax etc.
        if hasattr(data_struct, "__array__") and not isinstance(data_struct, torch.Tensor):
            data_struct = data_struct.__array__()
        # support for nested types like struct of list of struct
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype == object:  # torch tensors cannot be instantied from an array of objects
                return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (list, tuple)):
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        return self._tensorize(data_struct)

    def recursive_tensorize(self, data_struct: dict):
        return map_nested(self._recursive_tensorize, data_struct, map_list=False)

    def format_row(self, pa_table: pa.Table) -> Mapping:
        row = self.numpy_arrow_extractor().extract_row(pa_table)
        row = self.python_features_decoder.decode_row(row)
        return self.recursive_tensorize(row)

    def format_column(self, pa_table: pa.Table) -> "torch.Tensor":
        column = self.numpy_arrow_extractor().extract_column(pa_table)
        column = self.python_features_decoder.decode_column(column, pa_table.column_names[0])
        column = self.recursive_tensorize(column)
        column = self._consolidate(column)
        return column

    def format_batch(self, pa_table: pa.Table) -> Mapping:
        batch = self.numpy_arrow_extractor().extract_batch(pa_table)
        batch = self.python_features_decoder.decode_batch(batch)
        batch = self.recursive_tensorize(batch)
        for column_name in batch:
            batch[column_name] = self._consolidate(batch[column_name])
        return batch
