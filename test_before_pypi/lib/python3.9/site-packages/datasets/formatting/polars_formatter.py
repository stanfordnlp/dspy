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

import sys
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Optional

import pyarrow as pa

from .. import config
from ..features import Features
from ..features.features import decode_nested_example
from ..utils.py_utils import no_op_if_value_is_null
from .formatting import BaseArrowExtractor, TensorFormatter


if TYPE_CHECKING:
    import polars as pl


class PolarsArrowExtractor(BaseArrowExtractor["pl.DataFrame", "pl.Series", "pl.DataFrame"]):
    def extract_row(self, pa_table: pa.Table) -> "pl.DataFrame":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table.slice(length=1))
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")

    def extract_column(self, pa_table: pa.Table) -> "pl.Series":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table.select([0]))[pa_table.column_names[0]]
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")

    def extract_batch(self, pa_table: pa.Table) -> "pl.DataFrame":
        if config.POLARS_AVAILABLE:
            if "polars" not in sys.modules:
                import polars
            else:
                polars = sys.modules["polars"]

            return polars.from_arrow(pa_table)
        else:
            raise ValueError("Polars needs to be installed to be able to return Polars dataframes.")


class PolarsFeaturesDecoder:
    def __init__(self, features: Optional[Features]):
        self.features = features
        import polars as pl  # noqa: F401 - import pl at initialization

    def decode_row(self, row: "pl.DataFrame") -> "pl.DataFrame":
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
            row[list(decode.keys())] = row.map_rows(decode)
        return row

    def decode_column(self, column: "pl.Series", column_name: str) -> "pl.Series":
        decode = (
            no_op_if_value_is_null(partial(decode_nested_example, self.features[column_name]))
            if self.features and column_name in self.features and self.features._column_requires_decoding[column_name]
            else None
        )
        if decode:
            column = column.map_elements(decode)
        return column

    def decode_batch(self, batch: "pl.DataFrame") -> "pl.DataFrame":
        return self.decode_row(batch)


class PolarsFormatter(TensorFormatter[Mapping, "pl.DataFrame", Mapping]):
    def __init__(self, features=None, **np_array_kwargs):
        super().__init__(features=features)
        self.np_array_kwargs = np_array_kwargs
        self.polars_arrow_extractor = PolarsArrowExtractor
        self.polars_features_decoder = PolarsFeaturesDecoder(features)
        import polars as pl  # noqa: F401 - import pl at initialization

    def format_row(self, pa_table: pa.Table) -> "pl.DataFrame":
        row = self.polars_arrow_extractor().extract_row(pa_table)
        row = self.polars_features_decoder.decode_row(row)
        return row

    def format_column(self, pa_table: pa.Table) -> "pl.Series":
        column = self.polars_arrow_extractor().extract_column(pa_table)
        column = self.polars_features_decoder.decode_column(column, pa_table.column_names[0])
        return column

    def format_batch(self, pa_table: pa.Table) -> "pl.DataFrame":
        row = self.polars_arrow_extractor().extract_batch(pa_table)
        row = self.polars_features_decoder.decode_batch(row)
        return row
