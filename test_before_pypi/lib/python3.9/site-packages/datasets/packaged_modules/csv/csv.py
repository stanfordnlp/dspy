import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

import datasets
import datasets.config
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
from datasets.utils.py_utils import Literal


logger = datasets.utils.logging.get_logger(__name__)

_PANDAS_READ_CSV_NO_DEFAULT_PARAMETERS = ["names", "prefix"]
_PANDAS_READ_CSV_DEPRECATED_PARAMETERS = ["warn_bad_lines", "error_bad_lines", "mangle_dupe_cols"]
_PANDAS_READ_CSV_NEW_1_3_0_PARAMETERS = ["encoding_errors", "on_bad_lines"]
_PANDAS_READ_CSV_NEW_2_0_0_PARAMETERS = ["date_format"]
_PANDAS_READ_CSV_DEPRECATED_2_2_0_PARAMETERS = ["verbose"]


@dataclass
class CsvConfig(datasets.BuilderConfig):
    """BuilderConfig for CSV."""

    sep: str = ","
    delimiter: Optional[str] = None
    header: Optional[Union[int, List[int], str]] = "infer"
    names: Optional[List[str]] = None
    column_names: Optional[List[str]] = None
    index_col: Optional[Union[int, str, List[int], List[str]]] = None
    usecols: Optional[Union[List[int], List[str]]] = None
    prefix: Optional[str] = None
    mangle_dupe_cols: bool = True
    engine: Optional[Literal["c", "python", "pyarrow"]] = None
    converters: Dict[Union[int, str], Callable[[Any], Any]] = None
    true_values: Optional[list] = None
    false_values: Optional[list] = None
    skipinitialspace: bool = False
    skiprows: Optional[Union[int, List[int]]] = None
    nrows: Optional[int] = None
    na_values: Optional[Union[str, List[str]]] = None
    keep_default_na: bool = True
    na_filter: bool = True
    verbose: bool = False
    skip_blank_lines: bool = True
    thousands: Optional[str] = None
    decimal: str = "."
    lineterminator: Optional[str] = None
    quotechar: str = '"'
    quoting: int = 0
    escapechar: Optional[str] = None
    comment: Optional[str] = None
    encoding: Optional[str] = None
    dialect: Optional[str] = None
    error_bad_lines: bool = True
    warn_bad_lines: bool = True
    skipfooter: int = 0
    doublequote: bool = True
    memory_map: bool = False
    float_precision: Optional[str] = None
    chunksize: int = 10_000
    features: Optional[datasets.Features] = None
    encoding_errors: Optional[str] = "strict"
    on_bad_lines: Literal["error", "warn", "skip"] = "error"
    date_format: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.delimiter is not None:
            self.sep = self.delimiter
        if self.column_names is not None:
            self.names = self.column_names

    @property
    def pd_read_csv_kwargs(self):
        pd_read_csv_kwargs = {
            "sep": self.sep,
            "header": self.header,
            "names": self.names,
            "index_col": self.index_col,
            "usecols": self.usecols,
            "prefix": self.prefix,
            "mangle_dupe_cols": self.mangle_dupe_cols,
            "engine": self.engine,
            "converters": self.converters,
            "true_values": self.true_values,
            "false_values": self.false_values,
            "skipinitialspace": self.skipinitialspace,
            "skiprows": self.skiprows,
            "nrows": self.nrows,
            "na_values": self.na_values,
            "keep_default_na": self.keep_default_na,
            "na_filter": self.na_filter,
            "verbose": self.verbose,
            "skip_blank_lines": self.skip_blank_lines,
            "thousands": self.thousands,
            "decimal": self.decimal,
            "lineterminator": self.lineterminator,
            "quotechar": self.quotechar,
            "quoting": self.quoting,
            "escapechar": self.escapechar,
            "comment": self.comment,
            "encoding": self.encoding,
            "dialect": self.dialect,
            "error_bad_lines": self.error_bad_lines,
            "warn_bad_lines": self.warn_bad_lines,
            "skipfooter": self.skipfooter,
            "doublequote": self.doublequote,
            "memory_map": self.memory_map,
            "float_precision": self.float_precision,
            "chunksize": self.chunksize,
            "encoding_errors": self.encoding_errors,
            "on_bad_lines": self.on_bad_lines,
            "date_format": self.date_format,
        }

        # some kwargs must not be passed if they don't have a default value
        # some others are deprecated and we can also not pass them if they are the default value
        for pd_read_csv_parameter in _PANDAS_READ_CSV_NO_DEFAULT_PARAMETERS + _PANDAS_READ_CSV_DEPRECATED_PARAMETERS:
            if pd_read_csv_kwargs[pd_read_csv_parameter] == getattr(CsvConfig(), pd_read_csv_parameter):
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 1.3 new arguments
        if not (datasets.config.PANDAS_VERSION.major >= 1 and datasets.config.PANDAS_VERSION.minor >= 3):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_NEW_1_3_0_PARAMETERS:
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 2.0 new arguments
        if not (datasets.config.PANDAS_VERSION.major >= 2):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_NEW_2_0_0_PARAMETERS:
                del pd_read_csv_kwargs[pd_read_csv_parameter]

        # Remove 2.2 deprecated arguments
        if datasets.config.PANDAS_VERSION.release >= (2, 2):
            for pd_read_csv_parameter in _PANDAS_READ_CSV_DEPRECATED_2_2_0_PARAMETERS:
                if pd_read_csv_kwargs[pd_read_csv_parameter] == getattr(CsvConfig(), pd_read_csv_parameter):
                    del pd_read_csv_kwargs[pd_read_csv_parameter]

        return pd_read_csv_kwargs


class Csv(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = CsvConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all(not require_storage_cast(feature) for feature in self.config.features.values()):
                # cheaper cast
                pa_table = pa.Table.from_arrays([pa_table[field.name] for field in schema], schema=schema)
            else:
                # more expensive cast; allows str <-> int/float or str to Audio for example
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_tables(self, files):
        schema = self.config.features.arrow_schema if self.config.features else None
        # dtype allows reading an int column as str
        dtype = (
            {
                name: dtype.to_pandas_dtype() if not require_storage_cast(feature) else object
                for name, dtype, feature in zip(schema.names, schema.types, self.config.features.values())
            }
            if schema is not None
            else None
        )
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            csv_file_reader = pd.read_csv(file, iterator=True, dtype=dtype, **self.config.pd_read_csv_kwargs)
            try:
                for batch_idx, df in enumerate(csv_file_reader):
                    pa_table = pa.Table.from_pandas(df)
                    # Uncomment for debugging (will print the Arrow table size and elements)
                    # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                    # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                    yield (file_idx, batch_idx), self._cast_table(pa_table)
            except ValueError as e:
                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise
