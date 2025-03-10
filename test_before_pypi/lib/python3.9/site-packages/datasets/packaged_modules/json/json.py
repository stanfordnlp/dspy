import io
import itertools
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.json as paj

import datasets
import datasets.config
from datasets.table import table_cast
from datasets.utils.file_utils import readline


logger = datasets.utils.logging.get_logger(__name__)


def ujson_dumps(*args, **kwargs):
    try:
        return pd.io.json.ujson_dumps(*args, **kwargs)
    except AttributeError:
        # Before pandas-2.2.0, ujson_dumps was renamed to dumps: import ujson_dumps as dumps
        return pd.io.json.dumps(*args, **kwargs)


def ujson_loads(*args, **kwargs):
    try:
        return pd.io.json.ujson_loads(*args, **kwargs)
    except AttributeError:
        # Before pandas-2.2.0, ujson_loads was renamed to loads: import ujson_loads as loads
        return pd.io.json.loads(*args, **kwargs)


def pandas_read_json(path_or_buf, **kwargs):
    if datasets.config.PANDAS_VERSION.major >= 2:
        kwargs["dtype_backend"] = "pyarrow"
    return pd.read_json(path_or_buf, **kwargs)


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    encoding: str = "utf-8"
    encoding_errors: Optional[str] = None
    field: Optional[str] = None
    use_threads: bool = True  # deprecated
    block_size: Optional[int] = None  # deprecated
    chunksize: int = 10 << 20  # 10MB
    newlines_in_values: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        if self.config.block_size is not None:
            logger.warning("The JSON loader parameter `block_size` is deprecated. Please use `chunksize` instead")
            self.config.chunksize = self.config.block_size
        if self.config.use_threads is not True:
            logger.warning(
                "The JSON loader parameter `use_threads` is deprecated and doesn't have any effect anymore."
            )
        if self.config.newlines_in_values is not None:
            raise ValueError("The JSON loader parameter `newlines_in_values` is no longer supported")
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
            # adding missing columns
            for column_name in set(self.config.features) - set(pa_table.column_names):
                type = self.config.features.arrow_schema.field(column_name).type
                pa_table = pa_table.append_column(column_name, pa.array([None] * len(pa_table), type=type))
            # more expensive cast to support nested structures with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.config.features.arrow_schema)
        return pa_table

    def _generate_tables(self, files):
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            # If the file is one json object and if we need to look at the items in one specific field
            if self.config.field is not None:
                with open(file, encoding=self.config.encoding, errors=self.config.encoding_errors) as f:
                    dataset = ujson_loads(f.read())
                # We keep only the field we are interested in
                dataset = dataset[self.config.field]
                df = pandas_read_json(io.StringIO(ujson_dumps(dataset)))
                if df.columns.tolist() == [0]:
                    df.columns = list(self.config.features) if self.config.features else ["text"]
                pa_table = pa.Table.from_pandas(df, preserve_index=False)
                yield file_idx, self._cast_table(pa_table)

            # If the file has one json object per line
            else:
                with open(file, "rb") as f:
                    batch_idx = 0
                    # Use block_size equal to the chunk size divided by 32 to leverage multithreading
                    # Set a default minimum value of 16kB if the chunk size is really small
                    block_size = max(self.config.chunksize // 32, 16 << 10)
                    encoding_errors = (
                        self.config.encoding_errors if self.config.encoding_errors is not None else "strict"
                    )
                    while True:
                        batch = f.read(self.config.chunksize)
                        if not batch:
                            break
                        # Finish current line
                        try:
                            batch += f.readline()
                        except (AttributeError, io.UnsupportedOperation):
                            batch += readline(f)
                        # PyArrow only accepts utf-8 encoded bytes
                        if self.config.encoding != "utf-8":
                            batch = batch.decode(self.config.encoding, errors=encoding_errors).encode("utf-8")
                        try:
                            while True:
                                try:
                                    pa_table = paj.read_json(
                                        io.BytesIO(batch), read_options=paj.ReadOptions(block_size=block_size)
                                    )
                                    break
                                except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                                    if (
                                        isinstance(e, pa.ArrowInvalid)
                                        and "straddling" not in str(e)
                                        or block_size > len(batch)
                                    ):
                                        raise
                                    else:
                                        # Increase the block size in case it was too small.
                                        # The block size will be reset for the next file.
                                        logger.debug(
                                            f"Batch of {len(batch)} bytes couldn't be parsed with block_size={block_size}. Retrying with block_size={block_size * 2}."
                                        )
                                        block_size *= 2
                        except pa.ArrowInvalid as e:
                            try:
                                with open(
                                    file, encoding=self.config.encoding, errors=self.config.encoding_errors
                                ) as f:
                                    df = pandas_read_json(f)
                            except ValueError:
                                logger.error(f"Failed to load JSON from file '{file}' with error {type(e)}: {e}")
                                raise e
                            if df.columns.tolist() == [0]:
                                df.columns = list(self.config.features) if self.config.features else ["text"]
                            try:
                                pa_table = pa.Table.from_pandas(df, preserve_index=False)
                            except pa.ArrowInvalid as e:
                                logger.error(
                                    f"Failed to convert pandas DataFrame to Arrow Table from file '{file}' with error {type(e)}: {e}"
                                )
                                raise ValueError(
                                    f"Failed to convert pandas DataFrame to Arrow Table from file {file}."
                                ) from None
                            yield file_idx, self._cast_table(pa_table)
                            break
                        yield (file_idx, batch_idx), self._cast_table(pa_table)
                        batch_idx += 1
