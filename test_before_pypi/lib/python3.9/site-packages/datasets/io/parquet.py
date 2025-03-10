import os
from typing import BinaryIO, Optional, Union

import fsspec
import numpy as np
import pyarrow.parquet as pq

from .. import Audio, Dataset, Features, Image, NamedSplit, Value, config
from ..features.features import FeatureType, _visit
from ..formatting import query_table
from ..packaged_modules import _PACKAGED_DATASETS_MODULES
from ..packaged_modules.parquet.parquet import Parquet
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


def get_writer_batch_size(features: Features) -> Optional[int]:
    """
    Get the writer_batch_size that defines the maximum row group size in the parquet files.
    The default in `datasets` is 1,000 but we lower it to 100 for image datasets.
    This allows to optimize random access to parquet file, since accessing 1 row requires
    to read its entire row group.

    This can be improved to get optimized size for querying/iterating
    but at least it matches the dataset viewer expectations on HF.

    Args:
        ds_config_info (`datasets.info.DatasetInfo`):
            Dataset info from `datasets`.
    Returns:
        writer_batch_size (`Optional[int]`):
            Writer batch size to pass to a dataset builder.
            If `None`, then it will use the `datasets` default.
    """

    batch_size = np.inf

    def set_batch_size(feature: FeatureType) -> None:
        nonlocal batch_size
        if isinstance(feature, Image):
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS)
        elif isinstance(feature, Audio):
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS)
        elif isinstance(feature, Value) and feature.dtype == "binary":
            batch_size = min(batch_size, config.PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS)

    _visit(features, set_batch_size)

    return None if batch_size is np.inf else batch_size


class ParquetDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        hash = _PACKAGED_DATASETS_MODULES["parquet"][1]
        self.builder = Parquet(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            hash=hash,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(
                split=self.split, verification_mode=verification_mode, in_memory=self.keep_in_memory
            )
        return dataset


class ParquetDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **parquet_writer_kwargs,
    ):
        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size or get_writer_batch_size(dataset.features)
        self.storage_options = storage_options or {}
        self.parquet_writer_kwargs = parquet_writer_kwargs

    def write(self) -> int:
        batch_size = self.batch_size if self.batch_size else config.DEFAULT_MAX_BATCH_SIZE

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(self.path_or_buf, "wb", **(self.storage_options or {})) as buffer:
                written = self._write(file_obj=buffer, batch_size=batch_size, **self.parquet_writer_kwargs)
        else:
            written = self._write(file_obj=self.path_or_buf, batch_size=batch_size, **self.parquet_writer_kwargs)
        return written

    def _write(self, file_obj: BinaryIO, batch_size: int, **parquet_writer_kwargs) -> int:
        """Writes the pyarrow table as Parquet to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0
        _ = parquet_writer_kwargs.pop("path_or_buf", None)
        schema = self.dataset.features.arrow_schema

        writer = pq.ParquetWriter(file_obj, schema=schema, **parquet_writer_kwargs)

        for offset in hf_tqdm(
            range(0, len(self.dataset), batch_size),
            unit="ba",
            desc="Creating parquet from Arrow format",
        ):
            batch = query_table(
                table=self.dataset._data,
                key=slice(offset, offset + batch_size),
                indices=self.dataset._indices,
            )
            writer.write_table(batch)
            written += batch.nbytes
        writer.close()
        return written
