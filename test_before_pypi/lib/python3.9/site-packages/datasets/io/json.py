import multiprocessing
import os
from typing import BinaryIO, Optional, Union

import fsspec

from .. import Dataset, Features, NamedSplit, config
from ..formatting import query_table
from ..packaged_modules.json.json import Json
from ..utils import tqdm as hf_tqdm
from ..utils.typing import NestedDataStructureLike, PathLike
from .abc import AbstractDatasetReader


class JsonDatasetReader(AbstractDatasetReader):
    def __init__(
        self,
        path_or_paths: NestedDataStructureLike[PathLike],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        field: Optional[str] = None,
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
        self.field = field
        path_or_paths = path_or_paths if isinstance(path_or_paths, dict) else {self.split: path_or_paths}
        self.builder = Json(
            cache_dir=cache_dir,
            data_files=path_or_paths,
            features=features,
            field=field,
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
                # try_from_hf_gcs=try_from_hf_gcs,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(
                split=self.split, verification_mode=verification_mode, in_memory=self.keep_in_memory
            )
        return dataset


class JsonDatasetWriter:
    def __init__(
        self,
        dataset: Dataset,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **to_json_kwargs,
    ):
        if num_proc is not None and num_proc <= 0:
            raise ValueError(f"num_proc {num_proc} must be an integer > 0.")

        self.dataset = dataset
        self.path_or_buf = path_or_buf
        self.batch_size = batch_size if batch_size else config.DEFAULT_MAX_BATCH_SIZE
        self.num_proc = num_proc
        self.encoding = "utf-8"
        self.storage_options = storage_options or {}
        self.to_json_kwargs = to_json_kwargs

    def write(self) -> int:
        _ = self.to_json_kwargs.pop("path_or_buf", None)
        orient = self.to_json_kwargs.pop("orient", "records")
        lines = self.to_json_kwargs.pop("lines", True if orient == "records" else False)
        if "index" not in self.to_json_kwargs and orient in ["split", "table"]:
            self.to_json_kwargs["index"] = False

        # Determine the default compression value based on self.path_or_buf type
        default_compression = "infer" if isinstance(self.path_or_buf, (str, bytes, os.PathLike)) else None
        compression = self.to_json_kwargs.pop("compression", default_compression)

        if compression not in [None, "infer", "gzip", "bz2", "xz"]:
            raise NotImplementedError(f"`datasets` currently does not support {compression} compression")

        if isinstance(self.path_or_buf, (str, bytes, os.PathLike)):
            with fsspec.open(
                self.path_or_buf, "wb", compression=compression, **(self.storage_options or {})
            ) as buffer:
                written = self._write(file_obj=buffer, orient=orient, lines=lines, **self.to_json_kwargs)
        else:
            if compression:
                raise NotImplementedError(
                    f"The compression parameter is not supported when writing to a buffer, but compression={compression}"
                    " was passed. Please provide a local path instead."
                )
            written = self._write(file_obj=self.path_or_buf, orient=orient, lines=lines, **self.to_json_kwargs)
        return written

    def _batch_json(self, args):
        offset, orient, lines, to_json_kwargs = args

        batch = query_table(
            table=self.dataset.data,
            key=slice(offset, offset + self.batch_size),
            indices=self.dataset._indices,
        )
        json_str = batch.to_pandas().to_json(path_or_buf=None, orient=orient, lines=lines, **to_json_kwargs)
        if not json_str.endswith("\n"):
            json_str += "\n"
        return json_str.encode(self.encoding)

    def _write(
        self,
        file_obj: BinaryIO,
        orient,
        lines,
        **to_json_kwargs,
    ) -> int:
        """Writes the pyarrow table as JSON lines to a binary file handle.

        Caller is responsible for opening and closing the handle.
        """
        written = 0

        if self.num_proc is None or self.num_proc == 1:
            for offset in hf_tqdm(
                range(0, len(self.dataset), self.batch_size),
                unit="ba",
                desc="Creating json from Arrow format",
            ):
                json_str = self._batch_json((offset, orient, lines, to_json_kwargs))
                written += file_obj.write(json_str)
        else:
            num_rows, batch_size = len(self.dataset), self.batch_size
            with multiprocessing.Pool(self.num_proc) as pool:
                for json_str in hf_tqdm(
                    pool.imap(
                        self._batch_json,
                        [(offset, orient, lines, to_json_kwargs) for offset in range(0, num_rows, batch_size)],
                    ),
                    total=(num_rows // batch_size) + 1 if num_rows % batch_size else num_rows // batch_size,
                    unit="ba",
                    desc="Creating json from Arrow format",
                ):
                    written += file_obj.write(json_str)

        return written
