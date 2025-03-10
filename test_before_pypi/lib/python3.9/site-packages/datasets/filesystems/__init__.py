import importlib
import shutil
import warnings
from typing import List

import fsspec
import fsspec.asyn
from fsspec.implementations.local import LocalFileSystem

from ..utils.deprecation_utils import deprecated
from . import compression


_has_s3fs = importlib.util.find_spec("s3fs") is not None

if _has_s3fs:
    from .s3filesystem import S3FileSystem

COMPRESSION_FILESYSTEMS: List[compression.BaseCompressedFileFileSystem] = [
    compression.Bz2FileSystem,
    compression.GzipFileSystem,
    compression.Lz4FileSystem,
    compression.XzFileSystem,
    compression.ZstdFileSystem,
]

# Register custom filesystems
for fs_class in COMPRESSION_FILESYSTEMS:
    if fs_class.protocol in fsspec.registry and fsspec.registry[fs_class.protocol] is not fs_class:
        warnings.warn(f"A filesystem protocol was already set for {fs_class.protocol} and will be overwritten.")
    fsspec.register_implementation(fs_class.protocol, fs_class, clobber=True)


@deprecated(
    "This function is deprecated and will be removed in a future version. Please use `fsspec.core.strip_protocol` instead."
)
def extract_path_from_uri(dataset_path: str) -> str:
    """
    Preprocesses `dataset_path` and removes remote filesystem (e.g. removing `s3://`).

    Args:
        dataset_path (`str`):
            Path (e.g. `dataset/train`) or remote uri (e.g. `s3://my-bucket/dataset/train`) of the dataset directory.
    """
    if "://" in dataset_path:
        dataset_path = dataset_path.split("://")[1]
    return dataset_path


def is_remote_filesystem(fs: fsspec.AbstractFileSystem) -> bool:
    """
    Checks if `fs` is a remote filesystem.

    Args:
        fs (`fsspec.spec.AbstractFileSystem`):
            An abstract super-class for pythonic file-systems, e.g. `fsspec.filesystem(\'file\')` or [`datasets.filesystems.S3FileSystem`].
    """
    return not isinstance(fs, LocalFileSystem)


def rename(fs: fsspec.AbstractFileSystem, src: str, dst: str):
    """
    Renames the file `src` in `fs` to `dst`.
    """
    if not is_remote_filesystem(fs):
        # LocalFileSystem.mv does copy + rm, it is more efficient to simply move a local directory
        shutil.move(fs._strip_protocol(src), fs._strip_protocol(dst))
    else:
        fs.mv(src, dst, recursive=True)
