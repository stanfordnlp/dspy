import os
from functools import partial
from typing import Optional

import fsspec
from fsspec.archive import AbstractArchiveFileSystem


class BaseCompressedFileFileSystem(AbstractArchiveFileSystem):
    """Read contents of compressed file as a filesystem with one file inside."""

    root_marker = ""
    protocol: str = (
        None  # protocol passed in prefix to the url. ex: "gzip", for gzip://file.txt::http://foo.bar/file.txt.gz
    )
    compression: str = None  # compression type in fsspec. ex: "gzip"
    extension: str = None  # extension of the filename to strip. ex: "".gz" to get file.txt from file.txt.gz

    def __init__(
        self, fo: str = "", target_protocol: Optional[str] = None, target_options: Optional[dict] = None, **kwargs
    ):
        """
        The compressed file system can be instantiated from any compressed file.
        It reads the contents of compressed file as a filesystem with one file inside, as if it was an archive.

        The single file inside the filesystem is named after the compresssed file,
        without the compression extension at the end of the filename.

        Args:
            fo (:obj:``str``): Path to compressed file. Will fetch file using ``fsspec.open()``
            mode (:obj:``str``): Currently, only 'rb' accepted
            target_protocol(:obj:``str``, optional): To override the FS protocol inferred from a URL.
            target_options (:obj:``dict``, optional): Kwargs passed when instantiating the target FS.
        """
        super().__init__(self, **kwargs)
        self.fo = fo.__fspath__() if hasattr(fo, "__fspath__") else fo
        # always open as "rb" since fsspec can then use the TextIOWrapper to make it work for "r" mode
        self._open_with_fsspec = partial(
            fsspec.open,
            self.fo,
            mode="rb",
            protocol=target_protocol,
            compression=self.compression,
            client_kwargs={
                "requote_redirect_url": False,  # see https://github.com/huggingface/datasets/pull/5459
                "trust_env": True,  # Enable reading proxy env variables.
                **(target_options or {}).pop("client_kwargs", {}),  # To avoid issues if it was already passed.
            },
            **(target_options or {}),
        )
        self.compressed_name = os.path.basename(self.fo.split("::")[0])
        self.uncompressed_name = (
            self.compressed_name[: self.compressed_name.rindex(".")]
            if "." in self.compressed_name
            else self.compressed_name
        )
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        # compressed file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def _get_dirs(self):
        if self.dir_cache is None:
            f = {**self._open_with_fsspec().fs.info(self.fo), "name": self.uncompressed_name}
            self.dir_cache = {f["name"]: f}

    def cat(self, path: str):
        with self._open_with_fsspec().open() as f:
            return f.read()

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if mode != "rb":
            raise ValueError(f"Tried to read with mode {mode} on file {self.fo} opened with mode 'rb'")
        return self._open_with_fsspec().open()


class Bz2FileSystem(BaseCompressedFileFileSystem):
    """Read contents of BZ2 file as a filesystem with one file inside."""

    protocol = "bz2"
    compression = "bz2"
    extension = ".bz2"


class GzipFileSystem(BaseCompressedFileFileSystem):
    """Read contents of GZIP file as a filesystem with one file inside."""

    protocol = "gzip"
    compression = "gzip"
    extension = ".gz"


class Lz4FileSystem(BaseCompressedFileFileSystem):
    """Read contents of LZ4 file as a filesystem with one file inside."""

    protocol = "lz4"
    compression = "lz4"
    extension = ".lz4"


class XzFileSystem(BaseCompressedFileFileSystem):
    """Read contents of .xz (LZMA) file as a filesystem with one file inside."""

    protocol = "xz"
    compression = "xz"
    extension = ".xz"


class ZstdFileSystem(BaseCompressedFileFileSystem):
    """
    Read contents of .zstd file as a filesystem with one file inside.
    """

    protocol = "zstd"
    compression = "zstd"
    extension = ".zst"
