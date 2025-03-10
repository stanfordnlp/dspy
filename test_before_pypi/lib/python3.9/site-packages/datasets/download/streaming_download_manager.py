import io
import os
from typing import Iterable, List, Optional, Tuple, Union

from ..utils.file_utils import (  # noqa: F401 # backward compatibility
    SINGLE_FILE_COMPRESSION_PROTOCOLS,
    ArchiveIterable,
    FilesIterable,
    _get_extraction_protocol,
    _get_path_extension,
    _prepare_path_and_storage_options,
    is_relative_path,
    url_or_path_join,
    xbasename,
    xdirname,
    xet_parse,
    xexists,
    xgetsize,
    xglob,
    xgzip_open,
    xisdir,
    xisfile,
    xjoin,
    xlistdir,
    xnumpy_load,
    xopen,
    xpandas_read_csv,
    xpandas_read_excel,
    xPath,
    xpyarrow_parquet_read_table,
    xrelpath,
    xsio_loadmat,
    xsplit,
    xsplitext,
    xwalk,
    xxml_dom_minidom_parse,
)
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig


logger = get_logger(__name__)


class StreamingDownloadManager:
    """
    Download manager that uses the "::" separator to navigate through (possibly remote) compressed archives.
    Contrary to the regular `DownloadManager`, the `download` and `extract` methods don't actually download nor extract
    data, but they rather return the path or url that could be opened using the `xopen` function which extends the
    built-in `open` function to stream data from remote files.
    """

    is_streaming = True

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        base_path: Optional[str] = None,
    ):
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._base_path = base_path or os.path.abspath(".")
        self.download_config = download_config or DownloadConfig()

    @property
    def manual_dir(self):
        return self._data_dir

    def download(self, url_or_urls):
        """Normalize URL(s) of files to stream data from.
        This is the lazy version of `DownloadManager.download` for streaming.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) of files to stream data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input url_or_urls.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        ```
        """
        url_or_urls = map_nested(self._download_single, url_or_urls, map_tuple=True)
        return url_or_urls

    def _download_single(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        if is_relative_path(urlpath):
            # append the relative path to the base_path
            urlpath = url_or_path_join(self._base_path, urlpath)
        return urlpath

    def extract(self, url_or_urls):
        """Add extraction protocol for given url(s) for streaming.

        This is the lazy version of `DownloadManager.extract` for streaming.

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) of files to stream data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.

        Example:

        ```py
        >>> downloaded_files = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        >>> extracted_files = dl_manager.extract(downloaded_files)
        ```
        """
        urlpaths = map_nested(self._extract, url_or_urls, map_tuple=True)
        return urlpaths

    def _extract(self, urlpath: str) -> str:
        urlpath = str(urlpath)
        protocol = _get_extraction_protocol(urlpath, download_config=self.download_config)
        # get inner file: zip://train-00000.json.gz::https://foo.bar/data.zip -> zip://train-00000.json.gz
        path = urlpath.split("::")[0]
        extension = _get_path_extension(path)
        if extension in ["tgz", "tar"] or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
            raise NotImplementedError(
                f"Extraction protocol for TAR archives like '{urlpath}' is not implemented in streaming mode. "
                f"Please use `dl_manager.iter_archive` instead.\n\n"
                f"Example usage:\n\n"
                f"\turl = dl_manager.download(url)\n"
                f"\ttar_archive_iterator = dl_manager.iter_archive(url)\n\n"
                f"\tfor filename, file in tar_archive_iterator:\n"
                f"\t\t..."
            )
        if protocol is None:
            # no extraction
            return urlpath
        elif protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
            # there is one single file which is the uncompressed file
            inner_file = os.path.basename(urlpath.split("::")[0])
            inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
            return f"{protocol}://{inner_file}::{urlpath}"
        else:
            return f"{protocol}://::{urlpath}"

    def download_and_extract(self, url_or_urls):
        """Prepare given `url_or_urls` for streaming (add extraction protocol).

        This is the lazy version of `DownloadManager.download_and_extract` for streaming.

        Is equivalent to:

        ```
        urls = dl_manager.extract(dl_manager.download(url_or_urls))
        ```

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL(s) to stream from data from. Each url is a `str`.

        Returns:
            url(s): (`str` or `list` or `dict`), URL(s) to stream data from matching the given input `url_or_urls`.
        """
        return self.extract(self.download(url_or_urls))

    def iter_archive(self, urlpath_or_buf: Union[str, io.BufferedReader]) -> Iterable[Tuple]:
        """Iterate over files within an archive.

        Args:
            urlpath_or_buf (`str` or `io.BufferedReader`):
                Archive path or archive binary file object.

        Yields:
            `tuple[str, io.BufferedReader]`:
                2-tuple (path_within_archive, file_object).
                File object is opened in binary mode.

        Example:

        ```py
        >>> archive = dl_manager.download('https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz')
        >>> files = dl_manager.iter_archive(archive)
        ```
        """

        if hasattr(urlpath_or_buf, "read"):
            return ArchiveIterable.from_buf(urlpath_or_buf)
        else:
            return ArchiveIterable.from_urlpath(urlpath_or_buf, download_config=self.download_config)

    def iter_files(self, urlpaths: Union[str, List[str]]) -> Iterable[str]:
        """Iterate over files.

        Args:
            urlpaths (`str` or `list` of `str`):
                Root paths.

        Yields:
            str: File URL path.

        Example:

        ```py
        >>> files = dl_manager.download_and_extract('https://huggingface.co/datasets/beans/resolve/main/data/train.zip')
        >>> files = dl_manager.iter_files(files)
        ```
        """
        return FilesIterable.from_urlpaths(urlpaths, download_config=self.download_config)
