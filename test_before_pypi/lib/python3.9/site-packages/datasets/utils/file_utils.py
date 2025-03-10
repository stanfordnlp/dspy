"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import asyncio
import copy
import glob
import io
import json
import multiprocessing
import os
import posixpath
import re
import shutil
import sys
import tarfile
import time
import urllib
import warnings
import xml.dom.minidom
import zipfile
from contextlib import closing, contextmanager
from functools import partial
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Generator, List, Optional, Tuple, TypeVar, Union
from unittest.mock import patch
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import aiohttp.client_exceptions
import fsspec
import huggingface_hub
import requests
from fsspec.core import strip_protocol, url_to_fs
from fsspec.utils import can_be_local
from huggingface_hub.utils import EntryNotFoundError, insecure_hashlib
from packaging import version

from .. import __version__, config
from ..download.download_config import DownloadConfig
from ..filesystems import COMPRESSION_FILESYSTEMS
from . import _tqdm, logging
from . import tqdm as hf_tqdm
from ._filelock import FileLock
from .extract import ExtractManager
from .track import TrackedIterableFromGenerator


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INCOMPLETE_SUFFIX = ".incomplete"

T = TypeVar("T", str, Path)


def init_hf_modules(hf_modules_cache: Optional[Union[Path, str]] = None) -> str:
    """
    Add hf_modules_cache to the python path.
    By default hf_modules_cache='~/.cache/huggingface/modules'.
    It can also be set with the environment variable HF_MODULES_CACHE.
    This is used to add modules such as `datasets_modules`
    """
    hf_modules_cache = hf_modules_cache if hf_modules_cache is not None else config.HF_MODULES_CACHE
    hf_modules_cache = str(hf_modules_cache)
    if hf_modules_cache not in sys.path:
        sys.path.append(hf_modules_cache)

        os.makedirs(hf_modules_cache, exist_ok=True)
        if not os.path.exists(os.path.join(hf_modules_cache, "__init__.py")):
            with open(os.path.join(hf_modules_cache, "__init__.py"), "w"):
                pass
    return hf_modules_cache


def is_remote_url(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme != "" and not os.path.ismount(urlparse(url_or_filename).scheme + ":/")


def is_local_path(url_or_filename: str) -> bool:
    # On unix the scheme of a local path is empty (for both absolute and relative),
    # while on windows the scheme is the drive name (ex: "c") for absolute paths.
    # for details on the windows behavior, see https://bugs.python.org/issue42215
    return urlparse(url_or_filename).scheme == "" or os.path.ismount(urlparse(url_or_filename).scheme + ":/")


def is_relative_path(url_or_filename: str) -> bool:
    return urlparse(url_or_filename).scheme == "" and not os.path.isabs(url_or_filename)


def relative_to_absolute_path(path: T) -> T:
    """Convert relative path to absolute path."""
    abs_path_str = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    return Path(abs_path_str) if isinstance(path, Path) else abs_path_str


def hf_bucket_url(identifier: str, filename: str, use_cdn=False, dataset=True) -> str:
    if dataset:
        endpoint = config.CLOUDFRONT_DATASETS_DISTRIB_PREFIX if use_cdn else config.S3_DATASETS_BUCKET_PREFIX
    else:
        endpoint = config.CLOUDFRONT_METRICS_DISTRIB_PREFIX if use_cdn else config.S3_METRICS_BUCKET_PREFIX
    return "/".join((endpoint, identifier, filename))


def head_hf_s3(
    identifier: str, filename: str, use_cdn=False, dataset=True, max_retries=0
) -> Union[requests.Response, Exception]:
    return http_head(
        hf_bucket_url(identifier=identifier, filename=filename, use_cdn=use_cdn, dataset=dataset),
        max_retries=max_retries,
    )


def hf_github_url(path: str, name: str, dataset=True, revision: Optional[str] = None) -> str:
    default_revision = "main" if version.parse(__version__).is_devrelease else __version__
    revision = revision or default_revision
    if dataset:
        return config.REPO_DATASETS_URL.format(revision=revision, path=path, name=name)
    else:
        return config.REPO_METRICS_URL.format(revision=revision, path=path, name=name)


def url_or_path_join(base_name: str, *pathnames: str) -> str:
    if is_remote_url(base_name):
        return posixpath.join(base_name, *(str(pathname).replace(os.sep, "/").lstrip("/") for pathname in pathnames))
    else:
        return Path(base_name, *pathnames).as_posix()


def url_or_path_parent(url_or_path: str) -> str:
    if is_remote_url(url_or_path):
        return url_or_path[: url_or_path.rindex("/")]
    else:
        return os.path.dirname(url_or_path)


def hash_url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".py"):
        filename += ".py"

    return filename


def cached_path(
    url_or_filename,
    download_config=None,
    **download_kwargs,
) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
        ValueError: if it couldn't parse the url or filename correctly
        requests.exceptions.ConnectionError: in case of internet connection issue
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)

    cache_dir = download_config.cache_dir or config.DOWNLOADED_DATASETS_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    # Convert fsspec URL in the format "file://local/path" to "local/path"
    if can_be_local(url_or_filename):
        url_or_filename = strip_protocol(url_or_filename)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        url_or_filename, storage_options = _prepare_path_and_storage_options(
            url_or_filename, download_config=download_config
        )
        # Pass HTTP storage_options to get_from_cache only if passed HTTP download_config.storage_options
        if (
            storage_options
            and storage_options.keys() < {"http", "https"}
            and not (download_config.storage_options and download_config.storage_options.keys() < {"http", "https"})
        ):
            storage_options = {}
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=download_config.force_download,
            proxies=download_config.proxies,
            resume_download=download_config.resume_download,
            user_agent=download_config.user_agent,
            local_files_only=download_config.local_files_only,
            use_etag=download_config.use_etag,
            max_retries=download_config.max_retries,
            token=download_config.token,
            ignore_url_params=download_config.ignore_url_params,
            storage_options=storage_options,
            download_desc=download_config.download_desc,
            disable_tqdm=download_config.disable_tqdm,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif is_local_path(url_or_filename):
        # File, but it doesn't exist.
        raise FileNotFoundError(f"Local file {url_or_filename} doesn't exist")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if output_path is None:
        return output_path

    if download_config.extract_compressed_file:
        if download_config.extract_on_the_fly:
            # Add a compression prefix to the compressed file so that it can be extracted
            # as it's being read using xopen.
            protocol = _get_extraction_protocol(output_path, download_config=download_config)
            extension = _get_path_extension(url_or_filename.split("::")[0])
            if (
                protocol
                and extension not in ["tgz", "tar"]
                and not url_or_filename.split("::")[0].endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
            ):
                output_path = relative_to_absolute_path(output_path)
                if protocol in SINGLE_FILE_COMPRESSION_PROTOCOLS:
                    # there is one single file which is the uncompressed file
                    inner_file = os.path.basename(output_path)
                    inner_file = inner_file[: inner_file.rindex(".")] if "." in inner_file else inner_file
                    output_path = f"{protocol}://{inner_file}::{output_path}"
                else:
                    output_path = f"{protocol}://::{output_path}"
                return output_path

        # Eager extraction
        output_path = ExtractManager(cache_dir=download_config.cache_dir).extract(
            output_path, force_extract=download_config.force_extract
        )
    return relative_to_absolute_path(output_path)


def get_datasets_user_agent(user_agent: Optional[Union[str, dict]] = None) -> str:
    ua = f"datasets/{__version__}"
    ua += f"; python/{config.PY_VERSION}"
    ua += f"; huggingface_hub/{huggingface_hub.__version__}"
    ua += f"; pyarrow/{config.PYARROW_VERSION}"
    if config.TORCH_AVAILABLE:
        ua += f"; torch/{config.TORCH_VERSION}"
    if config.TF_AVAILABLE:
        ua += f"; tensorflow/{config.TF_VERSION}"
    if config.JAX_AVAILABLE:
        ua += f"; jax/{config.JAX_VERSION}"
    if config.BEAM_AVAILABLE:
        ua += f"; apache_beam/{config.BEAM_VERSION}"
    if isinstance(user_agent, dict):
        ua += f"; {'; '.join(f'{k}/{v}' for k, v in user_agent.items())}"
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def get_authentication_headers_for_url(
    url: str, token: Optional[Union[str, bool]] = None, use_auth_token: Optional[Union[str, bool]] = "deprecated"
) -> dict:
    """Handle the HF authentication"""
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if url.startswith(config.HF_ENDPOINT):
        return huggingface_hub.utils.build_hf_headers(
            token=token, library_name="datasets", library_version=__version__
        )
    else:
        return {}


class OfflineModeIsEnabled(ConnectionError):
    pass


def _raise_if_offline_mode_is_enabled(msg: Optional[str] = None):
    """Raise an OfflineModeIsEnabled error (subclass of ConnectionError) if HF_HUB_OFFLINE is True."""
    if config.HF_HUB_OFFLINE:
        raise OfflineModeIsEnabled(
            "Offline mode is enabled." if msg is None else "Offline mode is enabled. " + str(msg)
        )


def _request_with_retry(
    method: str,
    url: str,
    max_retries: int = 0,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
    timeout: float = 10.0,
    **params,
) -> requests.Response:
    """Wrapper around requests to retry in case it fails with a ConnectTimeout, with exponential backoff.

    Note that if the environment variable HF_HUB_OFFLINE is set to 1, then a OfflineModeIsEnabled error is raised.

    Args:
        method (str): HTTP method, such as 'GET' or 'HEAD'.
        url (str): The URL of the resource to fetch.
        max_retries (int): Maximum number of retries, defaults to 0 (no retries).
        base_wait_time (float): Duration (in seconds) to wait before retrying the first time. Wait time between
            retries then grows exponentially, capped by max_wait_time.
        max_wait_time (float): Maximum amount of time between two retries, in seconds.
        **params (additional keyword arguments): Params to pass to :obj:`requests.request`.
    """
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    tries, success = 0, False
    while not success:
        tries += 1
        try:
            response = requests.request(method=method.upper(), url=url, timeout=timeout, **params)
            success = True
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError) as err:
            if tries > max_retries:
                raise err
            else:
                logger.info(f"{method} request to {url} timed out, retrying... [{tries/max_retries}]")
                sleep_time = min(max_wait_time, base_wait_time * 2 ** (tries - 1))  # Exponential backoff
                time.sleep(sleep_time)
    return response


def fsspec_head(url, storage_options=None):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    return fs.info(path)


def stack_multiprocessing_download_progress_bars():
    # Stack downloads progress bars automatically using HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS=1
    # We use environment variables since the download may happen in a subprocess
    return patch.dict(os.environ, {"HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS": "1"})


class TqdmCallback(fsspec.callbacks.TqdmCallback):
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        if config.FSSPEC_VERSION < version.parse("2024.2.0"):
            super().__init__(tqdm_kwargs, *args, **kwargs)
            self._tqdm = _tqdm  # replace tqdm module by datasets.utils.tqdm module
        else:
            kwargs["tqdm_cls"] = _tqdm.tqdm
            super().__init__(tqdm_kwargs, *args, **kwargs)


def fsspec_get(url, temp_file, storage_options=None, desc=None, disable_tqdm=False):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    fs, path = url_to_fs(url, **(storage_options or {}))
    callback = TqdmCallback(
        tqdm_kwargs={
            "desc": desc or "Downloading",
            "unit": "B",
            "unit_scale": True,
            "position": multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
            if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
            and multiprocessing.current_process()._identity
            else None,
            "disable": disable_tqdm,
        }
    )
    fs.get_file(path, temp_file.name, callback=callback)


def ftp_head(url, timeout=10.0):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    try:
        with closing(urllib.request.urlopen(url, timeout=timeout)) as r:
            r.read(1)
    except Exception:
        return False
    return True


def ftp_get(url, temp_file, timeout=10.0):
    _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
    try:
        logger.info(f"Getting through FTP {url} into {temp_file.name}")
        with closing(urllib.request.urlopen(url, timeout=timeout)) as r:
            shutil.copyfileobj(r, temp_file)
    except urllib.error.URLError as e:
        raise ConnectionError(e) from None


def http_get(
    url,
    temp_file,
    proxies=None,
    resume_size=0,
    headers=None,
    cookies=None,
    timeout=100.0,
    max_retries=0,
    desc=None,
    disable_tqdm=False,
) -> Optional[requests.Response]:
    headers = dict(headers) if headers is not None else {}
    headers["user-agent"] = get_datasets_user_agent(user_agent=headers.get("user-agent"))
    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size:d}-"
    response = _request_with_retry(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        max_retries=max_retries,
        timeout=timeout,
    )
    if temp_file is None:
        return response
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    with hf_tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc=desc or "Downloading",
        position=multiprocessing.current_process()._identity[-1]  # contains the ranks of subprocesses
        if os.environ.get("HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS") == "1"
        and multiprocessing.current_process()._identity
        else None,
        disable=disable_tqdm,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            progress.update(len(chunk))
            temp_file.write(chunk)


def http_head(
    url, proxies=None, headers=None, cookies=None, allow_redirects=True, timeout=10.0, max_retries=0
) -> requests.Response:
    headers = copy.deepcopy(headers) or {}
    headers["user-agent"] = get_datasets_user_agent(user_agent=headers.get("user-agent"))
    response = _request_with_retry(
        method="HEAD",
        url=url,
        proxies=proxies,
        headers=headers,
        cookies=cookies,
        allow_redirects=allow_redirects,
        timeout=timeout,
        max_retries=max_retries,
    )
    return response


def request_etag(
    url: str, token: Optional[Union[str, bool]] = None, use_auth_token: Optional[Union[str, bool]] = "deprecated"
) -> Optional[str]:
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if urlparse(url).scheme not in ("http", "https"):
        return None
    headers = get_authentication_headers_for_url(url, token=token)
    response = http_head(url, headers=headers, max_retries=3)
    response.raise_for_status()
    etag = response.headers.get("ETag") if response.ok else None
    return etag


def get_from_cache(
    url,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=100,
    resume_download=False,
    user_agent=None,
    local_files_only=False,
    use_etag=True,
    max_retries=0,
    token=None,
    use_auth_token="deprecated",
    ignore_url_params=False,
    storage_options=None,
    download_desc=None,
    disable_tqdm=False,
) -> str:
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.

    Return:
        Local path (string)

    Raises:
        FileNotFoundError: in case of non-recoverable file
            (non-existent or no cache on disk)
        ConnectionError: in case of unreachable url
            and no cache on disk
    """
    if storage_options is None:
        storage_options = {}
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'token={use_auth_token}' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if cache_dir is None:
        cache_dir = config.HF_DATASETS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    if ignore_url_params:
        # strip all query parameters and #fragments from the URL
        cached_url = urljoin(url, urlparse(url).path)
    else:
        cached_url = url  # additional parameters may be added to the given URL

    connected = False
    response = None
    cookies = None
    etag = None
    head_error = None
    scheme = None

    # Try a first time to file the file on the local file system without eTag (None)
    # if we don't ask for 'force_download' then we spare a request
    filename = hash_url_to_filename(cached_url, etag=None)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download and not use_etag:
        return cache_path

    # Prepare headers for authentication
    headers = get_authentication_headers_for_url(url, token=token)
    if user_agent is not None:
        headers["user-agent"] = user_agent

    # We don't have the file locally or we need an eTag
    if not local_files_only:
        scheme = urlparse(url).scheme
        if scheme == "ftp":
            connected = ftp_head(url)
        elif scheme not in {"http", "https"} or storage_options.get(scheme):
            response = fsspec_head(url, storage_options=storage_options)
            # s3fs uses "ETag", gcsfs uses "etag"
            etag = (response.get("ETag", None) or response.get("etag", None)) if use_etag else None
            connected = True
        else:
            try:
                response = http_head(
                    url,
                    allow_redirects=True,
                    proxies=proxies,
                    timeout=etag_timeout,
                    max_retries=max_retries,
                    headers=headers,
                )
                if response.status_code == 200:  # ok
                    etag = response.headers.get("ETag") if use_etag else None
                    for k, v in response.cookies.items():
                        # In some edge cases, we need to get a confirmation token
                        if k.startswith("download_warning") and "drive.google.com" in url:
                            url += "&confirm=" + v
                            cookies = response.cookies
                    connected = True
                    # Fix Google Drive URL to avoid Virus scan warning
                    if "drive.google.com" in url and "confirm=" not in url:
                        url += "&confirm=t"
                # In some edge cases, head request returns 400 but the connection is actually ok
                elif (
                    (response.status_code == 400 and "firebasestorage.googleapis.com" in url)
                    or (response.status_code == 405 and "drive.google.com" in url)
                    or (
                        response.status_code == 403
                        and (
                            re.match(r"^https?://github.com/.*?/.*?/releases/download/.*?/.*?$", url)
                            or re.match(r"^https://.*?s3.*?amazonaws.com/.*?$", response.url)
                        )
                    )
                    or (response.status_code == 403 and "ndownloader.figstatic.com" in url)
                ):
                    connected = True
                    logger.info(f"Couldn't get ETag version for url {url}")
                elif response.status_code == 401 and config.HF_ENDPOINT in url and token is None:
                    raise ConnectionError(
                        f"Unauthorized for URL {url}. Please use the parameter `token=True` after logging in with `huggingface-cli login`"
                    )
            except (OSError, requests.exceptions.Timeout) as e:
                # not connected
                head_error = e
                pass

    # connected == False = we don't have a connection, or url doesn't exist, or is otherwise inaccessible.
    # try to get the last downloaded one
    if not connected:
        if os.path.exists(cache_path) and not force_download:
            return cache_path
        if local_files_only:
            raise FileNotFoundError(
                f"Cannot find the requested files in the cached path at {cache_path} and outgoing traffic has been"
                " disabled. To enable file online look-ups, set 'local_files_only' to False."
            )
        elif response is not None and response.status_code == 404:
            raise FileNotFoundError(f"Couldn't find file at {url}")
        _raise_if_offline_mode_is_enabled(f"Tried to reach {url}")
        if head_error is not None:
            raise ConnectionError(f"Couldn't reach {url} ({repr(head_error)})")
        elif response is not None:
            raise ConnectionError(f"Couldn't reach {url} (error {response.status_code})")
        else:
            raise ConnectionError(f"Couldn't reach {url}")

    # Try a second time
    filename = hash_url_to_filename(cached_url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # From now on, connected is True.
    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        # Retry in case previously locked processes just enter after the precedent process releases the lock
        if os.path.exists(cache_path) and not force_download:
            return cache_path

        incomplete_path = cache_path + ".incomplete"

        @contextmanager
        def temp_file_manager(mode="w+b"):
            with open(incomplete_path, mode) as f:
                yield f

        resume_size = 0
        if resume_download:
            temp_file_manager = partial(temp_file_manager, mode="a+b")
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size

        # Download to temporary file, then copy to cache path once finished.
        # Otherwise, you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            # GET file object
            if scheme == "ftp":
                ftp_get(url, temp_file)
            elif scheme not in {"http", "https"} or storage_options.get(scheme):
                fsspec_get(
                    url, temp_file, storage_options=storage_options, desc=download_desc, disable_tqdm=disable_tqdm
                )
            else:
                http_get(
                    url,
                    temp_file=temp_file,
                    proxies=proxies,
                    resume_size=resume_size,
                    headers=headers,
                    cookies=cookies,
                    max_retries=max_retries,
                    desc=download_desc,
                    disable_tqdm=disable_tqdm,
                )

        logger.info(f"storing {url} in cache at {cache_path}")
        shutil.move(temp_file.name, cache_path)
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + "\n\n" + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "\n\n" + "".join(docstr)
        return fn

    return docstring_decorator


def estimate_dataset_size(paths):
    return sum(path.stat().st_size for path in paths)


def readline(f: io.RawIOBase):
    # From: https://github.com/python/cpython/blob/d27e2f4d118e7a9909b6a3e5da06c5ff95806a85/Lib/_pyio.py#L525
    res = bytearray()
    while True:
        b = f.read(1)
        if not b:
            break
        res += b
        if res.endswith(b"\n"):
            break
    return bytes(res)


#######################
# Streaming utilities #
#######################

BASE_KNOWN_EXTENSIONS = [
    "txt",
    "csv",
    "json",
    "jsonl",
    "tsv",
    "conll",
    "conllu",
    "orig",
    "parquet",
    "pkl",
    "pickle",
    "rel",
    "xml",
]
COMPRESSION_EXTENSION_TO_PROTOCOL = {
    # single file compression
    **{fs_class.extension.lstrip("."): fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS},
    # archive compression
    "zip": "zip",
}
SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL = {
    fs_class.extension.lstrip("."): fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS
}
SINGLE_FILE_COMPRESSION_PROTOCOLS = {fs_class.protocol for fs_class in COMPRESSION_FILESYSTEMS}
SINGLE_SLASH_AFTER_PROTOCOL_PATTERN = re.compile(r"(?<!:):/")


MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL = {
    bytes.fromhex("504B0304"): "zip",
    bytes.fromhex("504B0506"): "zip",  # empty archive
    bytes.fromhex("504B0708"): "zip",  # spanned archive
    bytes.fromhex("425A68"): "bz2",
    bytes.fromhex("1F8B"): "gzip",
    bytes.fromhex("FD377A585A00"): "xz",
    bytes.fromhex("04224D18"): "lz4",
    bytes.fromhex("28B52FFD"): "zstd",
}
MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL = {
    b"Rar!": "rar",
}
MAGIC_NUMBER_MAX_LENGTH = max(
    len(magic_number)
    for magic_number in chain(MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL, MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL)
)


class NonStreamableDatasetError(Exception):
    pass


def _get_path_extension(path: str) -> str:
    # Get extension: https://foo.bar/train.json.gz -> gz
    extension = path.split(".")[-1]
    # Remove query params ("dl=1", "raw=true"): gz?dl=1 -> gz
    # Remove shards infos (".txt_1", ".txt-00000-of-00100"): txt_1 -> txt
    for symb in "?-_":
        extension = extension.split(symb)[0]
    return extension


def _get_extraction_protocol_with_magic_number(f) -> Optional[str]:
    """read the magic number from a file-like object and return the compression protocol"""
    # Check if the file object is seekable even before reading the magic number (to avoid https://bugs.python.org/issue26440)
    try:
        f.seek(0)
    except (AttributeError, io.UnsupportedOperation):
        return None
    magic_number = f.read(MAGIC_NUMBER_MAX_LENGTH)
    f.seek(0)
    for i in range(MAGIC_NUMBER_MAX_LENGTH):
        compression = MAGIC_NUMBER_TO_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            return compression
        compression = MAGIC_NUMBER_TO_UNSUPPORTED_COMPRESSION_PROTOCOL.get(magic_number[: MAGIC_NUMBER_MAX_LENGTH - i])
        if compression is not None:
            raise NotImplementedError(f"Compression protocol '{compression}' not implemented.")


def _get_extraction_protocol(urlpath: str, download_config: Optional[DownloadConfig] = None) -> Optional[str]:
    # get inner file: zip://train-00000.json.gz::https://foo.bar/data.zip -> zip://train-00000.json.gz
    urlpath = str(urlpath)
    path = urlpath.split("::")[0]
    extension = _get_path_extension(path)
    if (
        extension in BASE_KNOWN_EXTENSIONS
        or extension in ["tgz", "tar"]
        or path.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    ):
        return None
    elif extension in COMPRESSION_EXTENSION_TO_PROTOCOL:
        return COMPRESSION_EXTENSION_TO_PROTOCOL[extension]
    urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
    try:
        with fsspec.open(urlpath, **(storage_options or {})) as f:
            return _get_extraction_protocol_with_magic_number(f)
    except FileNotFoundError:
        if urlpath.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(
                urlpath + "\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`."
            ) from None
        else:
            raise


def xjoin(a, *p):
    """
    This function extends os.path.join to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xjoin function allows you to apply the join on the first path of the chain.

    Example::

        >>> xjoin("zip://folder1::https://host.com/archive.zip", "file.txt")
        zip://folder1/file.txt::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.join(a, *p)
    else:
        a = posixpath.join(a, *p)
        return "::".join([a] + b)


def xdirname(a):
    """
    This function extends os.path.dirname to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xdirname function allows you to apply the dirname on the first path of the chain.

    Example::

        >>> xdirname("zip://folder1/file.txt::https://host.com/archive.zip")
        zip://folder1::https://host.com/archive.zip
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        a = os.path.dirname(Path(a).as_posix())
    else:
        a = posixpath.dirname(a)
    # if we end up at the root of the protocol, we get for example a = 'http:'
    # so we have to fix it by adding the '//' that was removed:
    if a.endswith(":"):
        a += "//"
    return "::".join([a] + b)


def xexists(urlpath: str, download_config: Optional[DownloadConfig] = None):
    """Extend `os.path.exists` function to support both local and remote files.

    Args:
        urlpath (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """

    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return os.path.exists(main_hop)
    else:
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        return fs.exists(main_hop)


def xbasename(a):
    """
    This function extends os.path.basename to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xbasename function allows you to apply the basename on the first path of the chain.

    Example::

        >>> xbasename("zip://folder1/file.txt::https://host.com/archive.zip")
        file.txt
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.basename(Path(a).as_posix())
    else:
        return posixpath.basename(a)


def xsplit(a):
    """
    This function extends os.path.split to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplit function allows you to apply the xsplit on the first path of the chain.

    Example::

        >>> xsplit("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1::https://host.com/archive.zip', 'file.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.split(Path(a).as_posix())
    else:
        a, tail = posixpath.split(a)
        return "::".join([a + "//" if a.endswith(":") else a] + b), tail


def xsplitext(a):
    """
    This function extends os.path.splitext to support the "::" hop separator. It supports both paths and urls.

    A shorthand, particularly useful where you have multiple hops, is to “chain” the URLs with the special separator "::".
    This is used to access files inside a zip file over http for example.

    Let's say you have a zip file at https://host.com/archive.zip, and you want to access the file inside the zip file at /folder1/file.txt.
    Then you can just chain the url this way:

        zip://folder1/file.txt::https://host.com/archive.zip

    The xsplitext function allows you to apply the splitext on the first path of the chain.

    Example::

        >>> xsplitext("zip://folder1/file.txt::https://host.com/archive.zip")
        ('zip://folder1/file::https://host.com/archive.zip', '.txt')
    """
    a, *b = str(a).split("::")
    if is_local_path(a):
        return os.path.splitext(Path(a).as_posix())
    else:
        a, ext = posixpath.splitext(a)
        return "::".join([a] + b), ext


def xisfile(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isfile` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isfile(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        return fs.isfile(main_hop)


def xgetsize(path, download_config: Optional[DownloadConfig] = None) -> int:
    """Extend `os.path.getsize` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `int`: optional
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.getsize(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        try:
            size = fs.size(main_hop)
        except EntryNotFoundError:
            raise FileNotFoundError(f"No such file: {path}")
        if size is None:
            # use xopen instead of fs.open to make data fetching more robust
            with xopen(path, download_config=download_config) as f:
                size = len(f.read())
        return size


def xisdir(path, download_config: Optional[DownloadConfig] = None) -> bool:
    """Extend `os.path.isdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `bool`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.isdir(path)
    else:
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if not inner_path.strip("/"):
            return True
        return fs.isdir(inner_path)


def xrelpath(path, start=None):
    """Extend `os.path.relpath` function to support remote files.

    Args:
        path (`str`): URL path.
        start (`str`): Start URL directory path.

    Returns:
        `str`
    """
    main_hop, *rest_hops = str(path).split("::")
    if is_local_path(main_hop):
        return os.path.relpath(main_hop, start=start) if start else os.path.relpath(main_hop)
    else:
        return posixpath.relpath(main_hop, start=str(start).split("::")[0]) if start else os.path.relpath(main_hop)


def _add_retries_to_file_obj_read_method(file_obj):
    read = file_obj.read
    max_retries = config.STREAMING_READ_MAX_RETRIES

    def read_with_retries(*args, **kwargs):
        disconnect_err = None
        for retry in range(1, max_retries + 1):
            try:
                out = read(*args, **kwargs)
                break
            except (
                aiohttp.client_exceptions.ClientError,
                asyncio.TimeoutError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as err:
                disconnect_err = err
                logger.warning(
                    f"Got disconnected from remote data host. Retrying in {config.STREAMING_READ_RETRY_INTERVAL}sec [{retry}/{max_retries}]"
                )
                time.sleep(config.STREAMING_READ_RETRY_INTERVAL)
        else:
            raise ConnectionError("Server Disconnected") from disconnect_err
        return out

    try:
        file_obj.read = read_with_retries
    except AttributeError:  # read-only attribute
        orig_file_obj = file_obj
        file_obj = io.RawIOBase()
        file_obj.read = read_with_retries
        file_obj.__getattr__ = lambda _, attr: getattr(orig_file_obj, attr)
    return file_obj


def _prepare_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    prepared_urlpath = []
    prepared_storage_options = {}
    for hop in urlpath.split("::"):
        hop, storage_options = _prepare_single_hop_path_and_storage_options(hop, download_config=download_config)
        prepared_urlpath.append(hop)
        prepared_storage_options.update(storage_options)
    return "::".join(prepared_urlpath), storage_options


def _prepare_single_hop_path_and_storage_options(
    urlpath: str, download_config: Optional[DownloadConfig] = None
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Prepare the URL and the kwargs that must be passed to the HttpFileSystem or to requests.get/head

    In particular it resolves google drive URLs
    It also adds the authentication headers for the Hugging Face Hub, for both https:// and hf:// paths.

    Storage options are formatted in the form {protocol: storage_options_for_protocol}
    """
    token = None if download_config is None else download_config.token
    if urlpath.startswith(config.HF_ENDPOINT) and "/resolve/" in urlpath:
        urlpath = "hf://" + urlpath[len(config.HF_ENDPOINT) + 1 :].replace("/resolve/", "@", 1)
    protocol = urlpath.split("://")[0] if "://" in urlpath else "file"
    if download_config is not None and protocol in download_config.storage_options:
        storage_options = download_config.storage_options[protocol].copy()
    elif download_config is not None and protocol not in download_config.storage_options:
        storage_options = {
            option_name: option_value
            for option_name, option_value in download_config.storage_options.items()
            if option_name not in fsspec.available_protocols()
        }
    else:
        storage_options = {}
    if protocol in {"http", "https"}:
        client_kwargs = storage_options.pop("client_kwargs", {})
        storage_options["client_kwargs"] = {"trust_env": True, **client_kwargs}  # Enable reading proxy env variables
        if "drive.google.com" in urlpath:
            response = http_head(urlpath)
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    urlpath += "&confirm=" + v
                    cookies = response.cookies
                    storage_options = {"cookies": cookies, **storage_options}
            # Fix Google Drive URL to avoid Virus scan warning
            if "confirm=" not in urlpath:
                urlpath += "&confirm=t"
        if urlpath.startswith("https://raw.githubusercontent.com/"):
            # Workaround for served data with gzip content-encoding: https://github.com/fsspec/filesystem_spec/issues/389
            headers = storage_options.pop("headers", {})
            storage_options["headers"] = {"Accept-Encoding": "identity", **headers}
    elif protocol == "hf":
        storage_options = {
            "token": token,
            "endpoint": config.HF_ENDPOINT,
            **storage_options,
        }
        # streaming with block_size=0 is only implemented in 0.21 (see https://github.com/huggingface/huggingface_hub/pull/1967)
        if config.HF_HUB_VERSION < version.parse("0.21.0"):
            storage_options["block_size"] = "default"
    if storage_options:
        storage_options = {protocol: storage_options}
    return urlpath, storage_options


def xopen(file: str, mode="r", *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `open` function to support remote files using `fsspec`.

    It also has a retry mechanism in case connection fails.
    The `args` and `kwargs` are passed to `fsspec.open`, except `token` which is used for queries to private repos on huggingface.co

    Args:
        file (`str`): Path name of the file to be opened.
        mode (`str`, *optional*, default "r"): Mode in which the file is opened.
        *args: Arguments to be passed to `fsspec.open`.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Keyword arguments to be passed to `fsspec.open`.

    Returns:
        file object
    """
    # This works as well for `xopen(str(Path(...)))`
    file_str = _as_str(file)
    main_hop, *rest_hops = file_str.split("::")
    if is_local_path(main_hop):
        # ignore fsspec-specific kwargs
        kwargs.pop("block_size", None)
        return open(main_hop, mode, *args, **kwargs)
    # add headers and cookies for authentication on the HF Hub and for Google Drive
    file, storage_options = _prepare_path_and_storage_options(file_str, download_config=download_config)
    kwargs = {**kwargs, **(storage_options or {})}
    try:
        file_obj = fsspec.open(file, mode=mode, *args, **kwargs).open()
    except ValueError as e:
        if str(e) == "Cannot seek streaming HTTP file":
            raise NonStreamableDatasetError(
                "Streaming is not possible for this dataset because data host server doesn't support HTTP range "
                "requests. You can still load this dataset in non-streaming mode by passing `streaming=False` (default)"
            ) from e
        else:
            raise
    except FileNotFoundError:
        if file.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(
                file + "\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`."
            ) from None
        else:
            raise
    file_obj = _add_retries_to_file_obj_read_method(file_obj)
    return file_obj


def xlistdir(path: str, download_config: Optional[DownloadConfig] = None) -> List[str]:
    """Extend `os.listdir` function to support remote files.

    Args:
        path (`str`): URL path.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(path).split("::")
    if is_local_path(main_hop):
        return os.listdir(path)
    else:
        # globbing inside a zip in a private repo requires authentication
        path, storage_options = _prepare_path_and_storage_options(path, download_config=download_config)
        main_hop, *rest_hops = path.split("::")
        fs, *_ = url_to_fs(path, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            raise FileNotFoundError(f"Directory doesn't exist: {path}")
        paths = fs.listdir(inner_path, detail=False)
        return [os.path.basename(path.rstrip("/")) for path in paths]


def xglob(urlpath, *, recursive=False, download_config: Optional[DownloadConfig] = None):
    """Extend `glob.glob` function to support remote files.

    Args:
        urlpath (`str`): URL path with shell-style wildcard patterns.
        recursive (`bool`, default `False`): Whether to match the "**" pattern recursively to zero or more
            directories or subdirectories.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `list` of `str`
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        return glob.glob(main_hop, recursive=recursive)
    else:
        # globbing inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[1]
        globbed_paths = fs.glob(inner_path)
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        return ["::".join([f"{protocol}://{globbed_path}"] + rest_hops) for globbed_path in globbed_paths]


def xwalk(urlpath, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `os.walk` function to support remote files.

    Args:
        urlpath (`str`): URL root path.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs: Additional keyword arguments forwarded to the underlying filesystem.


    Yields:
        `tuple`: 3-tuple (dirpath, dirnames, filenames).
    """
    main_hop, *rest_hops = _as_str(urlpath).split("::")
    if is_local_path(main_hop):
        yield from os.walk(main_hop, **kwargs)
    else:
        # walking inside a zip in a private repo requires authentication
        urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
        main_hop, *rest_hops = urlpath.split("::")
        fs, *_ = url_to_fs(urlpath, **storage_options)
        inner_path = main_hop.split("://")[-1]
        if inner_path.strip("/") and not fs.isdir(inner_path):
            return []
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[-1]
        for dirpath, dirnames, filenames in fs.walk(inner_path, **kwargs):
            yield "::".join([f"{protocol}://{dirpath}"] + rest_hops), dirnames, filenames


class xPath(type(Path())):
    """Extension of `pathlib.Path` to support both local paths and remote URLs."""

    def __str__(self):
        path_str = super().__str__()
        main_hop, *rest_hops = path_str.split("::")
        if is_local_path(main_hop):
            return main_hop
        path_as_posix = path_str.replace("\\", "/")
        path_as_posix = SINGLE_SLASH_AFTER_PROTOCOL_PATTERN.sub("://", path_as_posix)
        path_as_posix += "//" if path_as_posix.endswith(":") else ""  # Add slashes to root of the protocol
        return path_as_posix

    def exists(self, download_config: Optional[DownloadConfig] = None):
        """Extend `pathlib.Path.exists` method to support both local and remote files.

        Args:
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Returns:
            `bool`
        """
        return xexists(str(self), download_config=download_config)

    def glob(self, pattern, download_config: Optional[DownloadConfig] = None):
        """Glob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.
            download_config : mainly use token or storage_options to support different platforms and auth types.

        Yields:
            [`xPath`]
        """
        posix_path = self.as_posix()
        main_hop, *rest_hops = posix_path.split("::")
        if is_local_path(main_hop):
            yield from Path(main_hop).glob(pattern)
        else:
            # globbing inside a zip in a private repo requires authentication
            if rest_hops:
                urlpath = rest_hops[0]
                urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
                storage_options = {urlpath.split("://")[0]: storage_options}
                posix_path = "::".join([main_hop, urlpath, *rest_hops[1:]])
            else:
                storage_options = None
            fs, *_ = url_to_fs(xjoin(posix_path, pattern), **(storage_options or {}))
            globbed_paths = fs.glob(xjoin(main_hop, pattern))
            for globbed_path in globbed_paths:
                yield type(self)("::".join([f"{fs.protocol}://{globbed_path}"] + rest_hops))

    def rglob(self, pattern, **kwargs):
        """Rglob function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Args:
            pattern (`str`): Pattern that resulting paths must match.

        Yields:
            [`xPath`]
        """
        return self.glob("**/" + pattern, **kwargs)

    @property
    def parent(self) -> "xPath":
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            [`xPath`]
        """
        return type(self)(xdirname(self.as_posix()))

    @property
    def name(self) -> str:
        """Name function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).name

    @property
    def stem(self) -> str:
        """Stem function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).stem

    @property
    def suffix(self) -> str:
        """Suffix function for argument of type :obj:`~pathlib.Path` that supports both local paths end remote URLs.

        Returns:
            `str`
        """
        return PurePosixPath(self.as_posix().split("::")[0]).suffix

    def open(self, *args, **kwargs):
        """Extend :func:`xopen` to support argument of type :obj:`~pathlib.Path`.

        Args:
            **args: Arguments passed to :func:`fsspec.open`.
            **kwargs: Keyword arguments passed to :func:`fsspec.open`.

        Returns:
            `io.FileIO`: File-like object.
        """
        return xopen(str(self), *args, **kwargs)

    def joinpath(self, *p: Tuple[str, ...]) -> "xPath":
        """Extend :func:`xjoin` to support argument of type :obj:`~pathlib.Path`.

        Args:
            *p (`tuple` of `str`): Other path components.

        Returns:
            [`xPath`]
        """
        return type(self)(xjoin(self.as_posix(), *p))

    def __truediv__(self, p: str) -> "xPath":
        return self.joinpath(p)

    def with_suffix(self, suffix):
        main_hop, *rest_hops = str(self).split("::")
        if is_local_path(main_hop):
            return type(self)(str(super().with_suffix(suffix)))
        return type(self)("::".join([type(self)(PurePosixPath(main_hop).with_suffix(suffix)).as_posix()] + rest_hops))


def _as_str(path: Union[str, Path, xPath]):
    return str(path) if isinstance(path, xPath) else str(xPath(str(path)))


def xgzip_open(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    import gzip

    if hasattr(filepath_or_buffer, "read"):
        return gzip.open(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return gzip.open(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)


def xnumpy_load(filepath_or_buffer, *args, download_config: Optional[DownloadConfig] = None, **kwargs):
    import numpy as np

    if hasattr(filepath_or_buffer, "read"):
        return np.load(filepath_or_buffer, *args, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return np.load(xopen(filepath_or_buffer, "rb", download_config=download_config), *args, **kwargs)


def xpandas_read_csv(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pandas as pd

    if hasattr(filepath_or_buffer, "read"):
        return pd.read_csv(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        if kwargs.get("compression", "infer") == "infer":
            kwargs["compression"] = _get_extraction_protocol(filepath_or_buffer, download_config=download_config)
        return pd.read_csv(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)


def xpandas_read_excel(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pandas as pd

    if hasattr(filepath_or_buffer, "read"):
        try:
            return pd.read_excel(filepath_or_buffer, **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(BytesIO(filepath_or_buffer.read()), **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        try:
            return pd.read_excel(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)
        except ValueError:  # Cannot seek streaming HTTP file
            return pd.read_excel(
                BytesIO(xopen(filepath_or_buffer, "rb", download_config=download_config).read()), **kwargs
            )


def xpyarrow_parquet_read_table(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import pyarrow.parquet as pq

    if hasattr(filepath_or_buffer, "read"):
        return pq.read_table(filepath_or_buffer, **kwargs)
    else:
        filepath_or_buffer = str(filepath_or_buffer)
        return pq.read_table(xopen(filepath_or_buffer, mode="rb", download_config=download_config), **kwargs)


def xsio_loadmat(filepath_or_buffer, download_config: Optional[DownloadConfig] = None, **kwargs):
    import scipy.io as sio

    if hasattr(filepath_or_buffer, "read"):
        return sio.loadmat(filepath_or_buffer, **kwargs)
    else:
        return sio.loadmat(xopen(filepath_or_buffer, "rb", download_config=download_config), **kwargs)


def xet_parse(source, parser=None, download_config: Optional[DownloadConfig] = None):
    """Extend `xml.etree.ElementTree.parse` function to support remote files.

    Args:
        source: File path or file object.
        parser (`XMLParser`, *optional*, default `XMLParser`): Parser instance.
        download_config : mainly use token or storage_options to support different platforms and auth types.

    Returns:
        `xml.etree.ElementTree.Element`: Root element of the given source document.
    """
    if hasattr(source, "read"):
        return ET.parse(source, parser=parser)
    else:
        with xopen(source, "rb", download_config=download_config) as f:
            return ET.parse(f, parser=parser)


def xxml_dom_minidom_parse(filename_or_file, download_config: Optional[DownloadConfig] = None, **kwargs):
    """Extend `xml.dom.minidom.parse` function to support remote files.

    Args:
        filename_or_file (`str` or file): File path or file object.
        download_config : mainly use token or storage_options to support different platforms and auth types.
        **kwargs (optional): Additional keyword arguments passed to `xml.dom.minidom.parse`.

    Returns:
        :obj:`xml.dom.minidom.Document`: Parsed document.
    """
    if hasattr(filename_or_file, "read"):
        return xml.dom.minidom.parse(filename_or_file, **kwargs)
    else:
        with xopen(filename_or_file, "rb", download_config=download_config) as f:
            return xml.dom.minidom.parse(f, **kwargs)


class ArchiveIterable(TrackedIterableFromGenerator):
    """An iterable of (path, fileobj) from a TAR archive, used by `iter_archive`"""

    @staticmethod
    def _iter_tar(f):
        stream = tarfile.open(fileobj=f, mode="r|*")
        for tarinfo in stream:
            file_path = tarinfo.name
            if not tarinfo.isreg():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = stream.extractfile(tarinfo)
            yield file_path, file_obj
            stream.members = []
        del stream

    @staticmethod
    def _iter_zip(f):
        zipf = zipfile.ZipFile(f)
        for member in zipf.infolist():
            file_path = member.filename
            if member.is_dir():
                continue
            if file_path is None:
                continue
            if os.path.basename(file_path).startswith((".", "__")):
                # skipping hidden files
                continue
            file_obj = zipf.open(member)
            yield file_path, file_obj

    @classmethod
    def _iter_from_fileobj(cls, f) -> Generator[Tuple, None, None]:
        compression = _get_extraction_protocol_with_magic_number(f)
        if compression == "zip":
            yield from cls._iter_zip(f)
        else:
            yield from cls._iter_tar(f)

    @classmethod
    def _iter_from_urlpath(
        cls, urlpath: str, download_config: Optional[DownloadConfig] = None
    ) -> Generator[Tuple, None, None]:
        compression = _get_extraction_protocol(urlpath, download_config=download_config)
        # Set block_size=0 to get faster streaming
        # (e.g. for hf:// and https:// it uses streaming Requests file-like instances)
        with xopen(urlpath, "rb", download_config=download_config, block_size=0) as f:
            if compression == "zip":
                yield from cls._iter_zip(f)
            else:
                yield from cls._iter_tar(f)

    @classmethod
    def from_buf(cls, fileobj) -> "ArchiveIterable":
        return cls(cls._iter_from_fileobj, fileobj)

    @classmethod
    def from_urlpath(cls, urlpath_or_buf, download_config: Optional[DownloadConfig] = None) -> "ArchiveIterable":
        return cls(cls._iter_from_urlpath, urlpath_or_buf, download_config)


class FilesIterable(TrackedIterableFromGenerator):
    """An iterable of paths from a list of directories or files"""

    @classmethod
    def _iter_from_urlpaths(
        cls, urlpaths: Union[str, List[str]], download_config: Optional[DownloadConfig] = None
    ) -> Generator[str, None, None]:
        if not isinstance(urlpaths, list):
            urlpaths = [urlpaths]
        for urlpath in urlpaths:
            if xisfile(urlpath, download_config=download_config):
                yield urlpath
            elif xisdir(urlpath, download_config=download_config):
                for dirpath, dirnames, filenames in xwalk(urlpath, download_config=download_config):
                    # in-place modification to prune the search
                    dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith((".", "__"))])
                    if xbasename(dirpath).startswith((".", "__")):
                        # skipping hidden directories
                        continue
                    for filename in sorted(filenames):
                        if filename.startswith((".", "__")):
                            # skipping hidden files
                            continue
                        yield xjoin(dirpath, filename)
            else:
                raise FileNotFoundError(urlpath)

    @classmethod
    def from_urlpaths(cls, urlpaths, download_config: Optional[DownloadConfig] = None) -> "FilesIterable":
        return cls(cls._iter_from_urlpaths, urlpaths, download_config)
