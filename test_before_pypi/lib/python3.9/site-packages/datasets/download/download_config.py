import copy
import warnings
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .. import config


@dataclass
class DownloadConfig:
    """Configuration for our cached path manager.

    Attributes:
        cache_dir (`str` or `Path`, *optional*):
            Specify a cache directory to save the file to (overwrite the
            default cache dir).
        force_download (`bool`, defaults to `False`):
            If `True`, re-dowload the file even if it's already cached in
            the cache dir.
        resume_download (`bool`, defaults to `False`):
            If `True`, resume the download if an incompletely received file is
            found.
        proxies (`dict`, *optional*):
        user_agent (`str`, *optional*):
            Optional string or dict that will be appended to the user-agent on remote
            requests.
        extract_compressed_file (`bool`, defaults to `False`):
            If `True` and the path point to a zip or tar file,
            extract the compressed file in a folder along the archive.
        force_extract (`bool`, defaults to `False`):
            If `True` when `extract_compressed_file` is `True` and the archive
            was already extracted, re-extract the archive and override the folder where it was extracted.
        delete_extracted (`bool`, defaults to `False`):
            Whether to delete (or keep) the extracted files.
        extract_on_the_fly (`bool`, defaults to `False`):
            If `True`, extract compressed files while they are being read.
        use_etag (`bool`, defaults to `True`):
            Whether to use the ETag HTTP response header to validate the cached files.
        num_proc (`int`, *optional*):
            The number of processes to launch to download the files in parallel.
        max_retries (`int`, default to `1`):
            The number of times to retry an HTTP request if it fails.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token
            for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token
            for remote files on the Datasets Hub. If `True`, or not specified, will get token from `~/.huggingface`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>

        ignore_url_params (`bool`, defaults to `False`):
            Whether to strip all query parameters and fragments from
            the download URL before using it for caching the file.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        download_desc (`str`, *optional*):
            A description to be displayed alongside with the progress bar while downloading the files.
        disable_tqdm (`bool`, defaults to `False`):
            Whether to disable the individual files download progress bar
    """

    cache_dir: Optional[Union[str, Path]] = None
    force_download: bool = False
    resume_download: bool = False
    local_files_only: bool = False
    proxies: Optional[Dict] = None
    user_agent: Optional[str] = None
    extract_compressed_file: bool = False
    force_extract: bool = False
    delete_extracted: bool = False
    extract_on_the_fly: bool = False
    use_etag: bool = True
    num_proc: Optional[int] = None
    max_retries: int = 1
    token: Optional[Union[str, bool]] = None
    use_auth_token: InitVar[Optional[Union[str, bool]]] = "deprecated"
    ignore_url_params: bool = False
    storage_options: Dict[str, Any] = field(default_factory=dict)
    download_desc: Optional[str] = None
    disable_tqdm: bool = False

    def __post_init__(self, use_auth_token):
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'token={use_auth_token}' instead.",
                FutureWarning,
            )
            self.token = use_auth_token

    def copy(self) -> "DownloadConfig":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def __setattr__(self, name, value):
        if name == "token" and getattr(self, "storage_options", None) is not None:
            if "hf" not in self.storage_options:
                self.storage_options["hf"] = {"token": value, "endpoint": config.HF_ENDPOINT}
            elif getattr(self.storage_options["hf"], "token", None) is None:
                self.storage_options["hf"]["token"] = value
        super().__setattr__(name, value)
