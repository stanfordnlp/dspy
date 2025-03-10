import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import huggingface_hub
from fsspec.core import url_to_fs
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem
from packaging import version
from tqdm.contrib.concurrent import thread_map

from . import config
from .download import DownloadConfig
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import _prepare_path_and_storage_options, is_local_path, is_relative_path, xbasename, xjoin
from .utils.py_utils import glob_pattern_to_regex, string_to_dict


SingleOriginMetadata = Union[Tuple[str, str], Tuple[str], Tuple[()]]


SANITIZED_DEFAULT_SPLIT = str(Split.TRAIN)


logger = logging.get_logger(__name__)


class Url(str):
    pass


class EmptyDatasetError(FileNotFoundError):
    pass


SPLIT_PATTERN_SHARDED = "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"

SPLIT_KEYWORDS = {
    Split.TRAIN: ["train", "training"],
    Split.VALIDATION: ["validation", "valid", "dev", "val"],
    Split.TEST: ["test", "testing", "eval", "evaluation"],
}
NON_WORDS_CHARS = "-._ 0-9"
if config.FSSPEC_VERSION < version.parse("2023.9.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**",
        "{keyword}[{sep}]*/**",
        "**[{sep}/]{keyword}/**",
        "**[{sep}/]{keyword}[{sep}]*/**",
    ]
elif config.FSSPEC_VERSION < version.parse("2023.12.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/*[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**/*",
        "{keyword}[{sep}]*/**/*",
        "**/*[{sep}/]{keyword}/**/*",
        "**/*[{sep}/]{keyword}[{sep}]*/**/*",
    ]
else:
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/{keyword}[{sep}]*", "**/*[{sep}]{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "**/{keyword}/**",
        "**/{keyword}[{sep}]*/**",
        "**/*[{sep}]{keyword}/**",
        "**/*[{sep}]{keyword}[{sep}]*/**",
    ]

DEFAULT_SPLITS = [Split.TRAIN, Split.VALIDATION, Split.TEST]
DEFAULT_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}
DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}


DEFAULT_PATTERNS_ALL = {
    Split.TRAIN: ["**"],
}

ALL_SPLIT_PATTERNS = [SPLIT_PATTERN_SHARDED]
ALL_DEFAULT_PATTERNS = [
    DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_PATTERNS_ALL,
]
if config.FSSPEC_VERSION < version.parse("2023.9.0"):
    METADATA_PATTERNS = [
        "metadata.csv",
        "**/metadata.csv",
        "metadata.jsonl",
        "**/metadata.jsonl",
    ]  # metadata file for ImageFolder and AudioFolder
else:
    METADATA_PATTERNS = [
        "**/metadata.csv",
        "**/metadata.jsonl",
    ]  # metadata file for ImageFolder and AudioFolder
WILDCARD_CHARACTERS = "*[]"
FILES_TO_IGNORE = [
    "README.md",
    "config.json",
    "dataset_info.json",
    "dataset_infos.json",
    "dummy_data.zip",
    "dataset_dict.json",
]


def contains_wildcards(pattern: str) -> bool:
    return any(wilcard_character in pattern for wilcard_character in WILDCARD_CHARACTERS)


def sanitize_patterns(patterns: Union[Dict, List, str]) -> Dict[str, Union[List[str], "DataFilesList"]]:
    """
    Take the data_files patterns from the user, and format them into a dictionary.
    Each key is the name of the split, and each value is a list of data files patterns (paths or urls).
    The default split is "train".

    Returns:
        patterns: dictionary of split_name -> list of patterns
    """
    if isinstance(patterns, dict):
        return {str(key): value if isinstance(value, list) else [value] for key, value in patterns.items()}
    elif isinstance(patterns, str):
        return {SANITIZED_DEFAULT_SPLIT: [patterns]}
    elif isinstance(patterns, list):
        if any(isinstance(pattern, dict) for pattern in patterns):
            for pattern in patterns:
                if not (
                    isinstance(pattern, dict)
                    and len(pattern) == 2
                    and "split" in pattern
                    and isinstance(pattern.get("path"), (str, list))
                ):
                    raise ValueError(
                        f"Expected each split to have a 'path' key which can be a string or a list of strings, but got {pattern}"
                    )
            splits = [pattern["split"] for pattern in patterns]
            if len(set(splits)) != len(splits):
                raise ValueError(f"Some splits are duplicated in data_files: {splits}")
            return {
                str(pattern["split"]): pattern["path"] if isinstance(pattern["path"], list) else [pattern["path"]]
                for pattern in patterns
            }
        else:
            return {SANITIZED_DEFAULT_SPLIT: patterns}
    else:
        return sanitize_patterns(list(patterns))


def _is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionnally check if it's inside a special directory
    we ignore by default (if it starts with a double underscore).

    Users can still explicitly request a filepath inside such a directory if "__pycache__" is
    mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── __pycache__
            └── b.txt

    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "**")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "*/b.txt")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__pycache__/*")
    False
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__*/*")
    False
    """
    # We just need to check if every special directories from the path is present explicly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the parent path and the parent pattern have the same number of special directories.
    data_dirs_to_ignore_in_path = [part for part in PurePath(matched_rel_path).parent.parts if part.startswith("__")]
    data_dirs_to_ignore_in_pattern = [part for part in PurePath(pattern).parent.parts if part.startswith("__")]
    return len(data_dirs_to_ignore_in_path) != len(data_dirs_to_ignore_in_pattern)


def _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionnally check if it's a hidden file or if it's inside
    a hidden directory we ignore by default, i.e. if the file name or a parent directory name starts with a dot.

    Users can still explicitly request a filepath that is hidden or is inside a hidden directory
    if the hidden part is mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", ".*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── a.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".*/*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".hidden_dir/*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/.*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/.*")
    False
    """
    # We just need to check if every hidden part from the path is present explicly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the path and the pattern have the same number of hidden parts.
    hidden_directories_in_path = [
        part for part in PurePath(matched_rel_path).parts if part.startswith(".") and not set(part) == {"."}
    ]
    hidden_directories_in_pattern = [
        part for part in PurePath(pattern).parts if part.startswith(".") and not set(part) == {"."}
    ]
    return len(hidden_directories_in_path) != len(hidden_directories_in_pattern)


def _get_data_files_patterns(pattern_resolver: Callable[[str], List[str]]) -> Dict[str, List[str]]:
    """
    Get the default pattern from a directory or repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    # first check the split patterns like data/{split}-00000-of-00001.parquet
    for split_pattern in ALL_SPLIT_PATTERNS:
        pattern = split_pattern.replace("{split}", "*")
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: Set[str] = {
                string_to_dict(xbasename(p), glob_pattern_to_regex(xbasename(split_pattern)))["split"]
                for p in data_files
            }
            if any(not re.match(_split_re, split) for split in splits):
                raise ValueError(f"Split name should match '{_split_re}'' but got '{splits}'.")
            sorted_splits = [str(split) for split in DEFAULT_SPLITS if split in splits] + sorted(
                splits - set(DEFAULT_SPLITS)
            )
            return {split: [split_pattern.format(split=split)] for split in sorted_splits}
    # then check the default patterns based on train/valid/test splits
    for patterns_dict in ALL_DEFAULT_PATTERNS:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")


def _get_metadata_files_patterns(pattern_resolver: Callable[[str], List[str]]) -> List[str]:
    """
    Get the supported metadata patterns from a directory or repository.
    """
    non_empty_patterns = []
    for pattern in METADATA_PATTERNS:
        try:
            metadata_files = pattern_resolver(pattern)
            if len(metadata_files) > 0:
                non_empty_patterns.append(pattern)
        except FileNotFoundError:
            pass
    if non_empty_patterns:
        return non_empty_patterns
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")


def resolve_pattern(
    pattern: str,
    base_path: str,
    allowed_extensions: Optional[List[str]] = None,
    download_config: Optional[DownloadConfig] = None,
) -> List[str]:
    """
    Resolve the paths and URLs of the data files from the pattern passed by the user.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob. In fsspec>=2023.12.0 this is equivalent to
    Python's glob.glob, Path.glob, Path.match and fnmatch where ** is unsupported with a prefix/suffix
    other than a forward slash /.

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicilty mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"

    Example::

        >>> from datasets.data_files import resolve_pattern
        >>> base_path = "."
        >>> resolve_pattern("docs/**/*.py", base_path)
        [/Users/mariosasko/Desktop/projects/datasets/docs/source/_config.py']

    Args:
        pattern (str): Unix pattern or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
            Remote filesystems using fsspec are supported, e.g. with the hf:// protocol.
        base_path (str): Base path to use when resolving relative paths.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=[".csv", ".json", ".txt", ".parquet"]
        download_config ([`DownloadConfig`], *optional*): Specific download configuration parameters.
    Returns:
        List[str]: List of paths or URLs to the local or remote files that match the patterns.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ""
    pattern, storage_options = _prepare_path_and_storage_options(pattern, download_config=download_config)
    fs, fs_pattern = url_to_fs(pattern, **storage_options)
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    protocol_prefix = protocol + "://" if protocol != "file" else ""
    glob_kwargs = {}
    if protocol == "hf" and config.HF_HUB_VERSION >= version.parse("0.20.0"):
        # 10 times faster glob with detail=True (ignores costly info like lastCommit)
        glob_kwargs["expand_info"] = False
    matched_paths = [
        filepath if filepath.startswith(protocol_prefix) else protocol_prefix + filepath
        for filepath, info in fs.glob(pattern, detail=True, **glob_kwargs).items()
        if info["type"] == "file"
        and (xbasename(filepath) not in files_to_ignore)
        and not _is_inside_unrequested_special_dir(filepath, fs_pattern)
        and not _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(filepath, fs_pattern)
    ]  # ignore .ipynb and __pycache__, but keep /../
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any("." + suffix in allowed_extensions for suffix in xbasename(filepath).split(".")[1:])
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(
                f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: {invalid_matched_files}"
            )
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return out


def get_data_patterns(base_path: str, download_config: Optional[DownloadConfig] = None) -> Dict[str, List[str]]:
    """
    Get the default pattern from a directory testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    Some examples of supported patterns:

    Input:

        my_dataset_repository/
        ├── README.md
        └── dataset.csv

    Output:

        {'train': ['**']}

    Input:

        my_dataset_repository/
        ├── README.md
        ├── train.csv
        └── test.csv

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train.csv
            └── test.csv

        my_dataset_repository/
        ├── README.md
        ├── train_0.csv
        ├── train_1.csv
        ├── train_2.csv
        ├── train_3.csv
        ├── test_0.csv
        └── test_1.csv

    Output:

        {'train': ['**/train[-._ 0-9]*', '**/*[-._ 0-9]train[-._ 0-9]*', '**/training[-._ 0-9]*', '**/*[-._ 0-9]training[-._ 0-9]*'],
         'test': ['**/test[-._ 0-9]*', '**/*[-._ 0-9]test[-._ 0-9]*', '**/testing[-._ 0-9]*', '**/*[-._ 0-9]testing[-._ 0-9]*', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train/
            │   ├── shard_0.csv
            │   ├── shard_1.csv
            │   ├── shard_2.csv
            │   └── shard_3.csv
            └── test/
                ├── shard_0.csv
                └── shard_1.csv

    Output:

        {'train': ['**/train/**', '**/train[-._ 0-9]*/**', '**/*[-._ 0-9]train/**', '**/*[-._ 0-9]train[-._ 0-9]*/**', ...],
         'test': ['**/test/**', '**/test[-._ 0-9]*/**', '**/*[-._ 0-9]test/**', '**/*[-._ 0-9]test[-._ 0-9]*/**', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train-00000-of-00003.csv
            ├── train-00001-of-00003.csv
            ├── train-00002-of-00003.csv
            ├── test-00000-of-00001.csv
            ├── random-00000-of-00003.csv
            ├── random-00001-of-00003.csv
            └── random-00002-of-00003.csv

    Output:

        {'train': ['data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'test': ['data/test-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'random': ['data/random-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*']}

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(f"The directory at {base_path} doesn't contain any data files") from None


def get_metadata_patterns(
    base_path: str,
    download_config: Optional[DownloadConfig] = None,
) -> List[str]:
    """
    Get the supported metadata patterns from a local directory.
    """
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_metadata_files_patterns(resolver)
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory at {base_path} doesn't contain any metadata file") from None


def _get_single_origin_metadata(
    data_file: str,
    download_config: Optional[DownloadConfig] = None,
) -> SingleOriginMetadata:
    data_file, storage_options = _prepare_path_and_storage_options(data_file, download_config=download_config)
    fs, *_ = url_to_fs(data_file, **storage_options)
    if isinstance(fs, HfFileSystem):
        resolved_path = fs.resolve_path(data_file)
        return resolved_path.repo_id, resolved_path.revision
    elif isinstance(fs, HTTPFileSystem) and data_file.startswith(config.HF_ENDPOINT):
        hffs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
        data_file = "hf://" + data_file[len(config.HF_ENDPOINT) + 1 :].replace("/resolve/", "@", 1)
        resolved_path = hffs.resolve_path(data_file)
        return resolved_path.repo_id, resolved_path.revision
    info = fs.info(data_file)
    # s3fs uses "ETag", gcsfs uses "etag", and for local we simply check mtime
    for key in ["ETag", "etag", "mtime"]:
        if key in info:
            return (str(info[key]),)
    return ()


def _get_origin_metadata(
    data_files: List[str],
    download_config: Optional[DownloadConfig] = None,
    max_workers: Optional[int] = None,
) -> List[SingleOriginMetadata]:
    max_workers = max_workers if max_workers is not None else config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
    return thread_map(
        partial(_get_single_origin_metadata, download_config=download_config),
        data_files,
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
        desc="Resolving data files",
        # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
        disable=len(data_files) <= 16 or None,
    )


class DataFilesList(List[str]):
    """
    List of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns:
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, DataFilesList has an additional attribute ``origin_metadata``.
    It can store:
    - the last modified time of local files
    - ETag of remote files
    - commit sha of a dataset repository

    Thanks to this additional attribute, it is possible to hash the list
    and get a different hash if and only if at least one file changed.
    This is useful for caching Dataset objects that are obtained from a list of data files.
    """

    def __init__(self, data_files: List[str], origin_metadata: List[SingleOriginMetadata]) -> None:
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    def __add__(self, other: "DataFilesList") -> "DataFilesList":
        return DataFilesList([*self, *other], self.origin_metadata + other.origin_metadata)

    @classmethod
    def from_hf_repo(
        cls,
        patterns: List[str],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = f"hf://datasets/{dataset_info.id}@{dataset_info.sha}/{base_path or ''}".rstrip("/")
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: List[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_patterns(
        cls,
        patterns: List[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern in patterns:
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)

    def filter_extensions(self, extensions: List[str]) -> "DataFilesList":
        pattern = "|".join("\\" + ext for ext in extensions)
        pattern = re.compile(f".*({pattern})(\\..+)?$")
        return DataFilesList(
            [data_file for data_file in self if pattern.match(data_file)],
            origin_metadata=self.origin_metadata,
        )


class DataFilesDict(Dict[str, DataFilesList]):
    """
    Dict of split_name -> list of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns :
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, each list is a DataFilesList. It is possible to hash the dictionary
    and get a different hash if and only if at least one file changed.
    For more info, see [`DataFilesList`].

    This is useful for caching Dataset objects that are obtained from a list of data files.

    Changing the order of the keys of this dictionary also doesn't change its hash.
    """

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: Dict[str, Union[List[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_local_or_remote(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_hf_repo(
        cls,
        patterns: Dict[str, Union[List[str], DataFilesList]],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_hf_repo(
                    patterns_for_key,
                    dataset_info=dataset_info,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_patterns(
        cls,
        patterns: Dict[str, Union[List[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_patterns(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    def filter_extensions(self, extensions: List[str]) -> "DataFilesDict":
        out = type(self)()
        for key, data_files_list in self.items():
            out[key] = data_files_list.filter_extensions(extensions)
        return out


class DataFilesPatternsList(List[str]):
    """
    List of data files patterns (absolute local paths or URLs).
    For each pattern there should also be a list of allowed extensions
    to keep, or a None ot keep all the files for the pattern.
    """

    def __init__(
        self,
        patterns: List[str],
        allowed_extensions: List[Optional[List[str]]],
    ):
        super().__init__(patterns)
        self.allowed_extensions = allowed_extensions

    def __add__(self, other):
        return DataFilesList([*self, *other], self.allowed_extensions + other.allowed_extensions)

    @classmethod
    def from_patterns(
        cls, patterns: List[str], allowed_extensions: Optional[List[str]] = None
    ) -> "DataFilesPatternsList":
        return cls(patterns, [allowed_extensions] * len(patterns))

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern, allowed_extensions in zip(self, self.allowed_extensions):
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return DataFilesList(data_files, origin_metadata)

    def filter_extensions(self, extensions: List[str]) -> "DataFilesPatternsList":
        return DataFilesPatternsList(
            self, [allowed_extensions + extensions for allowed_extensions in self.allowed_extensions]
        )


class DataFilesPatternsDict(Dict[str, DataFilesPatternsList]):
    """
    Dict of split_name -> list of data files patterns (absolute local paths or URLs).
    """

    @classmethod
    def from_patterns(
        cls, patterns: Dict[str, List[str]], allowed_extensions: Optional[List[str]] = None
    ) -> "DataFilesPatternsDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesPatternsList)
                else DataFilesPatternsList.from_patterns(
                    patterns_for_key,
                    allowed_extensions=allowed_extensions,
                )
            )
        return out

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = DataFilesDict()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.resolve(base_path, download_config)
        return out

    def filter_extensions(self, extensions: List[str]) -> "DataFilesPatternsDict":
        out = type(self)()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.filter_extensions(extensions)
        return out
