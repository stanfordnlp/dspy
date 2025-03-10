# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Access datasets."""

import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import fsspec
import requests
import yaml
from fsspec.core import url_to_fs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError

from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
    DEFAULT_PATTERNS_ALL,
    DataFilesDict,
    DataFilesList,
    DataFilesPatternsDict,
    DataFilesPatternsList,
    EmptyDatasetError,
    get_data_patterns,
    get_metadata_patterns,
    sanitize_patterns,
)
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
    _EXTENSION_TO_MODULE,
    _MODULE_SUPPORTS_METADATA,
    _MODULE_TO_EXTENSIONS,
    _PACKAGED_DATASETS_MODULES,
    _hash_python_lines,
)
from .splits import Split
from .utils import _dataset_viewer
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
    OfflineModeIsEnabled,
    _raise_if_offline_mode_is_enabled,
    cached_path,
    head_hf_s3,
    hf_github_url,
    init_hf_modules,
    is_relative_path,
    relative_to_absolute_path,
    url_or_path_join,
)
from .utils.hub import hf_dataset_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports, lock_importable_file
from .utils.typing import PathLike
from .utils.version import Version


logger = get_logger(__name__)

ALL_ALLOWED_EXTENSIONS = list(_EXTENSION_TO_MODULE.keys()) + [".zip"]


def _raise_timeout_error(signum, frame):
    raise ValueError(
        "Loading this dataset requires you to execute custom code contained in the dataset repository on your local "
        "machine. Please set the option `trust_remote_code=True` to permit loading of this dataset."
    )


def resolve_trust_remote_code(trust_remote_code: Optional[bool], repo_id: str) -> bool:
    """
    Copied and adapted from Transformers
    https://github.com/huggingface/transformers/blob/2098d343cc4b4b9d2aea84b3cf1eb5a1e610deff/src/transformers/dynamic_module_utils.py#L589
    """
    trust_remote_code = trust_remote_code if trust_remote_code is not None else config.HF_DATASETS_TRUST_REMOTE_CODE
    if trust_remote_code is None:
        if config.TIME_OUT_REMOTE_CODE > 0:
            try:
                signal.signal(signal.SIGALRM, _raise_timeout_error)
                signal.alarm(config.TIME_OUT_REMOTE_CODE)
                while trust_remote_code is None:
                    answer = input(
                        f"The repository for {repo_id} contains custom code which must be executed to correctly "
                        f"load the dataset. You can inspect the repository content at https://hf.co/datasets/{repo_id}.\n"
                        f"You can avoid this prompt in future by passing the argument `trust_remote_code=True`.\n\n"
                        f"Do you wish to run the custom code? [y/N] "
                    )
                    if answer.lower() in ["yes", "y", "1"]:
                        trust_remote_code = True
                    elif answer.lower() in ["no", "n", "0", ""]:
                        trust_remote_code = False
                signal.alarm(0)
            except Exception:
                # OS which does not support signal.SIGALRM
                raise ValueError(
                    f"The repository for {repo_id} contains custom code which must be executed to correctly "
                    f"load the dataset. You can inspect the repository content at https://hf.co/datasets/{repo_id}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )
        else:
            # For the CI which might put the timeout at 0
            _raise_timeout_error(None, None)
    return trust_remote_code


def init_dynamic_modules(
    name: str = config.MODULE_NAME_FOR_DYNAMIC_MODULES, hf_modules_cache: Optional[Union[Path, str]] = None
):
    """
    Create a module with name `name` in which you can add dynamic modules
    such as metrics or datasets. The module can be imported using its name.
    The module is created in the HF_MODULE_CACHE directory by default (~/.cache/huggingface/modules) but it can
    be overridden by specifying a path to another directory in `hf_modules_cache`.
    """
    hf_modules_cache = init_hf_modules(hf_modules_cache)
    dynamic_modules_path = os.path.join(hf_modules_cache, name)
    os.makedirs(dynamic_modules_path, exist_ok=True)
    if not os.path.exists(os.path.join(dynamic_modules_path, "__init__.py")):
        with open(os.path.join(dynamic_modules_path, "__init__.py"), "w"):
            pass
    return dynamic_modules_path


def import_main_class(module_path, dataset=True) -> Optional[Union[Type[DatasetBuilder], Type[Metric]]]:
    """Import a module at module_path and return its main class:
    - a DatasetBuilder if dataset is True
    - a Metric if dataset is False
    """
    module = importlib.import_module(module_path)

    if dataset:
        main_cls_type = DatasetBuilder
    else:
        main_cls_type = Metric

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            obj_module = inspect.getmodule(obj)
            if obj_module is not None and module == obj_module:
                break

    return module_main_cls


class _InitializeConfiguredDatasetBuilder:
    """
    From https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
    See also ConfiguredDatasetBuilder.__reduce__
    When called with the param value as the only argument, returns an
    un-initialized instance of the parameterized class. Subsequent __setstate__
    will be called by pickle.
    """

    def __call__(self, builder_cls, metadata_configs, default_config_name, name):
        # make a simple object which has no complex __init__ (this one will do)
        obj = _InitializeConfiguredDatasetBuilder()
        obj.__class__ = configure_builder_class(
            builder_cls, metadata_configs, default_config_name=default_config_name, dataset_name=name
        )
        return obj


def configure_builder_class(
    builder_cls: Type[DatasetBuilder],
    builder_configs: List[BuilderConfig],
    default_config_name: Optional[str],
    dataset_name: str,
) -> Type[DatasetBuilder]:
    """
    Dynamically create a builder class with custom builder configs parsed from README.md file,
    i.e. set BUILDER_CONFIGS class variable of a builder class to custom configs list.
    """

    class ConfiguredDatasetBuilder(builder_cls):
        BUILDER_CONFIGS = builder_configs
        DEFAULT_CONFIG_NAME = default_config_name

        __module__ = builder_cls.__module__  # so that the actual packaged builder can be imported

        def __reduce__(self):  # to make dynamically created class pickable, see _InitializeParameterizedDatasetBuilder
            parent_builder_cls = self.__class__.__mro__[1]
            return (
                _InitializeConfiguredDatasetBuilder(),
                (
                    parent_builder_cls,
                    self.BUILDER_CONFIGS,
                    self.DEFAULT_CONFIG_NAME,
                    self.dataset_name,
                ),
                self.__dict__.copy(),
            )

    ConfiguredDatasetBuilder.__name__ = (
        f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}"
    )
    ConfiguredDatasetBuilder.__qualname__ = (
        f"{builder_cls.__name__.lower().capitalize()}{snakecase_to_camelcase(dataset_name)}"
    )

    return ConfiguredDatasetBuilder


def get_dataset_builder_class(
    dataset_module: "DatasetModule", dataset_name: Optional[str] = None
) -> Type[DatasetBuilder]:
    with lock_importable_file(
        dataset_module.importable_file_path
    ) if dataset_module.importable_file_path else nullcontext():
        builder_cls = import_main_class(dataset_module.module_path)
    if dataset_module.builder_configs_parameters.builder_configs:
        dataset_name = dataset_name or dataset_module.builder_kwargs.get("dataset_name")
        if dataset_name is None:
            raise ValueError("dataset_name should be specified but got None")
        builder_cls = configure_builder_class(
            builder_cls,
            builder_configs=dataset_module.builder_configs_parameters.builder_configs,
            default_config_name=dataset_module.builder_configs_parameters.default_config_name,
            dataset_name=dataset_name,
        )
    return builder_cls


def files_to_hash(file_paths: List[str]) -> str:
    """
    Convert a list of scripts or text files provided in file_paths into a hashed filename in a repeatable way.
    """
    # List all python files in directories if directories are supplied as part of external imports
    to_use_files: List[Union[Path, str]] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            to_use_files.extend(list(Path(file_path).rglob("*.[pP][yY]")))
        else:
            to_use_files.append(file_path)

    # Get the code from all these files
    lines = []
    for file_path in to_use_files:
        with open(file_path, encoding="utf-8") as f:
            lines.extend(f.readlines())
    return _hash_python_lines(lines)


def increase_load_count(name: str, resource_type: str):
    """Update the download count of a dataset or metric."""
    if not config.HF_HUB_OFFLINE and config.HF_UPDATE_DOWNLOAD_COUNTS:
        try:
            head_hf_s3(name, filename=name + ".py", dataset=(resource_type == "dataset"))
        except Exception:
            pass


def _download_additional_modules(
    name: str, base_path: str, imports: Tuple[str, str, str, str], download_config: Optional[DownloadConfig]
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Download additional module for a module <name>.py at URL (or local path) <base_path>/<name>.py
    The imports must have been parsed first using ``get_imports``.

    If some modules need to be installed with pip, an error is raised showing how to install them.
    This function return the list of downloaded modules as tuples (import_name, module_file_path).

    The downloaded modules can then be moved into an importable directory with ``_copy_script_and_other_resources_in_importable_dir``.
    """
    local_imports = []
    library_imports = []
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = "Downloading extra modules"
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == "library":
            library_imports.append((import_name, import_path))  # Import from a library
            continue

        if import_name == name:
            raise ValueError(
                f"Error in the {name} script, importing relative {import_name} module "
                f"but {import_name} is the name of the script. "
                f"Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' "
                f"comment pointing to the original relative import file path."
            )
        if import_type == "internal":
            url_or_filename = url_or_path_join(base_path, import_path + ".py")
        elif import_type == "external":
            url_or_filename = import_path
        else:
            raise ValueError("Wrong import_type")

        local_import_path = cached_path(
            url_or_filename,
            download_config=download_config,
        )
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))

    return local_imports, library_imports


def _check_library_imports(name: str, library_imports: List[Tuple[str, str]]) -> None:
    # Check library imports
    needs_to_be_installed = {}
    for library_import_name, library_import_path in library_imports:
        try:
            lib = importlib.import_module(library_import_name)  # noqa F841
        except ImportError:
            if library_import_name not in needs_to_be_installed or library_import_path != library_import_name:
                needs_to_be_installed[library_import_name] = library_import_path
    if needs_to_be_installed:
        _dependencies_str = "dependencies" if len(needs_to_be_installed) > 1 else "dependency"
        _them_str = "them" if len(needs_to_be_installed) > 1 else "it"
        if "sklearn" in needs_to_be_installed.keys():
            needs_to_be_installed["sklearn"] = "scikit-learn"
        if "Bio" in needs_to_be_installed.keys():
            needs_to_be_installed["Bio"] = "biopython"
        raise ImportError(
            f"To be able to use {name}, you need to install the following {_dependencies_str}: "
            f"{', '.join(needs_to_be_installed)}.\nPlease install {_them_str} using 'pip install "
            f"{' '.join(needs_to_be_installed.values())}' for instance."
        )


def _copy_script_and_other_resources_in_importable_dir(
    name: str,
    importable_directory_path: str,
    subdirectory_name: str,
    original_local_path: str,
    local_imports: List[Tuple[str, str]],
    additional_files: List[Tuple[str, str]],
    download_mode: Optional[Union[DownloadMode, str]],
) -> str:
    """Copy a script and its required imports to an importable directory

    Args:
        name (str): name of the resource to load
        importable_directory_path (str): path to the loadable folder in the dynamic modules directory
        subdirectory_name (str): name of the subdirectory in importable_directory_path in which to place the script
        original_local_path (str): local path to the resource script
        local_imports (List[Tuple[str, str]]): list of (destination_filename, import_file_to_copy)
        additional_files (List[Tuple[str, str]]): list of (destination_filename, additional_file_to_copy)
        download_mode (Optional[Union[DownloadMode, str]]): download mode

    Return:
        importable_file: path to an importable module with importlib.import_module
    """
    # Define a directory with a unique name in our dataset or metric folder
    # path is: ./datasets|metrics/dataset|metric_name/hash_from_code/script.py
    # we use a hash as subdirectory_name to be able to have multiple versions of a dataset/metric processing file together
    importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
    importable_file = os.path.join(importable_subdirectory, name + ".py")
    # Prevent parallel disk operations
    with lock_importable_file(importable_file):
        # Create main dataset/metrics folder if needed
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
            shutil.rmtree(importable_directory_path)
        os.makedirs(importable_directory_path, exist_ok=True)

        # add an __init__ file to the main dataset folder if needed
        init_file_path = os.path.join(importable_directory_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Create hash dataset folder if needed
        os.makedirs(importable_subdirectory, exist_ok=True)
        # add an __init__ file to the hash dataset folder if needed
        init_file_path = os.path.join(importable_subdirectory, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Copy dataset.py file in hash folder if needed
        if not os.path.exists(importable_file):
            shutil.copyfile(original_local_path, importable_file)
        # Record metadata associating original dataset path with local unique folder
        # Use os.path.splitext to split extension from importable_local_file
        meta_path = os.path.splitext(importable_file)[0] + ".json"
        if not os.path.exists(meta_path):
            meta = {"original file path": original_local_path, "local file path": importable_file}
            # the filename is *.py in our case, so better rename to filename.json instead of filename.py.json
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

        # Copy all the additional imports
        for import_name, import_path in local_imports:
            if os.path.isfile(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name + ".py")
                if not os.path.exists(full_path_local_import):
                    shutil.copyfile(import_path, full_path_local_import)
            elif os.path.isdir(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name)
                if not os.path.exists(full_path_local_import):
                    shutil.copytree(import_path, full_path_local_import)
            else:
                raise ImportError(f"Error with local import at {import_path}")

        # Copy additional files like dataset_infos.json file if needed
        for file_name, original_path in additional_files:
            destination_additional_path = os.path.join(importable_subdirectory, file_name)
            if not os.path.exists(destination_additional_path) or not filecmp.cmp(
                original_path, destination_additional_path
            ):
                shutil.copyfile(original_path, destination_additional_path)
        return importable_file


def _get_importable_file_path(
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
) -> str:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
    return os.path.join(importable_directory_path, subdirectory_name, name.split("/")[-1] + ".py")


def _create_importable_file(
    local_path: str,
    local_imports: List[Tuple[str, str]],
    additional_files: List[Tuple[str, str]],
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
    download_mode: DownloadMode,
) -> None:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
    Path(importable_directory_path).mkdir(parents=True, exist_ok=True)
    (Path(importable_directory_path).parent / "__init__.py").touch(exist_ok=True)
    importable_local_file = _copy_script_and_other_resources_in_importable_dir(
        name=name.split("/")[-1],
        importable_directory_path=importable_directory_path,
        subdirectory_name=subdirectory_name,
        original_local_path=local_path,
        local_imports=local_imports,
        additional_files=additional_files,
        download_mode=download_mode,
    )
    logger.debug(f"Created importable dataset file at {importable_local_file}")


def _load_importable_file(
    dynamic_modules_path: str,
    module_namespace: str,
    subdirectory_name: str,
    name: str,
) -> Tuple[str, str]:
    module_path = ".".join(
        [
            os.path.basename(dynamic_modules_path),
            module_namespace,
            name.replace("/", "--"),
            subdirectory_name,
            name.split("/")[-1],
        ]
    )
    return module_path, subdirectory_name


def infer_module_for_data_files_list(
    data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None
) -> Tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of data files.

    It picks the module based on the most common file extension.
    In case of a draw ".parquet" is the favorite, and then alphabetical order.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): mainly use use_auth_token or storage_options to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    extensions_counter = Counter(
        ("." + suffix.lower(), xbasename(filepath) in ("metadata.jsonl", "metadata.csv"))
        for filepath in data_files_list[: config.DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE]
        for suffix in xbasename(filepath).split(".")[1:]
    )
    if extensions_counter:

        def sort_key(ext_count: Tuple[Tuple[str, bool], int]) -> Tuple[int, bool]:
            """Sort by count and set ".parquet" as the favorite in case of a draw, and ignore metadata files"""
            (ext, is_metadata), count = ext_count
            return (not is_metadata, count, ext == ".parquet", ext)

        for (ext, _), _ in sorted(extensions_counter.items(), key=sort_key, reverse=True):
            if ext in _EXTENSION_TO_MODULE:
                return _EXTENSION_TO_MODULE[ext]
            elif ext == ".zip":
                return infer_module_for_data_files_list_in_archives(data_files_list, download_config=download_config)
    return None, {}


def infer_module_for_data_files_list_in_archives(
    data_files_list: DataFilesList, download_config: Optional[DownloadConfig] = None
) -> Tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of archive data files.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): mainly use use_auth_token or storage_options to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    archived_files = []
    archive_files_counter = 0
    for filepath in data_files_list:
        if str(filepath).endswith(".zip"):
            archive_files_counter += 1
            if archive_files_counter > config.GLOBBED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE:
                break
            extracted = xjoin(StreamingDownloadManager().extract(filepath), "**")
            archived_files += [
                f.split("::")[0]
                for f in xglob(extracted, recursive=True, download_config=download_config)[
                    : config.ARCHIVED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE
                ]
            ]
    extensions_counter = Counter(
        "." + suffix.lower() for filepath in archived_files for suffix in xbasename(filepath).split(".")[1:]
    )
    if extensions_counter:
        most_common = extensions_counter.most_common(1)[0][0]
        if most_common in _EXTENSION_TO_MODULE:
            return _EXTENSION_TO_MODULE[most_common]
    return None, {}


def infer_module_for_data_files(
    data_files: DataFilesDict, path: Optional[str] = None, download_config: Optional[DownloadConfig] = None
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Infer module (and builder kwargs) from data files. Raise if module names for different splits don't match.

    Args:
        data_files ([`DataFilesDict`]): Dict of list of data files.
        path (str, *optional*): Dataset name or path.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters to authenticate on the Hugging Face Hub for private remote files.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - builder kwargs
    """
    split_modules = {
        split: infer_module_for_data_files_list(data_files_list, download_config=download_config)
        for split, data_files_list in data_files.items()
    }
    module_name, default_builder_kwargs = next(iter(split_modules.values()))
    if any((module_name, default_builder_kwargs) != split_module for split_module in split_modules.values()):
        raise ValueError(f"Couldn't infer the same data file format for all splits. Got {split_modules}")
    if not module_name:
        raise DataFilesNotFoundError("No (supported) data files found" + (f" in {path}" if path else ""))
    return module_name, default_builder_kwargs


def create_builder_configs_from_metadata_configs(
    module_path: str,
    metadata_configs: MetadataConfigs,
    supports_metadata: bool,
    base_path: Optional[str] = None,
    default_builder_kwargs: Dict[str, Any] = None,
    download_config: Optional[DownloadConfig] = None,
) -> Tuple[List[BuilderConfig], str]:
    builder_cls = import_main_class(module_path)
    builder_config_cls = builder_cls.BUILDER_CONFIG_CLASS
    default_config_name = metadata_configs.get_default_config_name()
    builder_configs = []
    default_builder_kwargs = {} if default_builder_kwargs is None else default_builder_kwargs

    base_path = base_path if base_path is not None else ""
    for config_name, config_params in metadata_configs.items():
        config_data_files = config_params.get("data_files")
        config_data_dir = config_params.get("data_dir")
        config_base_path = xjoin(base_path, config_data_dir) if config_data_dir else base_path
        try:
            config_patterns = (
                sanitize_patterns(config_data_files)
                if config_data_files is not None
                else get_data_patterns(config_base_path, download_config=download_config)
            )
            config_data_files_dict = DataFilesPatternsDict.from_patterns(
                config_patterns,
                allowed_extensions=ALL_ALLOWED_EXTENSIONS,
            )
        except EmptyDatasetError as e:
            raise EmptyDatasetError(
                f"Dataset at '{base_path}' doesn't contain data files matching the patterns for config '{config_name}',"
                f" check `data_files` and `data_fir` parameters in the `configs` YAML field in README.md. "
            ) from e
        if config_data_files is None and supports_metadata and config_patterns != DEFAULT_PATTERNS_ALL:
            try:
                config_metadata_patterns = get_metadata_patterns(base_path, download_config=download_config)
            except FileNotFoundError:
                config_metadata_patterns = None
            if config_metadata_patterns is not None:
                config_metadata_data_files_list = DataFilesPatternsList.from_patterns(config_metadata_patterns)
                config_data_files_dict = DataFilesPatternsDict(
                    {
                        split: data_files_list + config_metadata_data_files_list
                        for split, data_files_list in config_data_files_dict.items()
                    }
                )
        ignored_params = [
            param for param in config_params if not hasattr(builder_config_cls, param) and param != "default"
        ]
        if ignored_params:
            logger.warning(
                f"Some datasets params were ignored: {ignored_params}. "
                "Make sure to use only valid params for the dataset builder and to have "
                "a up-to-date version of the `datasets` library."
            )
        builder_configs.append(
            builder_config_cls(
                name=config_name,
                data_files=config_data_files_dict,
                data_dir=config_data_dir,
                **{
                    param: value
                    for param, value in {**default_builder_kwargs, **config_params}.items()
                    if hasattr(builder_config_cls, param) and param not in ("default", "data_files", "data_dir")
                },
            )
        )
    return builder_configs, default_config_name


@dataclass
class BuilderConfigsParameters:
    """Dataclass containing objects related to creation of builder configurations from yaml's metadata content.

    Attributes:
        metadata_configs (`MetadataConfigs`, *optional*):
            Configs parsed from yaml's metadata.
        builder_configs (`list[BuilderConfig]`, *optional*):
            List of BuilderConfig objects created from metadata_configs above.
        default_config_name (`str`):
            Name of default config taken from yaml's metadata.
    """

    metadata_configs: Optional[MetadataConfigs] = None
    builder_configs: Optional[List[BuilderConfig]] = None
    default_config_name: Optional[str] = None


@dataclass
class DatasetModule:
    module_path: str
    hash: str
    builder_kwargs: dict
    builder_configs_parameters: BuilderConfigsParameters = field(default_factory=BuilderConfigsParameters)
    dataset_infos: Optional[DatasetInfosDict] = None
    importable_file_path: Optional[str] = None


@dataclass
class MetricModule:
    module_path: str
    hash: str


class _DatasetModuleFactory:
    def get_module(self) -> DatasetModule:
        raise NotImplementedError


class _MetricModuleFactory:
    def get_module(self) -> MetricModule:
        raise NotImplementedError


class GithubMetricModuleFactory(_MetricModuleFactory):
    """Get the module of a metric. The metric script is downloaded from GitHub.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>
    """

    @deprecated("Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate")
    def __init__(
        self,
        name: str,
        revision: Optional[Union[str, Version]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[str] = None,
    ):
        self.name = name
        self.revision = revision
        self.download_config = download_config.copy() if download_config else DownloadConfig()
        if self.download_config.max_retries < 3:
            self.download_config.max_retries = 3
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code
        assert self.name.count("/") == 0
        increase_load_count(name, resource_type="metric")

    def download_loading_script(self, revision: Optional[str]) -> str:
        file_path = hf_github_url(path=self.name, name=self.name + ".py", revision=revision, dataset=False)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading builder script"
        return cached_path(file_path, download_config=download_config)

    def get_module(self) -> MetricModule:
        if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
            _loading_script_url = hf_github_url(
                path=self.name, name=self.name + ".py", revision=self.revision, dataset=False
            )
            warnings.warn(
                f"The repository for {self.name} contains custom code which must be executed to correctly "
                f"load the metric. You can inspect the repository content at {_loading_script_url}\n"
                f"You can avoid this message in future by passing the argument `trust_remote_code=True`.\n"
                f"Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.",
                FutureWarning,
            )
        # get script and other files
        revision = self.revision
        try:
            local_path = self.download_loading_script(revision)
            revision = self.revision
        except FileNotFoundError:
            if revision is not None:
                raise
            else:
                revision = "main"
                local_path = self.download_loading_script(revision)
                logger.warning(
                    f"Couldn't find a directory or a metric named '{self.name}' in this version. "
                    f"It was picked from the main branch on github instead."
                )
        imports = get_imports(local_path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=hf_github_url(path=self.name, name="", revision=revision, dataset=False),
            imports=imports,
            download_config=self.download_config,
        )
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([local_path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="metrics",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=local_path,
                    local_imports=local_imports,
                    additional_files=[],
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="metrics",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(
                    f"Loading {self.name} requires you to execute the dataset script in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="metrics",
            subdirectory_name=hash,
            name=self.name,
        )
        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        return MetricModule(module_path, hash)


class LocalMetricModuleFactory(_MetricModuleFactory):
    """Get the module of a local metric. The metric script is loaded from a local script.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>
    """

    @deprecated("Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate")
    def __init__(
        self,
        path: str,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[str] = None,
    ):
        self.path = path
        self.name = Path(path).stem
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code

    def get_module(self) -> MetricModule:
        if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
            warnings.warn(
                f"The repository for {self.name} contains custom code which must be executed to correctly "
                f"load the metric. You can inspect the repository content at {self.path}\n"
                f"You can avoid this message in future by passing the argument `trust_remote_code=True`.\n"
                f"Passing `trust_remote_code=True` will be mandatory to load this metric from the next major release of `datasets`.",
                FutureWarning,
            )
        # get script and other files
        imports = get_imports(self.path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=str(Path(self.path).parent),
            imports=imports,
            download_config=self.download_config,
        )
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([self.path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="metrics",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=self.path,
                    local_imports=local_imports,
                    additional_files=[],
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="metrics",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(
                    f"Loading {self.name} requires you to execute the dataset script in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="metrics",
            subdirectory_name=hash,
            name=self.name,
        )
        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        return MetricModule(module_path, hash)


class LocalDatasetModuleFactoryWithScript(_DatasetModuleFactory):
    """Get the module of a local dataset. The dataset script is loaded from a local script."""

    def __init__(
        self,
        path: str,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
    ):
        self.path = path
        self.name = Path(path).stem
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code

    def get_module(self) -> DatasetModule:
        if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
            warnings.warn(
                f"The repository for {self.name} contains custom code which must be executed to correctly "
                f"load the dataset. You can inspect the repository content at {self.path}\n"
                f"You can avoid this message in future by passing the argument `trust_remote_code=True`.\n"
                f"Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.",
                FutureWarning,
            )
        # get script and other files
        dataset_infos_path = Path(self.path).parent / config.DATASETDICT_INFOS_FILENAME
        dataset_readme_path = Path(self.path).parent / config.REPOCARD_FILENAME
        imports = get_imports(self.path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=str(Path(self.path).parent),
            imports=imports,
            download_config=self.download_config,
        )
        additional_files = []
        if dataset_infos_path.is_file():
            additional_files.append((config.DATASETDICT_INFOS_FILENAME, str(dataset_infos_path)))
        if dataset_readme_path.is_file():
            additional_files.append((config.REPOCARD_FILENAME, dataset_readme_path))
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([self.path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=self.path,
                    local_imports=local_imports,
                    additional_files=additional_files,
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="datasets",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(
                    f"Loading {self.name} requires you to execute the dataset script in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )

        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        builder_kwargs = {"base_path": str(Path(self.path).parent)}
        return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)


class LocalDatasetModuleFactoryWithoutScript(_DatasetModuleFactory):
    """Get the module of a dataset loaded from the user's data files. The dataset builder module to use is inferred
    from the data files extensions."""

    def __init__(
        self,
        path: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List, Dict]] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        if data_dir and os.path.isabs(data_dir):
            raise ValueError(f"`data_dir` must be relative to a dataset directory's root: {path}")

        self.path = Path(path).as_posix()
        self.name = Path(path).stem
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_mode = download_mode

    def get_module(self) -> DatasetModule:
        readme_path = os.path.join(self.path, config.REPOCARD_FILENAME)
        standalone_yaml_path = os.path.join(self.path, config.REPOYAML_FILENAME)
        dataset_card_data = DatasetCard.load(readme_path).data if os.path.isfile(readme_path) else DatasetCardData()
        if os.path.exists(standalone_yaml_path):
            with open(standalone_yaml_path, "r", encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        # we need a set of data files to find which dataset builder to use
        # because we need to infer module name by files extensions
        base_path = Path(self.path, self.data_dir or "").expanduser().resolve().as_posix()
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=ALL_ALLOWED_EXTENSIONS,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.path,
        )
        data_files = data_files.filter_extensions(_MODULE_TO_EXTENSIONS[module_name])
        # Collect metadata files if the module supports them
        supports_metadata = module_name in _MODULE_SUPPORTS_METADATA
        if self.data_files is None and supports_metadata:
            try:
                metadata_patterns = get_metadata_patterns(base_path)
            except FileNotFoundError:
                metadata_patterns = None
            if metadata_patterns is not None:
                metadata_data_files_list = DataFilesList.from_patterns(metadata_patterns, base_path=base_path)
                if metadata_data_files_list:
                    data_files = DataFilesDict(
                        {
                            split: data_files_list + metadata_data_files_list
                            for split, data_files_list in data_files.items()
                        }
                    )

        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                supports_metadata=supports_metadata,
                default_builder_kwargs=default_builder_kwargs,
            )
        else:
            builder_configs: List[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": self.path,
            "dataset_name": camelcase_to_snakecase(Path(self.path).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        # this file is deprecated and was created automatically in old versions of push_to_hub
        if os.path.isfile(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME)):
            with open(os.path.join(self.path, config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    # old config e.g. named "username--dataset_name"
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        hash = Hasher.hash({"dataset_infos": dataset_infos, "builder_configs": builder_configs})
        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class PackagedDatasetModuleFactory(_DatasetModuleFactory):
    """Get the dataset builder module from the ones that are packaged with the library: csv, json, etc."""

    def __init__(
        self,
        name: str,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List, Dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        self.name = name
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config
        self.download_mode = download_mode
        increase_load_count(name, resource_type="dataset")

    def get_module(self) -> DatasetModule:
        base_path = Path(self.data_dir or "").expanduser().resolve().as_posix()
        patterns = (
            sanitize_patterns(self.data_files)
            if self.data_files is not None
            else get_data_patterns(base_path, download_config=self.download_config)
        )
        data_files = DataFilesDict.from_patterns(
            patterns,
            download_config=self.download_config,
            base_path=base_path,
        )
        supports_metadata = self.name in _MODULE_SUPPORTS_METADATA
        if self.data_files is None and supports_metadata and patterns != DEFAULT_PATTERNS_ALL:
            try:
                metadata_patterns = get_metadata_patterns(base_path, download_config=self.download_config)
            except FileNotFoundError:
                metadata_patterns = None
            if metadata_patterns is not None:
                metadata_data_files_list = DataFilesList.from_patterns(
                    metadata_patterns, download_config=self.download_config, base_path=base_path
                )
                if metadata_data_files_list:
                    data_files = DataFilesDict(
                        {
                            split: data_files_list + metadata_data_files_list
                            for split, data_files_list in data_files.items()
                        }
                    )

        module_path, hash = _PACKAGED_DATASETS_MODULES[self.name]

        builder_kwargs = {
            "data_files": data_files,
            "dataset_name": self.name,
        }

        return DatasetModule(module_path, hash, builder_kwargs)


class HubDatasetModuleFactoryWithoutScript(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from data files of a dataset repository.
    The dataset builder module to use is inferred from the data files extensions.
    """

    def __init__(
        self,
        name: str,
        revision: Optional[Union[str, Version]] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List, Dict]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
    ):
        self.name = name
        self.revision = revision
        self.data_files = data_files
        self.data_dir = data_dir
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        increase_load_count(name, resource_type="dataset")

    def get_module(self) -> DatasetModule:
        hfh_dataset_info = HfApi(config.HF_ENDPOINT).dataset_info(
            self.name,
            revision=self.revision,
            token=self.download_config.token,
            timeout=100.0,
        )
        # even if metadata_configs is not None (which means that we will resolve files for each config later)
        # we cannot skip resolving all files because we need to infer module name by files extensions
        revision = hfh_dataset_info.sha  # fix the revision in case there are new commits in the meantime
        base_path = f"hf://datasets/{self.name}@{revision}/{self.data_dir or ''}".rstrip("/")

        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading readme"
        try:
            dataset_readme_path = cached_path(
                hf_dataset_url(self.name, config.REPOCARD_FILENAME, revision=revision),
                download_config=download_config,
            )
            dataset_card_data = DatasetCard.load(Path(dataset_readme_path)).data
        except FileNotFoundError:
            dataset_card_data = DatasetCardData()
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading standalone yaml"
        try:
            standalone_yaml_path = cached_path(
                hf_dataset_url(self.name, config.REPOYAML_FILENAME, revision=revision),
                download_config=download_config,
            )
            with open(standalone_yaml_path, "r", encoding="utf-8") as f:
                standalone_yaml_data = yaml.safe_load(f.read())
                if standalone_yaml_data:
                    _dataset_card_data_dict = dataset_card_data.to_dict()
                    _dataset_card_data_dict.update(standalone_yaml_data)
                    dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
        except FileNotFoundError:
            pass
        metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(dataset_card_data)
        # Use the infos from the parquet export except in some cases:
        if self.data_dir or self.data_files or (self.revision and self.revision != "main"):
            use_exported_dataset_infos = False
        else:
            use_exported_dataset_infos = True
        if config.USE_PARQUET_EXPORT and use_exported_dataset_infos:
            try:
                exported_dataset_infos = _dataset_viewer.get_exported_dataset_infos(
                    dataset=self.name, revision=self.revision, token=self.download_config.token
                )
                exported_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                        for config_name in exported_dataset_infos
                    }
                )
            except _dataset_viewer.DatasetViewerError:
                exported_dataset_infos = None
        else:
            exported_dataset_infos = None
        if exported_dataset_infos:
            exported_dataset_infos.update(dataset_infos)
            dataset_infos = exported_dataset_infos
        # we need a set of data files to find which dataset builder to use
        # because we need to infer module name by files extensions
        if self.data_files is not None:
            patterns = sanitize_patterns(self.data_files)
        elif metadata_configs and not self.data_dir and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(base_path, download_config=self.download_config)
        data_files = DataFilesDict.from_patterns(
            patterns,
            base_path=base_path,
            allowed_extensions=ALL_ALLOWED_EXTENSIONS,
            download_config=self.download_config,
        )
        module_name, default_builder_kwargs = infer_module_for_data_files(
            data_files=data_files,
            path=self.name,
            download_config=self.download_config,
        )
        data_files = data_files.filter_extensions(_MODULE_TO_EXTENSIONS[module_name])
        # Collect metadata files if the module supports them
        supports_metadata = module_name in _MODULE_SUPPORTS_METADATA
        if self.data_files is None and supports_metadata:
            try:
                metadata_patterns = get_metadata_patterns(base_path, download_config=self.download_config)
            except FileNotFoundError:
                metadata_patterns = None
            if metadata_patterns is not None:
                metadata_data_files_list = DataFilesList.from_patterns(
                    metadata_patterns, download_config=self.download_config, base_path=base_path
                )
                if metadata_data_files_list:
                    data_files = DataFilesDict(
                        {
                            split: data_files_list + metadata_data_files_list
                            for split, data_files_list in data_files.items()
                        }
                    )

        module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
        if metadata_configs:
            builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=base_path,
                supports_metadata=supports_metadata,
                default_builder_kwargs=default_builder_kwargs,
                download_config=self.download_config,
            )
        else:
            builder_configs: List[BuilderConfig] = [
                import_main_class(module_path).BUILDER_CONFIG_CLASS(
                    data_files=data_files,
                    **default_builder_kwargs,
                )
            ]
            default_config_name = None
        builder_kwargs = {
            "base_path": hf_dataset_url(self.name, "", revision=revision).rstrip("/"),
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }
        if self.data_dir:
            builder_kwargs["data_files"] = data_files
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading metadata"
        try:
            # this file is deprecated and was created automatically in old versions of push_to_hub
            dataset_infos_path = cached_path(
                hf_dataset_url(self.name, config.DATASETDICT_INFOS_FILENAME, revision=revision),
                download_config=download_config,
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                legacy_dataset_infos = DatasetInfosDict(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
                if len(legacy_dataset_infos) == 1:
                    # old config e.g. named "username--dataset_name"
                    legacy_config_name = next(iter(legacy_dataset_infos))
                    legacy_dataset_infos["default"] = legacy_dataset_infos.pop(legacy_config_name)
            legacy_dataset_infos.update(dataset_infos)
            dataset_infos = legacy_dataset_infos
        except FileNotFoundError:
            pass
        if default_config_name is None and len(dataset_infos) == 1:
            default_config_name = next(iter(dataset_infos))

        hash = revision
        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class HubDatasetModuleFactoryWithParquetExport(_DatasetModuleFactory):
    """
    Get the module of a dataset loaded from parquet files of a dataset repository parquet export.
    """

    def __init__(
        self,
        name: str,
        revision: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
    ):
        self.name = name
        self.revision = revision
        self.download_config = download_config or DownloadConfig()
        increase_load_count(name, resource_type="dataset")

    def get_module(self) -> DatasetModule:
        exported_parquet_files = _dataset_viewer.get_exported_parquet_files(
            dataset=self.name, revision=self.revision, token=self.download_config.token
        )
        exported_dataset_infos = _dataset_viewer.get_exported_dataset_infos(
            dataset=self.name, revision=self.revision, token=self.download_config.token
        )
        dataset_infos = DatasetInfosDict(
            {
                config_name: DatasetInfo.from_dict(exported_dataset_infos[config_name])
                for config_name in exported_dataset_infos
            }
        )
        hfh_dataset_info = HfApi(config.HF_ENDPOINT).dataset_info(
            self.name,
            revision="refs/convert/parquet",
            token=self.download_config.token,
            timeout=100.0,
        )
        revision = hfh_dataset_info.sha  # fix the revision in case there are new commits in the meantime
        metadata_configs = MetadataConfigs._from_exported_parquet_files_and_dataset_infos(
            revision=revision, exported_parquet_files=exported_parquet_files, dataset_infos=dataset_infos
        )
        module_path, _ = _PACKAGED_DATASETS_MODULES["parquet"]
        builder_configs, default_config_name = create_builder_configs_from_metadata_configs(
            module_path,
            metadata_configs,
            supports_metadata=False,
            download_config=self.download_config,
        )
        hash = self.revision
        builder_kwargs = {
            "repo_id": self.name,
            "dataset_name": camelcase_to_snakecase(Path(self.name).name),
        }

        return DatasetModule(
            module_path,
            hash,
            builder_kwargs,
            dataset_infos=dataset_infos,
            builder_configs_parameters=BuilderConfigsParameters(
                metadata_configs=metadata_configs,
                builder_configs=builder_configs,
                default_config_name=default_config_name,
            ),
        )


class HubDatasetModuleFactoryWithScript(_DatasetModuleFactory):
    """
    Get the module of a dataset from a dataset repository.
    The dataset script comes from the script inside the dataset repository.
    """

    def __init__(
        self,
        name: str,
        revision: Optional[Union[str, Version]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        dynamic_modules_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
    ):
        self.name = name
        self.revision = revision
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        self.trust_remote_code = trust_remote_code
        increase_load_count(name, resource_type="dataset")

    def download_loading_script(self) -> str:
        file_path = hf_dataset_url(self.name, self.name.split("/")[-1] + ".py", revision=self.revision)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading builder script"
        return cached_path(file_path, download_config=download_config)

    def download_dataset_infos_file(self) -> str:
        dataset_infos = hf_dataset_url(self.name, config.DATASETDICT_INFOS_FILENAME, revision=self.revision)
        # Download the dataset infos file if available
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading metadata"
        try:
            return cached_path(
                dataset_infos,
                download_config=download_config,
            )
        except (FileNotFoundError, ConnectionError):
            return None

    def download_dataset_readme_file(self) -> str:
        readme_url = hf_dataset_url(self.name, config.REPOCARD_FILENAME, revision=self.revision)
        # Download the dataset infos file if available
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading readme"
        try:
            return cached_path(
                readme_url,
                download_config=download_config,
            )
        except (FileNotFoundError, ConnectionError):
            return None

    def get_module(self) -> DatasetModule:
        if config.HF_DATASETS_TRUST_REMOTE_CODE and self.trust_remote_code is None:
            warnings.warn(
                f"The repository for {self.name} contains custom code which must be executed to correctly "
                f"load the dataset. You can inspect the repository content at https://hf.co/datasets/{self.name}\n"
                f"You can avoid this message in future by passing the argument `trust_remote_code=True`.\n"
                f"Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.",
                FutureWarning,
            )
        # get script and other files
        local_path = self.download_loading_script()
        dataset_infos_path = self.download_dataset_infos_file()
        dataset_readme_path = self.download_dataset_readme_file()
        imports = get_imports(local_path)
        local_imports, library_imports = _download_additional_modules(
            name=self.name,
            base_path=hf_dataset_url(self.name, "", revision=self.revision),
            imports=imports,
            download_config=self.download_config,
        )
        additional_files = []
        if dataset_infos_path:
            additional_files.append((config.DATASETDICT_INFOS_FILENAME, dataset_infos_path))
        if dataset_readme_path:
            additional_files.append((config.REPOCARD_FILENAME, dataset_readme_path))
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        hash = files_to_hash([local_path] + [loc[1] for loc in local_imports])
        importable_file_path = _get_importable_file_path(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        if not os.path.exists(importable_file_path):
            trust_remote_code = resolve_trust_remote_code(self.trust_remote_code, self.name)
            if trust_remote_code:
                _create_importable_file(
                    local_path=local_path,
                    local_imports=local_imports,
                    additional_files=additional_files,
                    dynamic_modules_path=dynamic_modules_path,
                    module_namespace="datasets",
                    subdirectory_name=hash,
                    name=self.name,
                    download_mode=self.download_mode,
                )
            else:
                raise ValueError(
                    f"Loading {self.name} requires you to execute the dataset script in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
        _check_library_imports(name=self.name, library_imports=library_imports)
        module_path, hash = _load_importable_file(
            dynamic_modules_path=dynamic_modules_path,
            module_namespace="datasets",
            subdirectory_name=hash,
            name=self.name,
        )
        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        builder_kwargs = {
            "base_path": hf_dataset_url(self.name, "", revision=self.revision).rstrip("/"),
            "repo_id": self.name,
        }
        return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)


class CachedDatasetModuleFactory(_DatasetModuleFactory):
    """
    Get the module of a dataset that has been loaded once already and cached.
    The script that is loaded from the cache is the most recent one with a matching name.
    """

    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
        dynamic_modules_path: Optional[str] = None,
    ):
        self.name = name
        self.cache_dir = cache_dir
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count("/") <= 1

    def get_module(self) -> DatasetModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, "datasets", self.name.replace("/", "--"))
        hashes = (
            [h for h in os.listdir(importable_directory_path) if len(h) == 64]
            if os.path.isdir(importable_directory_path)
            else None
        )
        if hashes:
            # get most recent
            def _get_modification_time(module_hash):
                return (
                    (Path(importable_directory_path) / module_hash / (self.name.split("/")[-1] + ".py"))
                    .stat()
                    .st_mtime
                )

            hash = sorted(hashes, key=_get_modification_time)[-1]
            warning_msg = (
                f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} "
                f"(last modified on {time.ctime(_get_modification_time(hash))}) since it "
                f"couldn't be found locally at {self.name}"
            )
            if not config.HF_HUB_OFFLINE:
                warning_msg += ", or remotely on the Hugging Face Hub."
            logger.warning(warning_msg)
            importable_file_path = _get_importable_file_path(
                dynamic_modules_path=dynamic_modules_path,
                module_namespace="datasets",
                subdirectory_name=hash,
                name=self.name,
            )
            module_path, hash = _load_importable_file(
                dynamic_modules_path=dynamic_modules_path,
                module_namespace="datasets",
                subdirectory_name=hash,
                name=self.name,
            )
            # make the new module to be noticed by the import system
            importlib.invalidate_caches()
            builder_kwargs = {
                "repo_id": self.name,
            }
            return DatasetModule(module_path, hash, builder_kwargs, importable_file_path=importable_file_path)
        cache_dir = os.path.expanduser(str(self.cache_dir or config.HF_DATASETS_CACHE))
        namespace_and_dataset_name = self.name.split("/")
        namespace_and_dataset_name[-1] = camelcase_to_snakecase(namespace_and_dataset_name[-1])
        cached_relative_path = "___".join(namespace_and_dataset_name)
        cached_datasets_directory_path_root = os.path.join(cache_dir, cached_relative_path)
        cached_directory_paths = [
            cached_directory_path
            for cached_directory_path in glob.glob(os.path.join(cached_datasets_directory_path_root, "*", "*", "*"))
            if os.path.isdir(cached_directory_path)
        ]
        if cached_directory_paths:
            builder_kwargs = {
                "repo_id": self.name,
                "dataset_name": self.name.split("/")[-1],
            }
            warning_msg = f"Using the latest cached version of the dataset since {self.name} couldn't be found on the Hugging Face Hub"
            if config.HF_HUB_OFFLINE:
                warning_msg += " (offline mode is enabled)."
            logger.warning(warning_msg)
            return DatasetModule(
                "datasets.packaged_modules.cache.cache",
                "auto",
                {**builder_kwargs, "version": "auto"},
            )
        raise FileNotFoundError(f"Dataset {self.name} is not cached in {self.cache_dir}")


class CachedMetricModuleFactory(_MetricModuleFactory):
    """
    Get the module of a metric that has been loaded once already and cached.
    The script that is loaded from the cache is the most recent one with a matching name.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>
    """

    @deprecated("Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate")
    def __init__(
        self,
        name: str,
        dynamic_modules_path: Optional[str] = None,
    ):
        self.name = name
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count("/") == 0

    def get_module(self) -> MetricModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, "metrics", self.name)
        hashes = (
            [h for h in os.listdir(importable_directory_path) if len(h) == 64]
            if os.path.isdir(importable_directory_path)
            else None
        )
        if not hashes:
            raise FileNotFoundError(f"Metric {self.name} is not cached in {dynamic_modules_path}")
        # get most recent

        def _get_modification_time(module_hash):
            return (Path(importable_directory_path) / module_hash / (self.name + ".py")).stat().st_mtime

        hash = sorted(hashes, key=_get_modification_time)[-1]
        logger.warning(
            f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} "
            f"(last modified on {time.ctime(_get_modification_time(hash))}) since it "
            f"couldn't be found locally at {self.name}, or remotely on the Hugging Face Hub."
        )
        # make the new module to be noticed by the import system
        module_path = ".".join([os.path.basename(dynamic_modules_path), "metrics", self.name, hash, self.name])
        importlib.invalidate_caches()
        return MetricModule(module_path, hash)


def dataset_module_factory(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    dynamic_modules_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[Dict, List, str, DataFilesDict]] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    _require_default_config_name=True,
    _require_custom_configs=False,
    **download_kwargs,
) -> DatasetModule:
    """
    Download/extract/cache a dataset module.

    Dataset codes are cached inside the dynamic modules cache to allow easy import (avoid ugly sys.path tweaks).

    Args:

        path (str): Path or name of the dataset.
            Depending on ``path``, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if ``path`` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. ``'./path/to/directory/with/my/csv/data'``.
            - if ``path`` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory):
              -> load the dataset builder from the dataset script
              e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``.

            For datasets on the Hugging Face Hub (list all available datasets with ``huggingface_hub.list_datasets()``)

            - if ``path`` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. ``'username/dataset_name'``, a dataset repository on the HF hub containing your data files.
            - if ``path`` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. ``glue``, ``squad``, ``'username/dataset_name'``, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        dynamic_modules_path (Optional str, defaults to HF_MODULES_CACHE / "datasets_modules", i.e. ~/.cache/huggingface/modules/datasets_modules):
            Optional path to the directory in which the dynamic modules are saved. It must have been initialized with :obj:`init_dynamic_modules`.
            By default, the datasets and metrics are stored inside the `datasets_modules` module.
        data_dir (:obj:`str`, optional): Directory with the data files. Used only if `data_files` is not specified,
            in which case it's equal to pass `os.path.join(data_dir, "**")` as `data_files`.
        data_files (:obj:`Union[Dict, List, str]`, optional): Defining the data_files of the dataset configuration.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.

            <Added version="2.16.0"/>
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

        **download_kwargs (additional keyword arguments): optional attributes for DownloadConfig() which will override
            the attributes in download_config if supplied.

    Returns:
        DatasetModule
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    download_config.extract_compressed_file = True
    download_config.force_extract = True
    download_config.force_download = download_mode == DownloadMode.FORCE_REDOWNLOAD

    filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)

    # We have several ways to get a dataset builder:
    #
    # - if path is the name of a packaged dataset module
    #   -> use the packaged module (json, csv, etc.)
    #
    # - if os.path.join(path, name) is a local python file
    #   -> use the module from the python file
    # - if path is a local directory (but no python file)
    #   -> use a packaged module (csv, text etc.) based on content of the directory
    #
    # - if path has one "/" and is dataset repository on the HF hub with a python file
    #   -> the module from the python file in the dataset repository
    # - if path has one "/" and is dataset repository on the HF hub without a python file
    #   -> use a packaged module (csv, text etc.) based on content of the repository

    # Try packaged
    if path in _PACKAGED_DATASETS_MODULES:
        return PackagedDatasetModuleFactory(
            path,
            data_dir=data_dir,
            data_files=data_files,
            download_config=download_config,
            download_mode=download_mode,
        ).get_module()
    # Try locally
    elif path.endswith(filename):
        if os.path.isfile(path):
            return LocalDatasetModuleFactoryWithScript(
                path,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                trust_remote_code=trust_remote_code,
            ).get_module()
        else:
            raise FileNotFoundError(f"Couldn't find a dataset script at {relative_to_absolute_path(path)}")
    elif os.path.isfile(combined_path):
        return LocalDatasetModuleFactoryWithScript(
            combined_path,
            download_mode=download_mode,
            dynamic_modules_path=dynamic_modules_path,
            trust_remote_code=trust_remote_code,
        ).get_module()
    elif os.path.isdir(path):
        return LocalDatasetModuleFactoryWithoutScript(
            path, data_dir=data_dir, data_files=data_files, download_mode=download_mode
        ).get_module()
    # Try remotely
    elif is_relative_path(path) and path.count("/") <= 1:
        try:
            _raise_if_offline_mode_is_enabled()
            hf_api = HfApi(config.HF_ENDPOINT)
            try:
                dataset_info = hf_api.dataset_info(
                    repo_id=path,
                    revision=revision,
                    token=download_config.token,
                    timeout=100.0,
                )
            except (
                OfflineModeIsEnabled,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
            ) as e:
                raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
            except GatedRepoError as e:
                message = f"Dataset '{path}' is a gated dataset on the Hub."
                if "401 Client Error" in str(e):
                    message += " You must be authenticated to access it."
                elif "403 Client Error" in str(e):
                    message += f" Visit the dataset page at https://huggingface.co/datasets/{path} to ask for access."
                raise DatasetNotFoundError(message) from e
            except RevisionNotFoundError as e:
                raise DatasetNotFoundError(
                    f"Revision '{revision}' doesn't exist for dataset '{path}' on the Hub."
                ) from e
            except RepositoryNotFoundError as e:
                raise DatasetNotFoundError(f"Dataset '{path}' doesn't exist on the Hub or cannot be accessed.") from e

            if filename in [sibling.rfilename for sibling in dataset_info.siblings]:  # contains a dataset script
                fs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
                if _require_custom_configs or (revision and revision != "main"):
                    can_load_config_from_parquet_export = False
                elif _require_default_config_name:
                    with fs.open(f"datasets/{path}/{filename}", "r", encoding="utf-8") as f:
                        can_load_config_from_parquet_export = "DEFAULT_CONFIG_NAME" not in f.read()
                else:
                    can_load_config_from_parquet_export = True
                if config.USE_PARQUET_EXPORT and can_load_config_from_parquet_export:
                    # If the parquet export is ready (parquet files + info available for the current sha), we can use it instead
                    # This fails when the dataset has multiple configs and a default config and
                    # the user didn't specify a configuration name (_require_default_config_name=True).
                    try:
                        return HubDatasetModuleFactoryWithParquetExport(
                            path, download_config=download_config, revision=dataset_info.sha
                        ).get_module()
                    except _dataset_viewer.DatasetViewerError:
                        pass
                # Otherwise we must use the dataset script if the user trusts it
                return HubDatasetModuleFactoryWithScript(
                    path,
                    revision=revision,
                    download_config=download_config,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            else:
                return HubDatasetModuleFactoryWithoutScript(
                    path,
                    revision=revision,
                    data_dir=data_dir,
                    data_files=data_files,
                    download_config=download_config,
                    download_mode=download_mode,
                ).get_module()
        except Exception as e1:
            # All the attempts failed, before raising the error we should check if the module is already cached
            try:
                return CachedDatasetModuleFactory(
                    path, dynamic_modules_path=dynamic_modules_path, cache_dir=cache_dir
                ).get_module()
            except Exception:
                # If it's not in the cache, then it doesn't exist.
                if isinstance(e1, OfflineModeIsEnabled):
                    raise ConnectionError(f"Couldn't reach the Hugging Face Hub for dataset '{path}': {e1}") from None
                if isinstance(e1, (DataFilesNotFoundError, DatasetNotFoundError, EmptyDatasetError)):
                    raise e1 from None
                if isinstance(e1, FileNotFoundError):
                    raise FileNotFoundError(
                        f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory. "
                        f"Couldn't find '{path}' on the Hugging Face Hub either: {type(e1).__name__}: {e1}"
                    ) from None
                raise e1 from None
    else:
        raise FileNotFoundError(
            f"Couldn't find a dataset script at {relative_to_absolute_path(combined_path)} or any data file in the same directory."
        )


@deprecated("Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate")
def metric_module_factory(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    dynamic_modules_path: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    **download_kwargs,
) -> MetricModule:
    """
    Download/extract/cache a metric module.

    <Deprecated version="2.5.0">

    Use the new library 🤗 Evaluate instead: https://huggingface.co/docs/evaluate

    </Deprecated>

    Metrics codes are cached inside the dynamic modules cache to allow easy import (avoid ugly sys.path tweaks).

    Args:

        path (str): Path or name of the metric script.

            - if ``path`` is a local metric script or a directory containing a local metric script (if the script has the same name as the directory):
              -> load the module from the metric script
              e.g. ``'./metrics/accuracy'`` or ``'./metrics/accuracy/accuracy.py'``.
            - if ``path`` is a metric on the Hugging Face Hub (ex: `glue`, `squad`)
              -> load the module from the metric script in the GitHub repository at huggingface/datasets
              e.g. ``'accuracy'`` or ``'rouge'``.

        revision (Optional ``Union[str, datasets.Version]``):
            If specified, the module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        dynamic_modules_path (Optional str, defaults to HF_MODULES_CACHE / "datasets_modules", i.e. ~/.cache/huggingface/modules/datasets_modules):
            Optional path to the directory in which the dynamic modules are saved. It must have been initialized with :obj:`init_dynamic_modules`.
            By default, the datasets and metrics are stored inside the `datasets_modules` module.
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

        **download_kwargs (additional keyword arguments): optional attributes for DownloadConfig() which will override
            the attributes in download_config if supplied.

    Returns:
        MetricModule
    """
    with warnings.catch_warnings():
        # Ignore equivalent warnings to the one already issued
        warnings.filterwarnings("ignore", message=".*https://huggingface.co/docs/evaluate$", category=FutureWarning)

        if download_config is None:
            download_config = DownloadConfig(**download_kwargs)
        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        download_config.extract_compressed_file = True
        download_config.force_extract = True

        filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
        if not filename.endswith(".py"):
            filename = filename + ".py"
        combined_path = os.path.join(path, filename)
        # Try locally
        if path.endswith(filename):
            if os.path.isfile(path):
                return LocalMetricModuleFactory(
                    path,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            else:
                raise FileNotFoundError(f"Couldn't find a metric script at {relative_to_absolute_path(path)}")
        elif os.path.isfile(combined_path):
            return LocalMetricModuleFactory(
                combined_path,
                download_mode=download_mode,
                dynamic_modules_path=dynamic_modules_path,
                trust_remote_code=trust_remote_code,
            ).get_module()
        elif is_relative_path(path) and path.count("/") == 0:
            try:
                return GithubMetricModuleFactory(
                    path,
                    revision=revision,
                    download_config=download_config,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                    trust_remote_code=trust_remote_code,
                ).get_module()
            except Exception as e1:  # noqa all the attempts failed, before raising the error we should check if the module is already cached.
                try:
                    return CachedMetricModuleFactory(path, dynamic_modules_path=dynamic_modules_path).get_module()
                except Exception:  # noqa if it's not in the cache, then it doesn't exist.
                    if not isinstance(e1, FileNotFoundError):
                        raise e1 from None
                    raise FileNotFoundError(
                        f"Couldn't find a metric script at {relative_to_absolute_path(combined_path)}. "
                        f"Metric '{path}' doesn't exist on the Hugging Face Hub either."
                    ) from None
        else:
            raise FileNotFoundError(f"Couldn't find a metric script at {relative_to_absolute_path(combined_path)}.")


@deprecated("Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate")
def load_metric(
    path: str,
    config_name: Optional[str] = None,
    process_id: int = 0,
    num_process: int = 1,
    cache_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    keep_in_memory: bool = False,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    trust_remote_code: Optional[bool] = None,
    **metric_init_kwargs,
) -> Metric:
    """Load a `datasets.Metric`.

    <Deprecated version="2.5.0">

    Use `evaluate.load` instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate

    </Deprecated>

    Args:

        path (``str``):
            path to the metric processing script with the metric builder. Can be either:
                - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                    e.g. ``'./metrics/rouge'`` or ``'./metrics/rogue/rouge.py'``
                - a metric identifier on the HuggingFace datasets repo (list all available metrics with ``datasets.list_metrics()``)
                    e.g. ``'rouge'`` or ``'bleu'``
        config_name (:obj:`str`, optional): selecting a configuration for the metric (e.g. the GLUE metric has a configuration for each subset)
        process_id (:obj:`int`, optional): for distributed evaluation: id of the process
        num_process (:obj:`int`, optional): for distributed evaluation: total number of processes
        cache_dir (Optional str): path to store the temporary predictions and references (default to `~/.cache/huggingface/metrics/`)
        experiment_id (``str``): A specific experiment id. This is used if several distributed evaluations share the same file system.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        keep_in_memory (bool): Whether to store the temporary results in memory (defaults to False)
        download_config (Optional ``datasets.DownloadConfig``: specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        revision (Optional ``Union[str, datasets.Version]``): if specified, the module will be loaded from the datasets repository
            at this version. By default, it is set to the local version of the lib. Specifying a version that is different from
            your local version of the lib might cause compatibility issues.
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

    Returns:
        `datasets.Metric`

    Example:

    ```py
    >>> from datasets import load_metric
    >>> accuracy = load_metric('accuracy')
    >>> accuracy.compute(references=[1, 0], predictions=[1, 1])
    {'accuracy': 0.5}
    ```
    """
    with warnings.catch_warnings():
        # Ignore equivalent warnings to the one already issued
        warnings.filterwarnings("ignore", message=".*https://huggingface.co/docs/evaluate$", category=FutureWarning)

        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        metric_module = metric_module_factory(
            path,
            revision=revision,
            download_config=download_config,
            download_mode=download_mode,
            trust_remote_code=trust_remote_code,
        ).module_path
        metric_cls = import_main_class(metric_module, dataset=False)
        metric = metric_cls(
            config_name=config_name,
            process_id=process_id,
            num_process=num_process,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            experiment_id=experiment_id,
            **metric_init_kwargs,
        )

        # Download and prepare resources for the metric
        metric.download_and_prepare(download_config=download_config)

        return metric


def load_dataset_builder(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token="deprecated",
    storage_options: Optional[Dict] = None,
    trust_remote_code: Optional[bool] = None,
    _require_default_config_name=True,
    **config_kwargs,
) -> DatasetBuilder:
    """Load a dataset builder from the Hugging Face Hub, or a local dataset. A dataset builder can be used to inspect general information that is required to build a dataset (cache directory, config, dataset info, etc.)
    without downloading the dataset itself.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains:

    - some data files in generic formats (JSON, CSV, Parquet, text, etc.)
    - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

    Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.

    Args:

        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features ([`Features`], *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`]
            and used in the [`DatasetBuilder`].

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
     'text': Value(dtype='string', id=None)}
    ```
    """
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            "You can remove this warning by passing 'token=<use_auth_token>' instead.",
            FutureWarning,
        )
        token = use_auth_token
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    if storage_options is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.storage_options.update(storage_options)
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=_require_default_config_name,
        _require_custom_configs=bool(config_kwargs),
    )
    # Get dataset builder class from the processing script
    builder_kwargs = dataset_module.builder_kwargs
    data_dir = builder_kwargs.pop("data_dir", data_dir)
    data_files = builder_kwargs.pop("data_files", data_files)
    config_name = builder_kwargs.pop(
        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None

    if (
        path in _PACKAGED_DATASETS_MODULES
        and data_files is None
        and dataset_module.builder_configs_parameters.builder_configs[0].data_files is None
    ):
        error_msg = f"Please specify the data files or data directory to load for the {path} dataset builder."
        example_extensions = [
            extension for extension in _EXTENSION_TO_MODULE if _EXTENSION_TO_MODULE[extension] == path
        ]
        if example_extensions:
            error_msg += f'\nFor example `data_files={{"train": "path/to/data/train/*.{example_extensions[0]}"}}`'
        raise ValueError(error_msg)

    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)
    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        config_name=config_name,
        data_dir=data_dir,
        data_files=data_files,
        hash=dataset_module.hash,
        info=info,
        features=features,
        token=token,
        storage_options=storage_options,
        **builder_kwargs,
        **config_kwargs,
    )
    builder_instance._use_legacy_cache_dir_if_possible(dataset_module)

    return builder_instance


def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    split: Optional[Union[str, Split]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    verification_mode: Optional[Union[VerificationMode, str]] = None,
    ignore_verifications="deprecated",
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    use_auth_token="deprecated",
    task="deprecated",
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    trust_remote_code: bool = None,
    **config_kwargs,
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load a dataset from the Hugging Face Hub, or a local dataset.

    You can find the list of datasets on the [Hub](https://huggingface.co/datasets) or with [`huggingface_hub.list_datasets`].

    A dataset is a directory that contains:

    - some data files in generic formats (JSON, CSV, Parquet, text, etc.).
    - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.

    Note that dataset scripts can also download and read data files from anywhere - in case your data files already exist online.

    This function does the following under the hood:

        1. Download and import in the library the dataset script from `path` if it's not already cached inside the library.

            If the dataset has no dataset script, then a generic dataset script is imported instead (JSON, CSV, Parquet, text, etc.)

            Dataset scripts are small python scripts that define dataset builders. They define the citation, info and format of the dataset,
            contain the path or URL to the original data files and the code to load examples from the original data files.

            You can find the complete list of datasets in the Datasets [Hub](https://huggingface.co/datasets).

        2. Run the dataset script which will:

            * Download the dataset file from the original URL (see the script) if it's not already available locally or cached.
            * Process and cache the dataset in typed Arrow tables for caching.

                Arrow table are arbitrarily long, typed tables which can store nested objects and be mapped to numpy/pandas/python generic types.
                They can be directly accessed from disk, loaded in RAM or even streamed over the web.

        3. Return a dataset built from the requested splits in `split` (default: all).

    It also allows to load a dataset from a local directory or a dataset repository on the Hugging Face Hub without dataset script.
    In this case, it automatically loads all the data files from the directory or the dataset repository.

    Args:

        path (`str`):
            Path or name of the dataset.
            Depending on `path`, the dataset builder that is used comes from a generic dataset script (JSON, CSV, Parquet, text etc.) or from the dataset script (a python file) inside the dataset directory.

            For local datasets:

            - if `path` is a local directory (containing data files only)
              -> load a generic dataset builder (csv, json, text etc.) based on the content of the directory
              e.g. `'./path/to/directory/with/my/csv/data'`.
            - if `path` is a local dataset script or a directory containing a local dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script
              e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.

            For datasets on the Hugging Face Hub (list all available datasets with [`huggingface_hub.list_datasets`])

            - if `path` is a dataset repository on the HF hub (containing data files only)
              -> load a generic dataset builder (csv, text etc.) based on the content of the repository
              e.g. `'username/dataset_name'`, a dataset repository on the HF hub containing your data files.
            - if `path` is a dataset repository on the HF hub with a dataset script (if the script has the same name as the directory)
              -> load the dataset builder from the dataset script in the dataset repository
              e.g. `glue`, `squad`, `'username/dataset_name'`, a dataset repository on the HF hub containing a dataset script `'dataset_name.py'`.

        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders (csv, text etc.) or the Hub datasets and `data_files` is `None`,
            the behavior is equal to passing `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`Split` or `str`):
            Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (`str`, *optional*):
            Directory to read/write data. Defaults to `"~/.cache/huggingface/datasets"`.
        features (`Features`, *optional*):
            Set the features type to use for this dataset.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
            Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

            <Added version="2.9.1"/>
        ignore_verifications (`bool`, defaults to `False`):
            Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).

            <Deprecated version="2.9.1">

            `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
            Please use `verification_mode` instead.

            </Deprecated>
        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.
        save_infos (`bool`, defaults to `False`):
            Save the dataset information (checksums/size/splits/...).
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>
        task (`str`):
            The task to prepare the dataset for during training and evaluation. Casts the dataset's [`Features`] to standardized column names and types as detailed in `datasets.tasks`.

            <Deprecated version="2.13.0">

            `task` was deprecated in version 2.13.0 and will be removed in 3.0.0.

            </Deprecated>
        streaming (`bool`, defaults to `False`):
            If set to `True`, don't download the data files. Instead, it streams the data progressively while
            iterating on the dataset. An [`IterableDataset`] or [`IterableDatasetDict`] is returned instead in this case.

            Note that streaming works for datasets that use data formats that support being iterated over like txt, csv, jsonl for example.
            Json files may be downloaded completely. Also streaming from remote zip or gzip files is supported but other compressed formats
            like rar and xz are not yet supported. The tgz format doesn't allow streaming.
        num_proc (`int`, *optional*, defaults to `None`):
            Number of processes when downloading and generating the dataset locally.
            Multiprocessing is disabled by default.

            <Added version="2.7.0"/>
        storage_options (`dict`, *optional*, defaults to `None`):
            **Experimental**. Key/value pairs to be passed on to the dataset file-system backend, if any.

            <Added version="2.11.0"/>
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Added version="2.16.0"/>

            <Changed version="2.20.0">

            `trust_remote_code` defaults to `False` if not specified.

            </Changed>

        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the `BuilderConfig`
            and used in the [`DatasetBuilder`].

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - if `split` is not `None`: the dataset requested,
        - if `split` is `None`, a [`~datasets.DatasetDict`] with each split.

        or [`IterableDataset`] or [`IterableDatasetDict`]: if `streaming=True`

        - if `split` is not `None`, the dataset is requested
        - if `split` is `None`, a [`~datasets.streaming.IterableDatasetDict`] with each split.

    Example:

    Load a dataset from the Hugging Face Hub:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('rotten_tomatoes', split='train')

    # Map data files to splits
    >>> data_files = {'train': 'train.csv', 'test': 'test.csv'}
    >>> ds = load_dataset('namespace/your_dataset_name', data_files=data_files)
    ```

    Load a local dataset:

    ```py
    # Load a CSV file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('csv', data_files='path/to/local/my_dataset.csv')

    # Load a JSON file
    >>> from datasets import load_dataset
    >>> ds = load_dataset('json', data_files='path/to/local/my_dataset.json')

    # Load from a local loading script
    >>> from datasets import load_dataset
    >>> ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')
    ```

    Load an [`~datasets.IterableDataset`]:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('rotten_tomatoes', split='train', streaming=True)
    ```

    Load an image dataset with the `ImageFolder` dataset builder:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset('imagefolder', data_dir='/path/to/images', split='train')
    ```
    """
    if use_auth_token != "deprecated":
        warnings.warn(
            "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
            "You can remove this warning by passing 'token=<use_auth_token>' instead.",
            FutureWarning,
        )
        token = use_auth_token
    if ignore_verifications != "deprecated":
        verification_mode = VerificationMode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
        warnings.warn(
            "'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.\n"
            f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
            FutureWarning,
        )
    if task != "deprecated":
        warnings.warn(
            "'task' was deprecated in version 2.13.0 and will be removed in 3.0.0.\n",
            FutureWarning,
        )
    else:
        task = None
    if data_files is not None and not data_files:
        raise ValueError(f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default).")
    if Path(path, config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError(
            "You are trying to load a dataset that was saved using `save_to_disk`. "
            "Please use `load_from_disk` instead."
        )

    if streaming and num_proc is not None:
        raise NotImplementedError(
            "Loading a streaming dataset in parallel with `num_proc` is not implemented. "
            "To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead."
        )

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode(
        (verification_mode or VerificationMode.BASIC_CHECKS) if not save_infos else VerificationMode.ALL_CHECKS
    )

    # Create a dataset builder
    builder_instance = load_dataset_builder(
        path=path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        cache_dir=cache_dir,
        features=features,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        storage_options=storage_options,
        trust_remote_code=trust_remote_code,
        _require_default_config_name=name is None,
        **config_kwargs,
    )

    # Return iterable dataset in case of streaming
    if streaming:
        return builder_instance.as_streaming_dataset(split=split)

    # Download and prepare data
    builder_instance.download_and_prepare(
        download_config=download_config,
        download_mode=download_mode,
        verification_mode=verification_mode,
        num_proc=num_proc,
        storage_options=storage_options,
    )

    # Build dataset for splits
    keep_in_memory = (
        keep_in_memory if keep_in_memory is not None else is_small_dataset(builder_instance.info.dataset_size)
    )
    ds = builder_instance.as_dataset(split=split, verification_mode=verification_mode, in_memory=keep_in_memory)
    # Rename and cast features to match task schema
    if task is not None:
        # To avoid issuing the same warning twice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            ds = ds.prepare_for_task(task)
    if save_infos:
        builder_instance._save_infos()

    return ds


def load_from_disk(
    dataset_path: PathLike,
    fs="deprecated",
    keep_in_memory: Optional[bool] = None,
    storage_options: Optional[dict] = None,
) -> Union[Dataset, DatasetDict]:
    """
    Loads a dataset that was previously saved using [`~Dataset.save_to_disk`] from a dataset directory, or
    from a filesystem using any implementation of `fsspec.spec.AbstractFileSystem`.

    Args:
        dataset_path (`path-like`):
            Path (e.g. `"dataset/train"`) or remote URI (e.g. `"s3://my-bucket/dataset/train"`)
            of the [`Dataset`] or [`DatasetDict`] directory where the dataset/dataset-dict will be
            loaded from.
        fs (`~filesystems.S3FileSystem` or `fsspec.spec.AbstractFileSystem`, *optional*):
            Instance of the remote filesystem used to download the files from.

            <Deprecated version="2.9.0">

            `fs` was deprecated in version 2.9.0 and will be removed in 3.0.0.
            Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`.

            </Deprecated>

        keep_in_memory (`bool`, defaults to `None`):
            Whether to copy the dataset in-memory. If `None`, the dataset
            will not be copied in-memory unless explicitly enabled by setting `datasets.config.IN_MEMORY_MAX_SIZE` to
            nonzero. See more details in the [improve performance](../cache#improve-performance) section.

        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the file-system backend, if any.

            <Added version="2.9.0"/>

    Returns:
        [`Dataset`] or [`DatasetDict`]:
        - If `dataset_path` is a path of a dataset directory: the dataset requested.
        - If `dataset_path` is a path of a dataset dict directory, a [`DatasetDict`] with each split.

    Example:

    ```py
    >>> from datasets import load_from_disk
    >>> ds = load_from_disk('path/to/dataset/directory')
    ```
    """
    if fs != "deprecated":
        warnings.warn(
            "'fs' was deprecated in favor of 'storage_options' in version 2.9.0 and will be removed in 3.0.0.\n"
            "You can remove this warning by passing 'storage_options=fs.storage_options' instead.",
            FutureWarning,
        )
        storage_options = fs.storage_options

    fs: fsspec.AbstractFileSystem
    fs, *_ = url_to_fs(dataset_path, **(storage_options or {}))
    if not fs.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    if fs.isfile(posixpath.join(dataset_path, config.DATASET_INFO_FILENAME)) and fs.isfile(
        posixpath.join(dataset_path, config.DATASET_STATE_JSON_FILENAME)
    ):
        return Dataset.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    elif fs.isfile(posixpath.join(dataset_path, config.DATASETDICT_JSON_FILENAME)):
        return DatasetDict.load_from_disk(dataset_path, keep_in_memory=keep_in_memory, storage_options=storage_options)
    else:
        raise FileNotFoundError(
            f"Directory {dataset_path} is neither a `Dataset` directory nor a `DatasetDict` directory."
        )
