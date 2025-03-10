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
"""DatasetBuilder base class."""

import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch

import fsspec
import pyarrow as pa
from fsspec.core import url_to_fs
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map

from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
    HF_GCP_BASE_URL,
    ArrowReader,
    DatasetNotOnHfGcsError,
    MissingFilesOnHfGcsError,
    ReadInstruction,
)
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
    is_remote_filesystem,
    rename,
)
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
    classproperty,
    convert_file_size_to_int,
    has_sufficient_disk_space,
    iflatmap_unordered,
    map_nested,
    memoize,
    size_str,
    temporary_assignment,
)
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list


if TYPE_CHECKING:
    from .load import DatasetModule


logger = logging.get_logger(__name__)


class InvalidConfigName(ValueError):
    pass


@dataclass
class BuilderConfig:
    """Base class for `DatasetBuilder` data configuration.

    `DatasetBuilder` subclasses with data configuration options should subclass
    `BuilderConfig` and add their own properties.

    Attributes:
        name (`str`, defaults to `default`):
            The name of the configuration.
        version (`Version` or `str`, defaults to `0.0.0`):
            The version of the configuration.
        data_dir (`str`, *optional*):
            Path to the directory containing the source data.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        description (`str`, *optional*):
            A human description of the configuration.
    """

    name: str = "default"
    version: Optional[Union[utils.Version, str]] = utils.Version("0.0.0")
    data_dir: Optional[str] = None
    data_files: Optional[Union[DataFilesDict, DataFilesPatternsDict]] = None
    description: Optional[str] = None

    def __post_init__(self):
        # The config name is used to name the cache directory.
        for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
            if invalid_char in self.name:
                raise InvalidConfigName(
                    f"Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{self.name}'. "
                    f"They could create issues when creating a directory for this config on Windows filesystem."
                )
        if self.data_files is not None and not isinstance(self.data_files, (DataFilesDict, DataFilesPatternsDict)):
            raise ValueError(f"Expected a DataFilesDict in data_files but got {self.data_files}")

    def __eq__(self, o):
        # we need to override the default dataclass __eq__ since it doesn't check for
        # other attributes that the ones of the signature.
        if set(self.__dict__.keys()) != set(o.__dict__.keys()):
            return False
        return all((k, getattr(self, k)) == (k, getattr(o, k)) for k in self.__dict__.keys())

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[Features] = None,
    ) -> str:
        """
        The config id is used to build the cache directory.
        By default it is equal to the config name.
        However the name of a config is not sufficient to have a unique identifier for the dataset being generated
        since it doesn't take into account:
        - the config kwargs that can be used to overwrite attributes
        - the custom features used to write the dataset
        - the data_files for json/text/csv/pandas datasets

        Therefore the config id is just the config name with an optional suffix based on these.
        """
        # Possibly add a suffix to the name to handle custom features/data_files/config_kwargs
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = config_kwargs.copy()
        # name and version are already used to build the cache directory
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        # data dir handling (when specified it points to the manually downloaded data):
        # it was previously ignored before the introduction of config id because we didn't want
        # to change the config name. Now it's fine to take it into account for the config id.
        # config_kwargs_to_add_to_suffix.pop("data_dir", None)
        if "data_dir" in config_kwargs_to_add_to_suffix:
            if config_kwargs_to_add_to_suffix["data_dir"] is None:
                config_kwargs_to_add_to_suffix.pop("data_dir", None)
            else:
                # canonicalize the data dir to avoid two paths to the same location having different
                # hashes
                data_dir = config_kwargs_to_add_to_suffix["data_dir"]
                data_dir = os.path.normpath(data_dir)
                config_kwargs_to_add_to_suffix["data_dir"] = data_dir
        if config_kwargs_to_add_to_suffix:
            # we don't care about the order of the kwargs
            config_kwargs_to_add_to_suffix = {
                k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)
            }
            if all(isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values()):
                suffix = ",".join(
                    str(k) + "=" + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items()
                )
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if custom_features is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            m.update(custom_features)
            suffix = m.hexdigest()

        if suffix:
            config_id = self.name + "-" + suffix
            if len(config_id) > config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name

    def _resolve_data_files(self, base_path: str, download_config: DownloadConfig) -> None:
        if isinstance(self.data_files, DataFilesPatternsDict):
            base_path = xjoin(base_path, self.data_dir) if self.data_dir else base_path
            self.data_files = self.data_files.resolve(base_path, download_config)


class DatasetBuilder:
    """Abstract base class for all datasets.

    `DatasetBuilder` has 3 key methods:

        - [`DatasetBuilder.info`]: Documents the dataset, including feature
          names, types, shapes, version, splits, citation, etc.
        - [`DatasetBuilder.download_and_prepare`]: Downloads the source data
          and writes it to disk.
        - [`DatasetBuilder.as_dataset`]: Generates a [`Dataset`].

    Some `DatasetBuilder`s expose multiple variants of the
    dataset by defining a [`BuilderConfig`] subclass and accepting a
    config object (or name) on construction. Configurable datasets expose a
    pre-defined set of configurations in [`DatasetBuilder.builder_configs`].

    Args:
        cache_dir (`str`, *optional*):
            Directory to cache data. Defaults to `"~/.cache/huggingface/datasets"`.
        dataset_name (`str`, *optional*):
            Name of the dataset, if different from the builder name. Useful for packaged builders
            like csv, imagefolder, audiofolder, etc. to reflect the difference between datasets
            that use the same packaged builder.
        config_name (`str`, *optional*):
            Name of the dataset configuration.
            It affects the data generated on disk. Different configurations will have their own subdirectories and
            versions.
            If not provided, the default configuration is used (if it exists).

            <Added version="2.3.0">

            Parameter `name` was renamed to `config_name`.

            </Added>
        hash (`str`, *optional*):
            Hash specific to the dataset code. Used to update the caching directory when the
            dataset loading script code is updated (to avoid reusing old data).
            The typical caching directory (defined in `self._relative_data_dir`) is `name/version/hash/`.
        base_path (`str`, *optional*):
            Base path for relative paths that are used to download files.
            This can be a remote URL.
        features ([`Features`], *optional*):
            Features types to use with this dataset.
            It can be used to change the [`Features`] types of a dataset, for example.
        token (`str` or `bool`, *optional*):
            String or boolean to use as Bearer token for remote files on the
            Datasets Hub. If `True`, will get token from `"~/.huggingface"`.
        repo_id (`str`, *optional*):
            ID of the dataset repository.
            Used to distinguish builders with the same name but not coming from the same namespace, for example "squad"
            and "lhoestq/squad" repo IDs. In the latter, the builder name would be "lhoestq___squad".
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
            For builders like "csv" or "json" that need the user to specify data files. They can be either
            local or remote files. For convenience, you can use a `DataFilesDict`.
        data_dir (`str`, *optional*):
            Path to directory containing source data file(s).
            Use only if `data_files` is not passed, in which case it is equivalent to passing
            `os.path.join(data_dir, "**")` as `data_files`.
            For builders that require manual download, it must be the path to the local directory containing the
            manually downloaded data.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        writer_batch_size (`int`, *optional*):
            Batch size used by the ArrowWriter.
            It defines the number of samples that are kept in memory before writing them
            and also the length of the arrow chunks.
            None means that the ArrowWriter will use its default value.
        name (`str`): Configuration name for the dataset.

            <Deprecated version="2.3.0">

            Use `config_name` instead.

            </Deprecated>

        **config_kwargs (additional keyword arguments): Keyword arguments to be passed to the corresponding builder
            configuration class, set on the class attribute [`DatasetBuilder.BUILDER_CONFIG_CLASS`]. The builder
            configuration class is [`BuilderConfig`] or a subclass of it.
    """

    # Default version
    VERSION = None  # Default version set in BuilderConfig

    # Class for the builder config.
    BUILDER_CONFIG_CLASS = BuilderConfig

    # Named configurations that modify the data generated by download_and_prepare.
    BUILDER_CONFIGS = []

    # Optional default config name to be used when name is None
    DEFAULT_CONFIG_NAME = None

    # Default batch size used by the ArrowWriter
    # It defines the number of samples that are kept in memory before writing them
    # and also the length of the arrow chunks
    # None means that the ArrowWriter will use its default value
    DEFAULT_WRITER_BATCH_SIZE = None

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[DatasetInfo] = None,
        features: Optional[Features] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token="deprecated",
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        name="deprecated",
        **config_kwargs,
    ):
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'token={use_auth_token}' instead.",
                FutureWarning,
            )
            token = use_auth_token
        if name != "deprecated":
            warnings.warn(
                "Parameter 'name' was renamed to 'config_name' in version 2.3.0 and will be removed in 3.0.0.",
                category=FutureWarning,
            )
            config_name = name
        # DatasetBuilder name
        self.name: str = camelcase_to_snakecase(self.__module__.split(".")[-1])
        self.hash: Optional[str] = hash
        self.base_path = base_path
        self.token = token
        # For backwards compatibility (e.g. if accessed in a dataset script)
        self.use_auth_token = token
        self.repo_id = repo_id
        self.storage_options = storage_options or {}
        self.dataset_name = camelcase_to_snakecase(dataset_name) if dataset_name else self.name
        self._writer_batch_size = writer_batch_size or self.DEFAULT_WRITER_BATCH_SIZE

        if data_files is not None and not isinstance(data_files, DataFilesDict):
            data_files = DataFilesDict.from_patterns(
                sanitize_patterns(data_files),
                base_path=base_path,
                download_config=DownloadConfig(token=token, storage_options=self.storage_options),
            )

        # Prepare config: DatasetConfig contains name, version and description but can be extended by each dataset
        if "features" in inspect.signature(self.BUILDER_CONFIG_CLASS.__init__).parameters and features is not None:
            config_kwargs["features"] = features
        if data_files is not None:
            config_kwargs["data_files"] = data_files
        if data_dir is not None:
            config_kwargs["data_dir"] = data_dir
        self.config_kwargs = config_kwargs
        self.config, self.config_id = self._create_builder_config(
            config_name=config_name,
            custom_features=features,
            **config_kwargs,
        )

        # prepare info: DatasetInfo are a standardized dataclass across all datasets
        # Prefill datasetinfo
        if info is None:
            # TODO FOR PACKAGED MODULES IT IMPORTS DATA FROM src/packaged_modules which doesn't make sense
            info = self.get_exported_dataset_info()
            info.update(self._info())
        info.builder_name = self.name
        info.dataset_name = self.dataset_name
        info.config_name = self.config.name
        info.version = self.config.version
        self.info = info
        # update info with user specified infos
        if features is not None:
            self.info.features = features

        # Prepare data dirs:
        # cache_dir can be a remote bucket on GCS or S3 (when using BeamBasedBuilder for distributed data processing)
        self._cache_dir_root = str(cache_dir or config.HF_DATASETS_CACHE)
        self._cache_dir_root = (
            self._cache_dir_root if is_remote_url(self._cache_dir_root) else os.path.expanduser(self._cache_dir_root)
        )
        self._cache_downloaded_dir = (
            posixpath.join(self._cache_dir_root, config.DOWNLOADED_DATASETS_DIR)
            if cache_dir
            else str(config.DOWNLOADED_DATASETS_PATH)
        )
        self._cache_downloaded_dir = (
            self._cache_downloaded_dir
            if is_remote_url(self._cache_downloaded_dir)
            else os.path.expanduser(self._cache_downloaded_dir)
        )

        # In case there exists a legacy cache directory
        self._legacy_relative_data_dir = None

        self._cache_dir = self._build_cache_dir()
        if not is_remote_url(self._cache_dir_root):
            os.makedirs(self._cache_dir_root, exist_ok=True)
            lock_path = os.path.join(
                self._cache_dir_root, Path(self._cache_dir).as_posix().replace("/", "_") + ".lock"
            )
            with FileLock(lock_path):
                if os.path.exists(self._cache_dir):  # check if data exist
                    if len(os.listdir(self._cache_dir)) > 0:
                        if os.path.exists(os.path.join(self._cache_dir, config.DATASET_INFO_FILENAME)):
                            logger.info("Overwrite dataset info from restored data version if exists.")
                            self.info = DatasetInfo.from_directory(self._cache_dir)
                    else:  # dir exists but no data, remove the empty dir as data aren't available anymore
                        logger.warning(
                            f"Old caching folder {self._cache_dir} for dataset {self.dataset_name} exists but no data were found. Removing it. "
                        )
                        os.rmdir(self._cache_dir)

        # Store in the cache by default unless the user specifies a custom output_dir to download_and_prepare
        self._output_dir = self._cache_dir
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem("file")

        # Set download manager
        self.dl_manager = None

        # Set to True by "datasets-cli test" to generate file checksums for (deprecated) dataset_infos.json independently of verification_mode value.
        self._record_infos = False

        # Set in `.download_and_prepare` once the format of the generated dataset is known
        self._file_format = None

        # Enable streaming (e.g. it patches "open" to work with remote files)
        extend_dataset_builder_for_streaming(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        # Re-enable streaming, since patched functions are not kept when pickling
        extend_dataset_builder_for_streaming(self)

    # Must be set for datasets that use 'data_dir' functionality - the ones
    # that require users to do additional steps to download the data
    # (this is usually due to some external regulations / rules).
    # This field should contain a string with user instructions, including
    # the list of files that should be present. It will be
    # displayed in the dataset documentation.
    @property
    def manual_download_instructions(self) -> Optional[str]:
        return None

    def _check_legacy_cache(self) -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{builder_name} from 2.13"""
        if (
            self.__module__.startswith("datasets.")
            and not is_remote_url(self._cache_dir_root)
            and self.config.name == "default"
        ):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            config_name = self.repo_id.replace("/", "--") if self.repo_id is not None else self.dataset_name
            config_id = config_name + self.config_id[len(self.config.name) :]
            hash = _PACKAGED_DATASETS_MODULES.get(self.name, "missing")[1]
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    def _check_legacy_cache2(self, dataset_module: "DatasetModule") -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{dataset_name}/{config_name}-xxx from 2.14 and 2.15"""
        if (
            self.__module__.startswith("datasets.")
            and not is_remote_url(self._cache_dir_root)
            and not (set(self.config_kwargs) - {"data_files", "data_dir"})
        ):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES_2_15_HASHES
            from .utils._dill import Pickler

            def update_hash_with_config_parameters(hash: str, config_parameters: dict) -> str:
                """
                Used to update hash of packaged modules which is used for creating unique cache directories to reflect
                different config parameters which are passed in metadata from readme.
                """
                params_to_exclude = {"config_name", "version", "description"}
                params_to_add_to_hash = {
                    param: value
                    for param, value in sorted(config_parameters.items())
                    if param not in params_to_exclude
                }
                m = Hasher()
                m.update(hash)
                m.update(params_to_add_to_hash)
                return m.hexdigest()

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            with patch.object(Pickler, "_legacy_no_dict_keys_sorting", True):
                config_id = self.config.name + "-" + Hasher.hash({"data_files": self.config.data_files})
            hash = _PACKAGED_DATASETS_MODULES_2_15_HASHES.get(self.name, "missing")
            if (
                dataset_module.builder_configs_parameters.metadata_configs
                and self.config.name in dataset_module.builder_configs_parameters.metadata_configs
            ):
                hash = update_hash_with_config_parameters(
                    hash, dataset_module.builder_configs_parameters.metadata_configs[self.config.name]
                )
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    @classmethod
    def get_all_exported_dataset_infos(cls) -> DatasetInfosDict:
        """Empty dict if doesn't exist

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> ds_builder = load_dataset_builder('rotten_tomatoes')
        >>> ds_builder.get_all_exported_dataset_infos()
        {'default': DatasetInfo(description="Movie Review Dataset.\nThis is a dataset of containing 5,331 positive and 5,331 negative processed\nsentences from Rotten Tomatoes movie reviews. This data was first used in Bo\nPang and Lillian Lee, ``Seeing stars: Exploiting class relationships for\nsentiment categorization with respect to rating scales.'', Proceedings of the\nACL, 2005.\n", citation='@InProceedings{Pang+Lee:05a,\n  author =       {Bo Pang and Lillian Lee},\n  title =        {Seeing stars: Exploiting class relationships for sentiment\n                  categorization with respect to rating scales},\n  booktitle =    {Proceedings of the ACL},\n  year =         2005\n}\n', homepage='http://www.cs.cornell.edu/people/pabo/movie-review-data/', license='', features={'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}, post_processed=None, supervised_keys=SupervisedKeysData(input='', output=''), task_templates=[TextClassification(task='text-classification', text_column='text', label_column='label')], builder_name='rotten_tomatoes_movie_review', config_name='default', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=1074810, num_examples=8530, dataset_name='rotten_tomatoes_movie_review'), 'validation': SplitInfo(name='validation', num_bytes=134679, num_examples=1066, dataset_name='rotten_tomatoes_movie_review'), 'test': SplitInfo(name='test', num_bytes=135972, num_examples=1066, dataset_name='rotten_tomatoes_movie_review')}, download_checksums={'https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz': {'num_bytes': 487770, 'checksum': 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'}}, download_size=487770, post_processing_size=None, dataset_size=1345461, size_in_bytes=1833231)}
        ```
        """
        return DatasetInfosDict.from_directory(cls.get_imported_module_dir())

    def get_exported_dataset_info(self) -> DatasetInfo:
        """Empty `DatasetInfo` if doesn't exist

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> ds_builder = load_dataset_builder('rotten_tomatoes')
        >>> ds_builder.get_exported_dataset_info()
        DatasetInfo(description="Movie Review Dataset.\nThis is a dataset of containing 5,331 positive and 5,331 negative processed\nsentences from Rotten Tomatoes movie reviews. This data was first used in Bo\nPang and Lillian Lee, ``Seeing stars: Exploiting class relationships for\nsentiment categorization with respect to rating scales.'', Proceedings of the\nACL, 2005.\n", citation='@InProceedings{Pang+Lee:05a,\n  author =       {Bo Pang and Lillian Lee},\n  title =        {Seeing stars: Exploiting class relationships for sentiment\n                  categorization with respect to rating scales},\n  booktitle =    {Proceedings of the ACL},\n  year =         2005\n}\n', homepage='http://www.cs.cornell.edu/people/pabo/movie-review-data/', license='', features={'text': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None)}, post_processed=None, supervised_keys=SupervisedKeysData(input='', output=''), task_templates=[TextClassification(task='text-classification', text_column='text', label_column='label')], builder_name='rotten_tomatoes_movie_review', config_name='default', version=1.0.0, splits={'train': SplitInfo(name='train', num_bytes=1074810, num_examples=8530, dataset_name='rotten_tomatoes_movie_review'), 'validation': SplitInfo(name='validation', num_bytes=134679, num_examples=1066, dataset_name='rotten_tomatoes_movie_review'), 'test': SplitInfo(name='test', num_bytes=135972, num_examples=1066, dataset_name='rotten_tomatoes_movie_review')}, download_checksums={'https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz': {'num_bytes': 487770, 'checksum': 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'}}, download_size=487770, post_processing_size=None, dataset_size=1345461, size_in_bytes=1833231)
        ```
        """
        return self.get_all_exported_dataset_infos().get(self.config.name, DatasetInfo())

    def _create_builder_config(
        self, config_name=None, custom_features=None, **config_kwargs
    ) -> Tuple[BuilderConfig, str]:
        """Create and validate BuilderConfig object as well as a unique config id for this config.
        Raises ValueError if there are multiple builder configs and config_name and DEFAULT_CONFIG_NAME are None.
        config_kwargs override the defaults kwargs in config
        """
        builder_config = None

        # try default config
        if config_name is None and self.BUILDER_CONFIGS:
            if self.DEFAULT_CONFIG_NAME is not None:
                builder_config = self.builder_configs.get(self.DEFAULT_CONFIG_NAME)
                logger.info(f"No config specified, defaulting to: {self.dataset_name}/{builder_config.name}")
            else:
                if len(self.BUILDER_CONFIGS) > 1:
                    if not config_kwargs:
                        example_of_usage = (
                            f"load_dataset('{self.repo_id or self.dataset_name}', '{self.BUILDER_CONFIGS[0].name}')"
                        )
                        raise ValueError(
                            "Config name is missing."
                            f"\nPlease pick one among the available configs: {list(self.builder_configs.keys())}"
                            + f"\nExample of usage:\n\t`{example_of_usage}`"
                        )
                else:
                    builder_config = self.BUILDER_CONFIGS[0]
                    logger.info(
                        f"No config specified, defaulting to the single config: {self.dataset_name}/{builder_config.name}"
                    )

        # try to get config by name
        if isinstance(config_name, str):
            builder_config = self.builder_configs.get(config_name)
            if builder_config is None and self.BUILDER_CONFIGS:
                raise ValueError(
                    f"BuilderConfig '{config_name}' not found. Available: {list(self.builder_configs.keys())}"
                )

        # if not using an existing config, then create a new config on the fly
        if not builder_config:
            if config_name is not None:
                config_kwargs["name"] = config_name
            elif self.DEFAULT_CONFIG_NAME and not config_kwargs:
                # Use DEFAULT_CONFIG_NAME only if no config_kwargs are passed
                config_kwargs["name"] = self.DEFAULT_CONFIG_NAME
            if "version" not in config_kwargs and hasattr(self, "VERSION") and self.VERSION:
                config_kwargs["version"] = self.VERSION
            builder_config = self.BUILDER_CONFIG_CLASS(**config_kwargs)

        # otherwise use the config_kwargs to overwrite the attributes
        else:
            builder_config = copy.deepcopy(builder_config) if config_kwargs else builder_config
            for key, value in config_kwargs.items():
                if value is not None:
                    if not hasattr(builder_config, key):
                        raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                    setattr(builder_config, key, value)

        if not builder_config.name:
            raise ValueError(f"BuilderConfig must have a name, got {builder_config.name}")

        # resolve data files if needed
        builder_config._resolve_data_files(
            base_path=self.base_path,
            download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),
        )

        # compute the config id that is going to be used for caching
        config_id = builder_config.create_config_id(
            config_kwargs,
            custom_features=custom_features,
        )
        is_custom = (config_id not in self.builder_configs) and config_id != "default"
        if is_custom:
            logger.info(f"Using custom data configuration {config_id}")
        else:
            if (
                builder_config.name in self.builder_configs
                and builder_config != self.builder_configs[builder_config.name]
            ):
                raise ValueError(
                    "Cannot name a custom BuilderConfig the same as an available "
                    f"BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}"
                )
            if not builder_config.version:
                raise ValueError(f"BuilderConfig {builder_config.name} must have a version")

        return builder_config, config_id

    @classproperty
    @classmethod
    @memoize()
    def builder_configs(cls) -> Dict[str, BuilderConfig]:
        """Dictionary of pre-defined configurations for this builder class."""
        configs = {config.name: config for config in cls.BUILDER_CONFIGS}
        if len(configs) != len(cls.BUILDER_CONFIGS):
            names = [config.name for config in cls.BUILDER_CONFIGS]
            raise ValueError(f"Names in BUILDER_CONFIGS must not be duplicated. Got {names}")
        return configs

    @property
    def cache_dir(self):
        return self._cache_dir

    def _use_legacy_cache_dir_if_possible(self, dataset_module: "DatasetModule"):
        # Check for the legacy cache directory template (datasets<3.0.0)
        self._legacy_relative_data_dir = (
            self._check_legacy_cache2(dataset_module) or self._check_legacy_cache() or None
        )
        self._cache_dir = self._build_cache_dir()
        self._output_dir = self._cache_dir

    def _relative_data_dir(self, with_version=True, with_hash=True) -> str:
        """Relative path of this dataset in cache_dir:
        Will be:
            self.dataset_name/self.config.version/self.hash/
        or if a repo_id with a namespace has been specified:
            self.namespace___self.dataset_name/self.config.version/self.hash/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
        if self._legacy_relative_data_dir is not None and with_version and with_hash:
            return self._legacy_relative_data_dir

        namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
        builder_data_dir = self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}"
        builder_data_dir = posixpath.join(builder_data_dir, self.config_id)
        if with_version:
            builder_data_dir = posixpath.join(builder_data_dir, str(self.config.version))
        if with_hash and self.hash and isinstance(self.hash, str):
            builder_data_dir = posixpath.join(builder_data_dir, self.hash)
        return builder_data_dir

    def _build_cache_dir(self):
        """Return the data directory for the current version."""
        builder_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=False))
        version_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=True))

        def _other_versions_on_disk():
            """Returns previous versions on disk."""
            if not os.path.exists(builder_data_dir):
                return []

            version_dirnames = []
            for dir_name in os.listdir(builder_data_dir):
                try:
                    version_dirnames.append((utils.Version(dir_name), dir_name))
                except ValueError:  # Invalid version (ex: incomplete data dir)
                    pass
            version_dirnames.sort(reverse=True)
            return version_dirnames

        # Check and warn if other versions exist
        if not is_remote_url(builder_data_dir):
            version_dirs = _other_versions_on_disk()
            if version_dirs:
                other_version = version_dirs[0][0]
                if other_version != self.config.version:
                    warn_msg = (
                        f"Found a different version {str(other_version)} of dataset {self.dataset_name} in "
                        f"cache_dir {self._cache_dir_root}. Using currently defined version "
                        f"{str(self.config.version)}."
                    )
                    logger.warning(warn_msg)

        return version_data_dir

    @abc.abstractmethod
    def _info(self) -> DatasetInfo:
        """Construct the DatasetInfo object. See `DatasetInfo` for details.

        Warning: This function is only called once and the result is cached for all
        following .info() calls.

        Returns:
            info: (DatasetInfo) The dataset information
        """
        raise NotImplementedError

    @classmethod
    def get_imported_module_dir(cls):
        """Return the path of the module of this class or subclass."""
        return os.path.dirname(inspect.getfile(inspect.getmodule(cls)))

    def _rename(self, src: str, dst: str):
        rename(self._fs, src, dst)

    def download_and_prepare(
        self,
        output_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        ignore_verifications="deprecated",
        try_from_hf_gcs="deprecated",
        dl_manager: Optional[DownloadManager] = None,
        base_path: Optional[str] = None,
        use_auth_token="deprecated",
        file_format: str = "arrow",
        max_shard_size: Optional[Union[int, str]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **download_and_prepare_kwargs,
    ):
        """Downloads and prepares dataset for reading.

        Args:
            output_dir (`str`, *optional*):
                Output directory for the dataset.
                Default to this builder's `cache_dir`, which is inside `~/.cache/huggingface/datasets` by default.

                <Added version="2.5.0"/>
            download_config (`DownloadConfig`, *optional*):
                Specific download configuration parameters.
            download_mode ([`DownloadMode`] or `str`, *optional*):
                Select the download/generate mode, default to `REUSE_DATASET_IF_EXISTS`.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            try_from_hf_gcs (`bool`):
                If `True`, it will try to download the already prepared dataset from the HF Google cloud storage.

                <Deprecated version="2.16.0">

                `try_from_hf_gcs` was deprecated in version 2.16.0 and will be removed in 3.0.0.
                Host the processed files on the Hugging Face Hub instead.

                </Deprecated>
            dl_manager (`DownloadManager`, *optional*):
                Specific `DownloadManger` to use.
            base_path (`str`, *optional*):
                Base path for relative paths that are used to download files. This can be a remote url.
                If not specified, the value of the `base_path` attribute (`self.base_path`) will be used instead.
            use_auth_token (`Union[str, bool]`, *optional*):
                Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
                If True, or not specified, will get token from ~/.huggingface.

                <Deprecated version="2.7.1">

                Pass `use_auth_token` to `load_dataset_builder` instead.

                </Deprecated>
            file_format (`str`, *optional*):
                Format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
                If the format is "parquet", then image and audio data are embedded into the Parquet files instead of pointing to local files.

                <Added version="2.5.0"/>
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.

                <Added version="2.5.0"/>
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the caching file-system backend, if any.

                <Added version="2.5.0"/>
            **download_and_prepare_kwargs (additional keyword arguments): Keyword arguments.

        Example:

        Download and prepare the dataset as Arrow files that can be loaded as a Dataset using `builder.as_dataset()`:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare()
        ```

        Download and prepare the dataset as sharded Parquet files locally:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare("./output_dir", file_format="parquet")
        ```

        Download and prepare the dataset as sharded Parquet files in a cloud storage:

        ```py
        >>> from datasets import load_dataset_builder
        >>> storage_options = {"key": aws_access_key_id, "secret": aws_secret_access_key}
        >>> builder = load_dataset_builder("rotten_tomatoes")
        >>> builder.download_and_prepare("s3://my-bucket/my_rotten_tomatoes", storage_options=storage_options, file_format="parquet")
        ```
        """
        if ignore_verifications != "deprecated":
            verification_mode = VerificationMode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(
                "'ignore_verifications' was deprecated in favor of 'verification_mode' in version 2.9.1 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
                FutureWarning,
            )
        if use_auth_token != "deprecated":
            warnings.warn(
                "'use_auth_token' was deprecated in version 2.7.1 and will be removed in 3.0.0. Pass `token` to `load_dataset_builder` instead.",
                FutureWarning,
            )
            token = use_auth_token
        else:
            token = self.token

        if try_from_hf_gcs != "deprecated":
            warnings.warn(
                "'try_from_hf_gcs' was deprecated in version 2.16.0 and will be removed in 3.0.0.",
                FutureWarning,
            )
        else:
            try_from_hf_gcs = False

        output_dir = output_dir if output_dir is not None else self._cache_dir
        # output_dir can be a remote bucket on GCS or S3 (when using BeamBasedBuilder for distributed data processing)
        fs, output_dir = url_to_fs(output_dir, **(storage_options or {}))
        self._fs = fs
        self._output_dir = output_dir if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(output_dir)

        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        base_path = base_path if base_path is not None else self.base_path

        if file_format is not None and file_format not in ["arrow", "parquet"]:
            raise ValueError(f"Unsupported file_format: {file_format}. Expected 'arrow' or 'parquet'")
        self._file_format = file_format

        if self._fs._strip_protocol(self._output_dir) == "":
            # We don't support the root directory, because it has no dirname,
            # and we need a dirname to use a <dirname>.incomplete directory
            # when the dataset is being written
            raise RuntimeError(
                f"Unable to download and prepare the dataset at the root {self._output_dir}. "
                f"Please specify a subdirectory, e.g. '{self._output_dir + self.dataset_name}'"
            )

        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(
                    cache_dir=self._cache_downloaded_dir,
                    force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    use_etag=False,
                    num_proc=num_proc,
                    token=token,
                    storage_options=self.storage_options,
                )  # We don't use etag for data files to speed up the process

            dl_manager = DownloadManager(
                dataset_name=self.dataset_name,
                download_config=download_config,
                data_dir=self.config.data_dir,
                base_path=base_path,
                record_checksums=(self._record_infos or verification_mode == VerificationMode.ALL_CHECKS),
            )

        is_local = not is_remote_filesystem(self._fs)

        if (
            isinstance(dl_manager, MockDownloadManager)
            or not is_local
            or file_format != "arrow"
            or max_shard_size is not None
        ):
            try_from_hf_gcs = False
        self.dl_manager = dl_manager

        # Prevent parallel local disk operations
        if is_local:
            # Create parent directory of the output_dir to put the lock file in there
            Path(self._output_dir).parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._output_dir + "_builder.lock"

        # File locking only with local paths; no file locking on GCS or S3
        with FileLock(lock_path) if is_local else contextlib.nullcontext():
            # Check if the data already exists
            data_exists = self._fs.exists(posixpath.join(self._output_dir, config.DATASET_INFO_FILENAME))
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                logger.info(f"Found cached dataset {self.dataset_name} ({self._output_dir})")
                # We need to update the info in case some splits were added in the meantime
                # for example when calling load_dataset from multiple workers.
                self.info = self._load_info()
                self.download_post_processing_resources(dl_manager)
                return

            logger.info(f"Generating dataset {self.dataset_name} ({self._output_dir})")
            if is_local:  # if cache dir is local, check for available space
                if not has_sufficient_disk_space(
                    self.info.size_in_bytes or 0, directory=Path(self._output_dir).parent
                ):
                    raise OSError(
                        f"Not enough disk space. Needed: {size_str(self.info.size_in_bytes or 0)} (download: {size_str(self.info.download_size or 0)}, generated: {size_str(self.info.dataset_size or 0)}, post-processed: {size_str(self.info.post_processing_size or 0)})"
                    )

            @contextlib.contextmanager
            def incomplete_dir(dirname):
                """Create temporary dir for dirname and rename on exit."""
                if not is_local:
                    self._fs.makedirs(dirname, exist_ok=True)
                    yield dirname
                else:
                    tmp_dir = dirname + ".incomplete"
                    os.makedirs(tmp_dir, exist_ok=True)
                    try:
                        yield tmp_dir
                        if os.path.isdir(dirname):
                            shutil.rmtree(dirname)
                        # LocalFileSystem.mv does copy + rm, it is more efficient to simply rename a local directory
                        shutil.move(tmp_dir, dirname)
                    finally:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)

            # Print is intentional: we want this to always go to stdout so user has
            # information needed to cancel download/preparation if needed.
            # This comes right before the progress bar.
            if self.info.size_in_bytes:
                logger.info(
                    f"Downloading and preparing dataset {self.dataset_name}/{self.config.name} "
                    f"(download: {size_str(self.info.download_size)}, generated: {size_str(self.info.dataset_size)}, "
                    f"post-processed: {size_str(self.info.post_processing_size)}, "
                    f"total: {size_str(self.info.size_in_bytes)}) to {self._output_dir}..."
                )
            else:
                _dest = self._fs._strip_protocol(self._output_dir) if is_local else self._output_dir
                logger.info(f"Downloading and preparing dataset {self.dataset_name}/{self.config.name} to {_dest}...")

            self._check_manual_download(dl_manager)

            # Create a tmp dir and rename to self._output_dir on successful exit.
            with incomplete_dir(self._output_dir) as tmp_output_dir:
                # Temporarily assign _output_dir to tmp_data_dir to avoid having to forward
                # it to every sub function.
                with temporary_assignment(self, "_output_dir", tmp_output_dir):
                    # Try to download the already prepared dataset files
                    downloaded_from_gcs = False
                    if try_from_hf_gcs:
                        try:
                            self._download_prepared_from_hf_gcs(dl_manager.download_config)
                            downloaded_from_gcs = True
                        except (DatasetNotOnHfGcsError, MissingFilesOnHfGcsError):
                            logger.info("Dataset not on Hf google storage. Downloading and preparing it from source")
                        except ConnectionError:
                            logger.warning("HF google storage unreachable. Downloading and preparing it from source")
                    if not downloaded_from_gcs:
                        prepare_split_kwargs = {"file_format": file_format}
                        if max_shard_size is not None:
                            prepare_split_kwargs["max_shard_size"] = max_shard_size
                        if num_proc is not None:
                            prepare_split_kwargs["num_proc"] = num_proc
                        self._download_and_prepare(
                            dl_manager=dl_manager,
                            verification_mode=verification_mode,
                            **prepare_split_kwargs,
                            **download_and_prepare_kwargs,
                        )
                    # Sync info
                    self.info.dataset_size = sum(split.num_bytes for split in self.info.splits.values())
                    self.info.download_checksums = dl_manager.get_recorded_sizes_checksums()
                    self.info.size_in_bytes = self.info.dataset_size + self.info.download_size
                    # Save info
                    self._save_info()

            # Download post processing resources
            self.download_post_processing_resources(dl_manager)

            logger.info(
                f"Dataset {self.dataset_name} downloaded and prepared to {self._output_dir}. "
                f"Subsequent calls will reuse this data."
            )

    def _check_manual_download(self, dl_manager):
        if self.manual_download_instructions is not None and dl_manager.manual_dir is None:
            raise ManualDownloadError(
                textwrap.dedent(
                    f"""\
                    The dataset {self.dataset_name} with config {self.config.name} requires manual data.
                    Please follow the manual download instructions:
                     {self.manual_download_instructions}
                    Manual data can be loaded with:
                     datasets.load_dataset("{self.repo_id or self.dataset_name}", data_dir="<path/to/manual/data>")"""
                )
            )

    def _download_prepared_from_hf_gcs(self, download_config: DownloadConfig):
        relative_data_dir = self._relative_data_dir(with_version=True, with_hash=False)
        reader = ArrowReader(self._output_dir, self.info)
        # use reader instructions to download the right files
        reader.download_from_hf_gcs(download_config, relative_data_dir)
        downloaded_info = DatasetInfo.from_directory(self._output_dir)
        self.info.update(downloaded_info)
        # download post processing resources
        remote_cache_dir = HF_GCP_BASE_URL + "/" + relative_data_dir.replace(os.sep, "/")
        for split in self.info.splits:
            for resource_file_name in self._post_processing_resources(split).values():
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
                try:
                    resource_path = cached_path(remote_cache_dir + "/" + resource_file_name)
                    shutil.move(resource_path, os.path.join(self._output_dir, resource_file_name))
                except ConnectionError:
                    logger.info(f"Couldn't download resourse file {resource_file_name} from Hf google storage.")
        logger.info("Dataset downloaded from Hf google storage.")

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs):
        """Downloads and prepares dataset for reading.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required data and generate
        the pre-processed datasets files.

        Args:
            dl_manager ([`DownloadManager`]):
                `DownloadManager` used to download and cache data.
            verification_mode ([`VerificationMode`]):
                if `ALL_CHECKS`, perform all the verifications including checksums.
                if `BASIC_CHECKS`, do not perform checksums, only perform split tests.
                if `NO_CHECKS`, do not perform any verification.
            prepare_split_kwargs: Additional options, such as `file_format`, `max_shard_size`
        """
        # Generating data for all splits
        split_dict = SplitDict(dataset_name=self.dataset_name)
        split_generators_kwargs = self._make_split_generators_kwargs(prepare_split_kwargs)
        split_generators = self._split_generators(dl_manager, **split_generators_kwargs)

        # Checksums verification
        if verification_mode == VerificationMode.ALL_CHECKS and dl_manager.record_checksums:
            verify_checksums(
                self.info.download_checksums, dl_manager.get_recorded_sizes_checksums(), "dataset source files"
            )

        # Build splits
        for split_generator in split_generators:
            if str(split_generator.split_info.name).lower() == "all":
                raise ValueError(
                    "`all` is a special split keyword corresponding to the "
                    "union of all splits, so cannot be used as key in "
                    "._split_generator()."
                )

            logger.info(f"Generating {split_generator.split_info.name} split")
            split_dict.add(split_generator.split_info)

            try:
                # Prepare split will record examples associated to the split
                self._prepare_split(split_generator, **prepare_split_kwargs)
            except OSError as e:
                raise OSError(
                    "Cannot find data file. "
                    + (self.manual_download_instructions or "")
                    + "\nOriginal error:\n"
                    + str(e)
                ) from None
            # If check_duplicates is set to True , then except DuplicatedKeysError
            except DuplicatedKeysError as e:
                raise DuplicatedKeysError(
                    e.key,
                    e.duplicate_key_indices,
                    fix_msg=f"To avoid duplicate keys, please fix the dataset script {self.name}.py",
                ) from None
            dl_manager.manage_extracted_files()

        if verification_mode == VerificationMode.BASIC_CHECKS or verification_mode == VerificationMode.ALL_CHECKS:
            verify_splits(self.info.splits, split_dict)

        # Update the info object with the splits.
        self.info.splits = split_dict
        self.info.download_size = dl_manager.downloaded_size

    def download_post_processing_resources(self, dl_manager):
        for split in self.info.splits or []:
            for resource_name, resource_file_name in self._post_processing_resources(split).items():
                if not not is_remote_filesystem(self._fs):
                    raise NotImplementedError(f"Post processing is not supported on filesystem {self._fs}")
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
                resource_path = os.path.join(self._output_dir, resource_file_name)
                if not os.path.exists(resource_path):
                    downloaded_resource_path = self._download_post_processing_resources(
                        split, resource_name, dl_manager
                    )
                    if downloaded_resource_path:
                        logger.info(f"Downloaded post-processing resource {resource_name} as {resource_file_name}")
                        shutil.move(downloaded_resource_path, resource_path)

    def _load_info(self) -> DatasetInfo:
        return DatasetInfo.from_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_info(self):
        file_lock = (
            FileLock(self._output_dir + "_info.lock")
            if not is_remote_filesystem(self._fs)
            else contextlib.nullcontext()
        )
        with file_lock:
            self.info.write_to_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_infos(self):
        file_lock = (
            FileLock(self._output_dir + "_infos.lock")
            if not is_remote_filesystem(self._fs)
            else contextlib.nullcontext()
        )
        with file_lock:
            DatasetInfosDict(**{self.config.name: self.info}).write_to_directory(self.get_imported_module_dir())

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        """Get kwargs for `self._split_generators()` from `prepare_split_kwargs`."""
        del prepare_split_kwargs
        return {}

    def as_dataset(
        self,
        split: Optional[Split] = None,
        run_post_process=True,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        ignore_verifications="deprecated",
        in_memory=False,
    ) -> Union[Dataset, DatasetDict]:
        """Return a Dataset for the specified split.

        Args:
            split (`datasets.Split`):
                Which subset of the data to return.
            run_post_process (`bool`, defaults to `True`):
                Whether to run post-processing dataset transforms and/or add
                indexes.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            ignore_verifications (`bool`, defaults to `False`):
                Whether to ignore the verifications of the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Deprecated version="2.9.1">

                `ignore_verifications` was deprecated in version 2.9.1 and will be removed in 3.0.0.
                Please use `verification_mode` instead.

                </Deprecated>
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            datasets.Dataset

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder('rotten_tomatoes')
        >>> builder.download_and_prepare()
        >>> ds = builder.as_dataset(split='train')
        >>> ds
        Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        ```
        """
        if ignore_verifications != "deprecated":
            verification_mode = verification_mode.NO_CHECKS if ignore_verifications else VerificationMode.ALL_CHECKS
            warnings.warn(
                "'ignore_verifications' was deprecated in favor of 'verification' in version 2.9.1 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'verification_mode={verification_mode.value}' instead.",
                FutureWarning,
            )
        if self._file_format is not None and self._file_format != "arrow":
            raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.")
        if not os.path.exists(self._output_dir):
            raise FileNotFoundError(
                f"Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call "
                "builder.download_and_prepare(), or use "
                "datasets.load_dataset() before trying to access the Dataset object."
            )

        logger.debug(f'Constructing Dataset for split {split or ", ".join(self.info.splits)}, from {self._output_dir}')

        # By default, return all splits
        if split is None:
            split = {s: s for s in self.info.splits}

        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)

        # Create a dataset for each of the given splits
        datasets = map_nested(
            partial(
                self._build_single_dataset,
                run_post_process=run_post_process,
                verification_mode=verification_mode,
                in_memory=in_memory,
            ),
            split,
            map_tuple=True,
            disable_tqdm=True,
        )
        if isinstance(datasets, dict):
            datasets = DatasetDict(datasets)
        return datasets

    def _build_single_dataset(
        self,
        split: Union[str, ReadInstruction, Split],
        run_post_process: bool,
        verification_mode: VerificationMode,
        in_memory: bool = False,
    ):
        """as_dataset for a single split."""
        if not isinstance(split, ReadInstruction):
            split = str(split)
            if split == "all":
                split = "+".join(self.info.splits.keys())
            split = Split(split)

        # Build base dataset
        ds = self._as_dataset(
            split=split,
            in_memory=in_memory,
        )
        if run_post_process:
            for resource_file_name in self._post_processing_resources(split).values():
                if os.sep in resource_file_name:
                    raise ValueError(f"Resources shouldn't be in a sub-directory: {resource_file_name}")
            resources_paths = {
                resource_name: os.path.join(self._output_dir, resource_file_name)
                for resource_name, resource_file_name in self._post_processing_resources(split).items()
            }
            post_processed = self._post_process(ds, resources_paths)
            if post_processed is not None:
                ds = post_processed
                recorded_checksums = {}
                record_checksums = False
                for resource_name, resource_path in resources_paths.items():
                    size_checksum = get_size_checksum_dict(resource_path)
                    recorded_checksums[resource_name] = size_checksum
                if verification_mode == VerificationMode.ALL_CHECKS and record_checksums:
                    if self.info.post_processed is None or self.info.post_processed.resources_checksums is None:
                        expected_checksums = None
                    else:
                        expected_checksums = self.info.post_processed.resources_checksums.get(split)
                    verify_checksums(expected_checksums, recorded_checksums, "post processing resources")
                if self.info.post_processed is None:
                    self.info.post_processed = PostProcessedInfo()
                if self.info.post_processed.resources_checksums is None:
                    self.info.post_processed.resources_checksums = {}
                self.info.post_processed.resources_checksums[str(split)] = recorded_checksums
                self.info.post_processing_size = sum(
                    checksums_dict["num_bytes"]
                    for split_checksums_dicts in self.info.post_processed.resources_checksums.values()
                    for checksums_dict in split_checksums_dicts.values()
                )
                if self.info.dataset_size is not None and self.info.download_size is not None:
                    self.info.size_in_bytes = (
                        self.info.dataset_size + self.info.download_size + self.info.post_processing_size
                    )
                self._save_info()
                ds._info.post_processed = self.info.post_processed
                ds._info.post_processing_size = self.info.post_processing_size
                ds._info.size_in_bytes = self.info.size_in_bytes
                if self.info.post_processed.features is not None:
                    if self.info.post_processed.features.type != ds.features.type:
                        raise ValueError(
                            f"Post-processed features info don't match the dataset:\nGot\n{self.info.post_processed.features}\nbut expected something like\n{ds.features}"
                        )
                    else:
                        ds.info.features = self.info.post_processed.features

        return ds

    def _as_dataset(self, split: Union[ReadInstruction, Split] = Split.TRAIN, in_memory: bool = False) -> Dataset:
        """Constructs a `Dataset`.

        This is the internal implementation to overwrite called when user calls
        `as_dataset`. It should read the pre-processed datasets files and generate
        the `Dataset` object.

        Args:
            split (`datasets.Split`):
                which subset of the data to read.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            `Dataset`
        """
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_name = self.dataset_name
        if self._check_legacy_cache():
            dataset_name = self.name
        dataset_kwargs = ArrowReader(cache_dir, self.info).read(
            name=dataset_name,
            instructions=split,
            split_infos=self.info.splits.values(),
            in_memory=in_memory,
        )
        fingerprint = self._get_dataset_fingerprint(split)
        return Dataset(fingerprint=fingerprint, **dataset_kwargs)

    def _get_dataset_fingerprint(self, split: Union[ReadInstruction, Split]) -> str:
        """The dataset fingerprint is the hash of the relative directory dataset_name/config_name/version/hash, as well as the split specs."""
        hasher = Hasher()
        hasher.update(Path(self._relative_data_dir()).as_posix())
        hasher.update(str(split))  # for example: train, train+test, train[:10%], test[:33%](pct1_dropremainder)
        fingerprint = hasher.hexdigest()
        return fingerprint

    def as_streaming_dataset(
        self,
        split: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> Union[Dict[str, IterableDataset], IterableDataset]:
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(
                f"Loading a streaming dataset cached in a {type(self._fs).__name__} is not supported yet."
            )

        dl_manager = StreamingDownloadManager(
            base_path=base_path or self.base_path,
            download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),
            dataset_name=self.dataset_name,
            data_dir=self.config.data_dir,
        )
        self._check_manual_download(dl_manager)
        splits_generators = {sg.name: sg for sg in self._split_generators(dl_manager)}
        # By default, return all splits
        if split is None:
            splits_generator = splits_generators
        elif split in splits_generators:
            splits_generator = splits_generators[split]
        else:
            raise ValueError(f"Bad split: {split}. Available splits: {list(splits_generators)}")

        # Create a dataset for each of the given splits
        datasets = map_nested(
            self._as_streaming_dataset_single,
            splits_generator,
            map_tuple=True,
        )
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _as_streaming_dataset_single(
        self,
        splits_generator,
    ) -> IterableDataset:
        ex_iterable = self._get_examples_iterable_for_split(splits_generator)
        # add auth to be able to access and decode audio/image files from private repositories.
        token_per_repo_id = {self.repo_id: self.token} if self.repo_id else {}
        return IterableDataset(
            ex_iterable, info=self.info, split=splits_generator.name, token_per_repo_id=token_per_repo_id
        )

    def _post_process(self, dataset: Dataset, resources_paths: Mapping[str, str]) -> Optional[Dataset]:
        """Run dataset transforms or add indexes"""
        return None

    def _post_processing_resources(self, split: str) -> Dict[str, str]:
        """Mapping resource_name -> resource_file_name"""
        return {}

    def _download_post_processing_resources(
        self, split: str, resource_name: str, dl_manager: DownloadManager
    ) -> Optional[str]:
        """Download the resource using the download manager and return the downloaded path."""
        return None

    @abc.abstractmethod
    def _split_generators(self, dl_manager: Union[DownloadManager, StreamingDownloadManager]):
        """Specify feature dictionary generators and dataset splits.

        This function returns a list of `SplitGenerator`s defining how to generate
        data and what splits to use.

        Example:

            return [
                    datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={'file': 'train_data.zip'},
                    ),
                    datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={'file': 'test_data.zip'},
                    ),
            ]

        The above code will first call `_generate_examples(file='train_data.zip')`
        to write the train data, then `_generate_examples(file='test_data.zip')` to
        write the test data.

        Datasets are typically split into different subsets to be used at various
        stages of training and evaluation.

        Note that for datasets without a `VALIDATION` split, you can use a
        fraction of the `TRAIN` data for evaluation as you iterate on your model
        so as not to overfit to the `TEST` data.

        For downloads and extractions, use the given `download_manager`.
        Note that the `DownloadManager` caches downloads, so it is fine to have each
        generator attempt to download the source data.

        A good practice is to download all data in this function, and then
        distribute the relevant parts to each split with the `gen_kwargs` argument

        Args:
            dl_manager (`Union[DownloadManager, StreamingDownloadManager]`):
                Download manager to download the data

        Returns:
            `list<SplitGenerator>`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        max_shard_size: Optional[Union[str, int]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Generate the examples and record them on disk.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
            file_format (`str`, *optional*):
                format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            **kwargs: Additional kwargs forwarded from _download_and_prepare (ex:
                beam pipeline)
        """
        raise NotImplementedError()

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        """Generate the examples on the fly.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
        """
        raise NotImplementedError()


class GeneratorBasedBuilder(DatasetBuilder):
    """Base class for datasets with data generation based on dict generators.

    `GeneratorBasedBuilder` is a convenience class that abstracts away much
    of the data writing and reading of `DatasetBuilder`. It expects subclasses to
    implement generators of feature dictionaries across the dataset splits
    (`_split_generators`). See the method docstrings for details.
    """

    @abc.abstractmethod
    def _generate_examples(self, **kwargs):
        """Default function generating examples for each `SplitGenerator`.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.
        This function is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples yielded here will be written on
        disk.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            key: `str` or `int`, a unique deterministic example identification key.
                * Unique: An error will be raised if two examples are yield with the
                    same key.
                * Deterministic: When generating the dataset twice, the same example
                    should have the same key.
                Good keys can be the image id, or line number if examples are extracted
                from a text file.
                The key will be hashed and sorted to shuffle examples deterministically,
                such as generating the dataset multiple times keep examples in the
                same order.
            example: `dict<str feature_name, feature_value>`, a feature dictionary
                ready to be encoded and written to disk. The example will be
                encoded with `self.info.features.encode_example({...})`.
        """
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        check_duplicate_keys: bool,
        file_format="arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)

        if self.info.splits is not None:
            split_info = self.info.splits[split_generator.name]
        else:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_input_shards <= 1:
                logger.warning(
                    f"Setting num_proc from {num_proc} back to 1 for the {split_info.name} split to disable multiprocessing as it only contains one shard."
                )
                num_proc = 1
            elif num_input_shards < num_proc:
                logger.warning(
                    f"Setting num_proc from {num_proc} to {num_input_shards} for the {split_info.name} split as it only contains {num_input_shards} shards."
                )
                num_proc = num_input_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
            "split_info": split_info,
            "check_duplicate_keys": check_duplicate_keys,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(
                    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
                ):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            # wrapping everything into lists for consistency with the multiprocessed code path
            assert result is not None, "Failed to retrieve results from prepare_split"
            examples_per_job, bytes_per_job, features_per_job, shards_per_job, shard_lengths_per_job = [
                [item] for item in result
            ]
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            # the content is the result of the job
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            # the content is the number of examples progress update
                            pbar.update(content)

            assert (
                None not in examples_per_job
            ), f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results"

        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            # use the -SSSSS-of-NNNNN pattern

            def _rename_shard(shard_and_job: Tuple[int]):
                shard_id, job_id = shard_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            shards_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shards_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [
                shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths
            ]
        else:
            # don't use any pattern
            shard_id, job_id = 0, 0
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: SplitInfo,
        check_duplicate_keys: bool,
        job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        generator = self._generate_examples(**gen_kwargs)
        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                hash_salt=split_info.name,
                check_duplicates=check_duplicate_keys,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for key, record in generator:
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            hash_salt=split_info.name,
                            check_duplicates=check_duplicate_keys,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    example = self.info.features.encode_example(record) if self.info.features is not None else record
                    writer.write(example, key)
                    num_examples_progress_update += 1
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            # Ignore the writer's error for no examples written to the file if this error was caused by the error in _generate_examples before the first example was yielded
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield job_id, True, (total_num_examples, total_num_bytes, writer._features, num_shards, shard_lengths)

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
        super()._download_and_prepare(
            dl_manager,
            verification_mode,
            check_duplicate_keys=verification_mode == VerificationMode.BASIC_CHECKS
            or verification_mode == VerificationMode.ALL_CHECKS,
            **prepare_splits_kwargs,
        )

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ExamplesIterable(self._generate_examples, split_generator.gen_kwargs)


class ArrowBasedBuilder(DatasetBuilder):
    """Base class for datasets with data generation based on Arrow loading functions (CSV/JSON/Parquet)."""

    @abc.abstractmethod
    def _generate_tables(self, **kwargs):
        """Default function generating examples for each `SplitGenerator`.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.
        This function is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples yielded here will be written on
        disk.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            key: `str` or `int`, a unique deterministic example identification key.
                * Unique: An error will be raised if two examples are yield with the
                    same key.
                * Deterministic: When generating the dataset twice, the same example
                    should have the same key.
                Good keys can be the image id, or line number if examples are extracted
                from a text file.
                The key will be hashed and sorted to shuffle examples deterministically,
                such as generating the dataset multiple times keep examples in the
                same order.
            example: `pyarrow.Table`, a feature table
                ready to be encoded and written to disk.
        """
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[str, int]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)

        try:
            split_info = self.info.splits[split_generator.name]
        except Exception:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_input_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_input_shards <= 1:
                logger.warning(
                    f"Setting num_proc from {num_proc} back to 1 for the {split_info.name} split to disable multiprocessing as it only contains one shard."
                )
                num_proc = 1
            elif num_input_shards < num_proc:
                logger.warning(
                    f"Setting num_proc from {num_proc} to {num_input_shards} for the {split_info.name} split as it only contains {num_input_shards} shards."
                )
                num_proc = num_input_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(
                    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
                ):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            # wrapping everything into lists for consistency with the multiprocessed code path
            assert result is not None, "Failed to retrieve results from prepare_split"
            examples_per_job, bytes_per_job, features_per_job, shards_per_job, shard_lengths_per_job = [
                [item] for item in result
            ]
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            # the content is the result of the job
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            # the content is the number of examples progress update
                            pbar.update(content)

            assert (
                None not in examples_per_job
            ), f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results"

        total_shards = sum(shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            # use the -SSSSS-of-NNNNN pattern

            def _rename_shard(shard_id_and_job: Tuple[int]):
                shard_id, job_id = shard_id_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            shard_ids_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shard_ids_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [
                shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths
            ]
        else:
            # don't use any pattern
            shard_id, job_id = 0, 0
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, job_id: int
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        gen_kwargs = {k: tracked_list(v) if isinstance(v, list) else v for k, v in gen_kwargs.items()}
        generator = self._generate_tables(**gen_kwargs)
        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for _, table in generator:
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    try:
                        writer.write_table(table)
                    except CastError as cast_error:
                        raise DatasetGenerationCastError.from_cast_error(
                            cast_error=cast_error,
                            builder_name=self.info.builder_name,
                            gen_kwargs=gen_kwargs,
                            token=self.token,
                        )
                    num_examples_progress_update += len(table)
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            # Ignore the writer's error for no examples written to the file if this error was caused by the error in _generate_examples before the first example was yielded
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            if isinstance(e, DatasetGenerationError):
                raise
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield job_id, True, (total_num_examples, total_num_bytes, writer._features, num_shards, shard_lengths)

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ArrowExamplesIterable(self._generate_tables, kwargs=split_generator.gen_kwargs)


class MissingBeamOptions(ValueError):
    pass


@deprecated("Use `GeneratorBasedBuilder` or `ArrowBasedBuilder` instead.")
class BeamBasedBuilder(DatasetBuilder):
    """Beam-based Builder."""

    def __init__(self, *args, beam_runner=None, beam_options=None, **kwargs):
        self._beam_runner = beam_runner
        self._beam_options = beam_options
        self._beam_writers = {}  # {split: beam_writer} mapping.
        super().__init__(*args, **kwargs)

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        # Pass `pipeline` into `_split_generators()` from `prepare_split_kwargs` if
        # it's in the call signature of `_split_generators()`.
        # This allows for global preprocessing in beam.
        split_generators_kwargs = {}
        split_generators_arg_names = inspect.signature(self._split_generators).parameters.keys()
        if "pipeline" in split_generators_arg_names:
            split_generators_kwargs["pipeline"] = prepare_split_kwargs["pipeline"]
        return split_generators_kwargs

    @abc.abstractmethod
    def _build_pcollection(self, pipeline, **kwargs):
        """Build the beam pipeline examples for each `SplitGenerator`.

        This function extracts examples from the raw data with parallel transforms
        in a Beam pipeline. It is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples from the PCollection will be
        encoded and written to disk.

        <Tip warning={true}>
        Warning: When running in a distributed setup, make sure that the data
        which will be read (download_dir, manual_dir,...) and written (cache_dir)
        can be accessed by the workers jobs. The data should be located in a
        shared filesystem, like GCS.
        </Tip>

        Args:
            pipeline ([`utils.beam_utils.BeamPipeline`]):
                Apache Beam pipeline.
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs.

        Returns:
            `beam.PCollection`: Apache Beam PCollection containing the
                example to send to `self.info.features.encode_example(...)`.

        Example:

        ```
        def _build_pcollection(pipeline, extracted_dir=None):
            return (
                    pipeline
                    | beam.Create(gfile.io.listdir(extracted_dir))
                    | beam.Map(_process_file)
            )
        ```
        """
        raise NotImplementedError()

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
        # Create the Beam pipeline and forward it to `_prepare_split`
        import apache_beam as beam

        import datasets.utils.beam_utils as beam_utils

        beam_runner = self._beam_runner
        beam_options = self._beam_options

        if not beam_runner and not beam_options:
            usage_example = f"load_dataset('{self.repo_id or self.dataset_name}', '{self.config.name}', beam_runner='DirectRunner')"
            raise MissingBeamOptions(
                "Trying to generate a dataset using Apache Beam, yet no Beam Runner "
                "or PipelineOptions() has been provided in `load_dataset` or in the "
                "builder arguments. For big datasets it has to run on large-scale data "
                "processing tools like Dataflow, Spark, etc. More information about "
                "Apache Beam runners at "
                "https://beam.apache.org/documentation/runners/capability-matrix/"
                "\nIf you really want to run it locally because you feel like the "
                "Dataset is small enough, you can use the local beam runner called "
                "`DirectRunner` (you may run out of memory). \nExample of usage: "
                f"\n\t`{usage_example}`"
            )
        if self._writer_batch_size is not None:
            logger.warning(
                "`writer_batch_size` is not supported for beam pipelines yet. Using the default chunk size for writing."
            )

        # Beam type checking assumes transforms multiple outputs are of same type,
        # which is not our case. Plus it doesn't handle correctly all types, so we
        # are better without it.
        pipeline_options = {"pipeline_type_check": False}
        if "num_proc" in prepare_splits_kwargs:
            num_workers = prepare_splits_kwargs.pop("num_proc")
            pipeline_options["direct_num_workers"] = num_workers
            pipeline_options["num_workers"] = num_workers
            pipeline_options["direct_running_mode"] = "multi_processing"
            # TODO: Fix ModuleNotFoundError: No module named 'datasets_modules' when running multiprocessed DirectRunner
            raise NotImplementedError("Using a DirectRunner with `num_proc` for multiprocessing it not supported yet.")
        beam_options = beam_options or beam.options.pipeline_options.PipelineOptions.from_dictionary(pipeline_options)
        # Use a single pipeline for all splits
        pipeline = beam_utils.BeamPipeline(
            runner=beam_runner,
            options=beam_options,
        )
        super()._download_and_prepare(
            dl_manager, verification_mode=VerificationMode.NO_CHECKS, pipeline=pipeline, **prepare_splits_kwargs
        )  # TODO handle verification_mode in beam datasets
        # Run pipeline
        pipeline_results = pipeline.run()
        pipeline_results.wait_until_finish()
        metrics = pipeline_results.metrics()
        # Update `info.splits`.
        split_dict = self.info.splits
        for split_name, beam_writer in self._beam_writers.items():
            m_filter = beam.metrics.MetricsFilter().with_namespace(namespace=split_name)
            num_examples, num_bytes = beam_writer.finalize(metrics.query(m_filter))
            split_info = split_dict[split_name]
            split_info.num_examples = num_examples
            split_info.num_bytes = num_bytes
            if hasattr(beam_writer, "_shard_lengths") and len(beam_writer._shard_lengths) > 1:
                # keep the -SSSSS-of-NNNNN pattern
                split_info.shard_lengths = beam_writer._shard_lengths
            else:
                # don't use any pattern
                file_format = prepare_splits_kwargs.get("file_format", "arrow")
                src_fname = f"{self.dataset_name}-{split_name}-00000-of-00001.{file_format}"
                dst_fname = f"{self.dataset_name}-{split_name}.{file_format}"
                src_fpath = posixpath.join(self._output_dir, src_fname)
                dst_fpath = posixpath.join(self._output_dir, dst_fname)
                self._rename(src_fpath, dst_fpath)

    def _save_info(self):
        download_config = (
            self.dl_manager.download_config
            if self.dl_manager
            else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
        )
        with xopen(f"{self._output_dir}/{config.DATASET_INFO_FILENAME}", "wb", download_config=download_config) as f:
            self.info._dump_info(f)
        if self.info.license:
            with xopen(f"{self._output_dir}/{config.LICENSE_FILENAME}", "wb", download_config=download_config) as f:
                self.info._dump_license(f)

    def _prepare_split(
        self, split_generator, pipeline, file_format="arrow", max_shard_size: Optional[Union[str, int]] = None
    ):
        import apache_beam as beam

        if max_shard_size is not None:
            raise NotImplementedError(
                "max_shard_size is not supported for Beam datasets."
                "Please set it to None to use the default Apache Beam sharding and get the best performance."
            )

        # To write examples in filesystem:
        split_name = split_generator.split_info.name
        fname = f"{self.dataset_name}-{split_name}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)
        beam_writer = BeamWriter(
            features=self.info.features, path=fpath, namespace=split_name, cache_dir=self._output_dir
        )
        self._beam_writers[split_name] = beam_writer

        encode_example = self.info.features.encode_example

        # Note: We need to wrap the pipeline in a PTransform to avoid re-using the
        # same label names for each split
        @beam.ptransform_fn
        def _build_pcollection(pipeline):
            """PTransformation which build a single split."""
            # Encode the PCollection
            pcoll_examples = self._build_pcollection(pipeline, **split_generator.gen_kwargs)
            pcoll_examples |= "Encode" >> beam.Map(lambda key_ex: (key_ex[0], encode_example(key_ex[1])))
            return beam_writer.write_from_pcollection(pcoll_examples)

        # Add the PCollection to the pipeline
        _ = pipeline | split_name >> _build_pcollection()  # pylint: disable=no-value-for-parameter max_bytes_per_shard

    def as_streaming_dataset(
        self,
        split: Optional[str] = None,
    ) -> Union[Dict[str, IterableDataset], IterableDataset]:
        self._request_info_from_hf_gcs()
        datasets = {
            split.name: IterableDataset(self._get_examples_iterable_for_split(split), info=self.info, split=split.name)
            for split in self.info.splits.values()
        }
        if split:
            try:
                datasets = datasets[split]
            except KeyError:
                raise ValueError(f"Bad split: {split}. Available splits: {list(datasets)}")
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _get_examples_iterable_for_split(self, split: SplitInfo) -> ExamplesIterable:
        return ExamplesIterable(self._generate_examples_from_hf_gcs, {"split": split})

    def _generate_examples_from_hf_gcs(self, split: SplitInfo):
        if split.shard_lengths:
            num_shards = len(split.shard_lengths)
            remote_prepared_urls = [
                f"{self._remote_cache_dir_from_hf_gcs}/{self.name}-{split.name}-{shard_id:05d}-of-{num_shards:05d}.arrow"
                for shard_id in range(num_shards)
            ]
        else:
            remote_prepared_urls = [f"{self._remote_cache_dir_from_hf_gcs}/{self.name}-{split.name}.arrow"]
        key = 0
        download_config = (
            self.dl_manager.download_config
            if self.dl_manager
            else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
        )
        for remote_prepared_url in remote_prepared_urls:
            with xopen(remote_prepared_url, "rb", download_config=download_config) as f:
                with pa.ipc.open_stream(f) as reader:
                    for record_batch in reader:
                        for record in record_batch.to_pylist():
                            yield key, record
                            key += 1

    def _request_info_from_hf_gcs(self):
        from .download.streaming_download_manager import xopen

        remote_dataset_info = f"{self._remote_cache_dir_from_hf_gcs}/{config.DATASET_INFO_FILENAME}"
        try:
            download_config = download_config = (
                self.dl_manager.download_config
                if self.dl_manager
                else DownloadConfig(token=self.token, storage_options=self._fs.storage_options)
            )
            with xopen(remote_dataset_info, download_config=download_config) as f:
                import json

                _info = json.load(f)
        except FileNotFoundError as err:
            raise DatasetNotOnHfGcsError(err) from None
        self.info.update(DatasetInfo.from_dict(_info))

    @property
    def _remote_cache_dir_from_hf_gcs(self):
        relative_data_dir = self._relative_data_dir(with_hash=False)
        return HF_GCP_BASE_URL + "/" + Path(relative_data_dir).as_posix()
