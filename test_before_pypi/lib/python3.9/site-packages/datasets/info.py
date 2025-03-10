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
"""DatasetInfo and MetricInfo record information we know about a dataset and a metric.

This includes things that we know about the dataset statically, i.e.:
 - description
 - canonical location
 - does it have validation and tests splits
 - size
 - etc.

This also includes the things that can and should be computed once we've
processed the dataset as well:
 - number of examples (in each split)
 - etc.
"""

import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

import fsspec
from fsspec.core import url_to_fs
from huggingface_hub import DatasetCard, DatasetCardData

from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values


logger = get_logger(__name__)


@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""


@dataclass
class DownloadChecksumsEntryData:
    key: str = ""
    value: str = ""


class MissingCachedSizesConfigError(Exception):
    """The expected cached sizes of the download file are missing."""


class NonMatchingCachedSizesError(Exception):
    """The prepared split doesn't have expected sizes."""


@dataclass
class PostProcessedInfo:
    features: Optional[Features] = None
    resources_checksums: Optional[dict] = None

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)

    @classmethod
    def from_dict(cls, post_processed_info_dict: dict) -> "PostProcessedInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in post_processed_info_dict.items() if k in field_names})


@dataclass
class DatasetInfo:
    """Information about a dataset.

    `DatasetInfo` documents datasets, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Not all fields are known on construction and may be updated later.

    Attributes:
        description (`str`):
            A description of the dataset.
        citation (`str`):
            A BibTeX citation of the dataset.
        homepage (`str`):
            A URL to the official homepage for the dataset.
        license (`str`):
            The dataset's license. It can be the name of the license or a paragraph containing the terms of the license.
        features ([`Features`], *optional*):
            The features used to specify the dataset's column types.
        post_processed (`PostProcessedInfo`, *optional*):
            Information regarding the resources of a possible post-processing of a dataset. For example, it can contain the information of an index.
        supervised_keys (`SupervisedKeysData`, *optional*):
            Specifies the input feature and the label for supervised learning if applicable for the dataset (legacy from TFDS).
        builder_name (`str`, *optional*):
            The name of the `GeneratorBasedBuilder` subclass used to create the dataset. Usually matched to the corresponding script name. It is also the snake_case version of the dataset builder class name.
        config_name (`str`, *optional*):
            The name of the configuration derived from [`BuilderConfig`].
        version (`str` or [`Version`], *optional*):
            The version of the dataset.
        splits (`dict`, *optional*):
            The mapping between split name and metadata.
        download_checksums (`dict`, *optional*):
            The mapping between the URL to download the dataset's checksums and corresponding metadata.
        download_size (`int`, *optional*):
            The size of the files to download to generate the dataset, in bytes.
        post_processing_size (`int`, *optional*):
            Size of the dataset in bytes after post-processing, if any.
        dataset_size (`int`, *optional*):
            The combined size in bytes of the Arrow tables for all splits.
        size_in_bytes (`int`, *optional*):
            The combined size in bytes of all files associated with the dataset (downloaded files + Arrow files).
        task_templates (`List[TaskTemplate]`, *optional*):
            The task templates to prepare the dataset for during training and evaluation. Each template casts the dataset's [`Features`] to standardized column names and types as detailed in `datasets.tasks`.
        **config_kwargs (additional keyword arguments):
            Keyword arguments to be passed to the [`BuilderConfig`] and used in the [`DatasetBuilder`].
    """

    # Set in the dataset scripts
    description: str = dataclasses.field(default_factory=str)
    citation: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    features: Optional[Features] = None
    post_processed: Optional[PostProcessedInfo] = None
    supervised_keys: Optional[SupervisedKeysData] = None
    task_templates: Optional[List[TaskTemplate]] = None

    # Set later by the builder
    builder_name: Optional[str] = None
    dataset_name: Optional[str] = None  # for packaged builders, to be different from builder_name
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    # Set later by `download_and_prepare`
    splits: Optional[dict] = None
    download_checksums: Optional[dict] = None
    download_size: Optional[int] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None

    _INCLUDED_INFO_IN_YAML: ClassVar[List[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "features",
        "splits",
    ]

    def __post_init__(self):
        # Convert back to the correct classes when we reload from dict
        if self.features is not None and not isinstance(self.features, Features):
            self.features = Features.from_dict(self.features)
        if self.post_processed is not None and not isinstance(self.post_processed, PostProcessedInfo):
            self.post_processed = PostProcessedInfo.from_dict(self.post_processed)
        if self.version is not None and not isinstance(self.version, Version):
            if isinstance(self.version, str):
                self.version = Version(self.version)
            else:
                self.version = Version.from_dict(self.version)
        if self.splits is not None and not isinstance(self.splits, SplitDict):
            self.splits = SplitDict.from_split_dict(self.splits)
        if self.supervised_keys is not None and not isinstance(self.supervised_keys, SupervisedKeysData):
            if isinstance(self.supervised_keys, (tuple, list)):
                self.supervised_keys = SupervisedKeysData(*self.supervised_keys)
            else:
                self.supervised_keys = SupervisedKeysData(**self.supervised_keys)

        # Parse and make a list of templates
        if self.task_templates is not None:
            if isinstance(self.task_templates, (list, tuple)):
                templates = [
                    template if isinstance(template, TaskTemplate) else task_template_from_dict(template)
                    for template in self.task_templates
                ]
                self.task_templates = [template for template in templates if template is not None]
            elif isinstance(self.task_templates, TaskTemplate):
                self.task_templates = [self.task_templates]
            else:
                template = task_template_from_dict(self.task_templates)
                self.task_templates = [template] if template is not None else []

        # Align task templates with features
        if self.task_templates is not None:
            self.task_templates = list(self.task_templates)
            if self.features is not None:
                self.task_templates = [
                    template.align_with_features(self.features) for template in (self.task_templates)
                ]

    def write_to_directory(
        self, dataset_info_dir, pretty_print=False, fs="deprecated", storage_options: Optional[dict] = None
    ):
        """Write `DatasetInfo` and license (if present) as JSON files to `dataset_info_dir`.

        Args:
            dataset_info_dir (`str`):
                Destination directory.
            pretty_print (`bool`, defaults to `False`):
                If `True`, the JSON will be pretty-printed with the indent level of 4.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem used to download the files from.

                <Deprecated version="2.9.0">

                `fs` was deprecated in version 2.9.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`.

                </Deprecated>

            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation")
        >>> ds.info.write_to_directory("/path/to/directory/")
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
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), "wb") as f:
            self._dump_info(f, pretty_print=pretty_print)
        if self.license:
            with fs.open(posixpath.join(dataset_info_dir, config.LICENSE_FILENAME), "wb") as f:
                self._dump_license(f)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(json.dumps(asdict(self), indent=4 if pretty_print else None).encode("utf-8"))

    def _dump_license(self, file):
        """Dump license in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(self.license.encode("utf-8"))

    @classmethod
    def from_merge(cls, dataset_infos: List["DatasetInfo"]):
        dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]

        if len(dataset_infos) > 0 and all(dataset_infos[0] == dset_info for dset_info in dataset_infos):
            # if all dataset_infos are equal we don't need to merge. Just return the first.
            return dataset_infos[0]

        description = "\n\n".join(unique_values(info.description for info in dataset_infos)).strip()
        citation = "\n\n".join(unique_values(info.citation for info in dataset_infos)).strip()
        homepage = "\n\n".join(unique_values(info.homepage for info in dataset_infos)).strip()
        license = "\n\n".join(unique_values(info.license for info in dataset_infos)).strip()
        features = None
        supervised_keys = None
        task_templates = None

        # Find common task templates across all dataset infos
        all_task_templates = [info.task_templates for info in dataset_infos if info.task_templates is not None]
        if len(all_task_templates) > 1:
            task_templates = list(set(all_task_templates[0]).intersection(*all_task_templates[1:]))
        elif len(all_task_templates):
            task_templates = list(set(all_task_templates[0]))
        # If no common task templates found, replace empty list with None
        task_templates = task_templates if task_templates else None

        return cls(
            description=description,
            citation=citation,
            homepage=homepage,
            license=license,
            features=features,
            supervised_keys=supervised_keys,
            task_templates=task_templates,
        )

    @classmethod
    def from_directory(
        cls, dataset_info_dir: str, fs="deprecated", storage_options: Optional[dict] = None
    ) -> "DatasetInfo":
        """Create [`DatasetInfo`] from the JSON file in `dataset_info_dir`.

        This function updates all the dynamically generated fields (num_examples,
        hash, time of creation,...) of the [`DatasetInfo`].

        This will overwrite all previous metadata.

        Args:
            dataset_info_dir (`str`):
                The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
            fs (`fsspec.spec.AbstractFileSystem`, *optional*):
                Instance of the remote filesystem used to download the files from.

                <Deprecated version="2.9.0">

                `fs` was deprecated in version 2.9.0 and will be removed in 3.0.0.
                Please use `storage_options` instead, e.g. `storage_options=fs.storage_options`.

                </Deprecated>

            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the file-system backend, if any.

                <Added version="2.9.0"/>

        Example:

        ```py
        >>> from datasets import DatasetInfo
        >>> ds_info = DatasetInfo.from_directory("/path/to/directory/")
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
        fs, *_ = url_to_fs(dataset_info_dir, **(storage_options or {}))
        logger.info(f"Loading Dataset info from {dataset_info_dir}")
        if not dataset_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")
        with fs.open(posixpath.join(dataset_info_dir, config.DATASET_INFO_FILENAME), "r", encoding="utf-8") as f:
            dataset_info_dict = json.load(f)
        return cls.from_dict(dataset_info_dict)

    @classmethod
    def from_dict(cls, dataset_info_dict: dict) -> "DatasetInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dataset_info_dict.items() if k in field_names})

    def update(self, other_dataset_info: "DatasetInfo", ignore_none=True):
        self_dict = self.__dict__
        self_dict.update(
            **{
                k: copy.deepcopy(v)
                for k, v in other_dataset_info.__dict__.items()
                if (v is not None or not ignore_none)
            }
        )

    def copy(self) -> "DatasetInfo":
        return self.__class__(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})

    def _to_yaml_dict(self) -> dict:
        yaml_dict = {}
        dataset_info_dict = asdict(self)
        for key in dataset_info_dict:
            if key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, "_to_yaml_list"):  # Features, SplitDict
                    yaml_dict[key] = value._to_yaml_list()
                elif hasattr(value, "_to_yaml_string"):  # Version
                    yaml_dict[key] = value._to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def _from_yaml_dict(cls, yaml_data: dict) -> "DatasetInfo":
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get("features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            yaml_data["splits"] = SplitDict._from_yaml_list(yaml_data["splits"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})


class DatasetInfosDict(Dict[str, DatasetInfo]):
    def write_to_directory(self, dataset_infos_dir, overwrite=False, pretty_print=False) -> None:
        total_dataset_infos = {}
        dataset_infos_path = os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)
        dataset_readme_path = os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)
        if not overwrite:
            total_dataset_infos = self.from_directory(dataset_infos_dir)
        total_dataset_infos.update(self)
        if os.path.exists(dataset_infos_path):
            # for backward compatibility, let's update the JSON file if it exists
            with open(dataset_infos_path, "w", encoding="utf-8") as f:
                dataset_infos_dict = {
                    config_name: asdict(dset_info) for config_name, dset_info in total_dataset_infos.items()
                }
                json.dump(dataset_infos_dict, f, indent=4 if pretty_print else None)
        # Dump the infos in the YAML part of the README.md file
        if os.path.exists(dataset_readme_path):
            dataset_card = DatasetCard.load(dataset_readme_path)
            dataset_card_data = dataset_card.data
        else:
            dataset_card = None
            dataset_card_data = DatasetCardData()
        if total_dataset_infos:
            total_dataset_infos.to_dataset_card_data(dataset_card_data)
            dataset_card = (
                DatasetCard("---\n" + str(dataset_card_data) + "\n---\n") if dataset_card is None else dataset_card
            )
            dataset_card.save(Path(dataset_readme_path))

    @classmethod
    def from_directory(cls, dataset_infos_dir) -> "DatasetInfosDict":
        logger.info(f"Loading Dataset Infos from {dataset_infos_dir}")
        # Load the info from the YAML part of README.md
        if os.path.exists(os.path.join(dataset_infos_dir, config.REPOCARD_FILENAME)):
            dataset_card_data = DatasetCard.load(Path(dataset_infos_dir) / config.REPOCARD_FILENAME).data
            if "dataset_info" in dataset_card_data:
                return cls.from_dataset_card_data(dataset_card_data)
        if os.path.exists(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME)):
            # this is just to have backward compatibility with dataset_infos.json files
            with open(os.path.join(dataset_infos_dir, config.DATASETDICT_INFOS_FILENAME), encoding="utf-8") as f:
                return cls(
                    {
                        config_name: DatasetInfo.from_dict(dataset_info_dict)
                        for config_name, dataset_info_dict in json.load(f).items()
                    }
                )
        else:
            return cls()

    @classmethod
    def from_dataset_card_data(cls, dataset_card_data: DatasetCardData) -> "DatasetInfosDict":
        if isinstance(dataset_card_data.get("dataset_info"), (list, dict)):
            if isinstance(dataset_card_data["dataset_info"], list):
                return cls(
                    {
                        dataset_info_yaml_dict.get("config_name", "default"): DatasetInfo._from_yaml_dict(
                            dataset_info_yaml_dict
                        )
                        for dataset_info_yaml_dict in dataset_card_data["dataset_info"]
                    }
                )
            else:
                dataset_info = DatasetInfo._from_yaml_dict(dataset_card_data["dataset_info"])
                dataset_info.config_name = dataset_card_data["dataset_info"].get("config_name", "default")
                return cls({dataset_info.config_name: dataset_info})
        else:
            return cls()

    def to_dataset_card_data(self, dataset_card_data: DatasetCardData) -> None:
        if self:
            # first get existing metadata info
            if "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], dict):
                dataset_metadata_infos = {
                    dataset_card_data["dataset_info"].get("config_name", "default"): dataset_card_data["dataset_info"]
                }
            elif "dataset_info" in dataset_card_data and isinstance(dataset_card_data["dataset_info"], list):
                dataset_metadata_infos = {
                    config_metadata["config_name"]: config_metadata
                    for config_metadata in dataset_card_data["dataset_info"]
                }
            else:
                dataset_metadata_infos = {}
            # update/rewrite existing metadata info with the one to dump
            total_dataset_infos = {
                **dataset_metadata_infos,
                **{config_name: dset_info._to_yaml_dict() for config_name, dset_info in self.items()},
            }
            # the config_name from the dataset_infos_dict takes over the config_name of the DatasetInfo
            for config_name, dset_info_yaml_dict in total_dataset_infos.items():
                dset_info_yaml_dict["config_name"] = config_name
            if len(total_dataset_infos) == 1:
                # use a struct instead of a list of configurations, since there's only one
                dataset_card_data["dataset_info"] = next(iter(total_dataset_infos.values()))
                config_name = dataset_card_data["dataset_info"].pop("config_name", None)
                if config_name != "default":
                    # if config_name is not "default" preserve it and put at the first position
                    dataset_card_data["dataset_info"] = {
                        "config_name": config_name,
                        **dataset_card_data["dataset_info"],
                    }
            else:
                dataset_card_data["dataset_info"] = []
                for config_name, dataset_info_yaml_dict in sorted(total_dataset_infos.items()):
                    # add the config_name field in first position
                    dataset_info_yaml_dict.pop("config_name", None)
                    dataset_info_yaml_dict = {"config_name": config_name, **dataset_info_yaml_dict}
                    dataset_card_data["dataset_info"].append(dataset_info_yaml_dict)


@dataclass
class MetricInfo:
    """Information about a metric.

    `MetricInfo` documents a metric, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.
    """

    # Set in the dataset scripts
    description: str
    citation: str
    features: Features
    inputs_description: str = dataclasses.field(default_factory=str)
    homepage: str = dataclasses.field(default_factory=str)
    license: str = dataclasses.field(default_factory=str)
    codebase_urls: List[str] = dataclasses.field(default_factory=list)
    reference_urls: List[str] = dataclasses.field(default_factory=list)
    streamable: bool = False
    format: Optional[str] = None

    # Set later by the builder
    metric_name: Optional[str] = None
    config_name: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if self.format is not None:
            for key, value in self.features.items():
                if not isinstance(value, Value):
                    raise ValueError(
                        f"When using 'numpy' format, all features should be a `datasets.Value` feature. "
                        f"Here {key} is an instance of {value.__class__.__name__}"
                    )

    def write_to_directory(self, metric_info_dir, pretty_print=False):
        """Write `MetricInfo` as JSON to `metric_info_dir`.
        Also save the license separately in LICENCE.
        If `pretty_print` is True, the JSON will be pretty-printed with the indent level of 4.

        Example:

        ```py
        >>> from datasets import load_metric
        >>> metric = load_metric("accuracy")
        >>> metric.info.write_to_directory("/path/to/directory/")
        ```
        """
        with open(os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4 if pretty_print else None)

        if self.license:
            with open(os.path.join(metric_info_dir, config.LICENSE_FILENAME), "w", encoding="utf-8") as f:
                f.write(self.license)

    @classmethod
    def from_directory(cls, metric_info_dir) -> "MetricInfo":
        """Create MetricInfo from the JSON file in `metric_info_dir`.

        Args:
            metric_info_dir: `str` The directory containing the metadata file. This
                should be the root directory of a specific dataset version.

        Example:

        ```py
        >>> from datasets import MetricInfo
        >>> metric_info = MetricInfo.from_directory("/path/to/directory/")
        ```
        """
        logger.info(f"Loading Metric info from {metric_info_dir}")
        if not metric_info_dir:
            raise ValueError("Calling MetricInfo.from_directory() with undefined metric_info_dir.")

        with open(os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME), encoding="utf-8") as f:
            metric_info_dict = json.load(f)
        return cls.from_dict(metric_info_dict)

    @classmethod
    def from_dict(cls, metric_info_dict: dict) -> "MetricInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in metric_info_dict.items() if k in field_names})
