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

__version__ = "2.21.0"

from .arrow_dataset import Dataset
from .arrow_reader import ReadInstruction
from .builder import ArrowBasedBuilder, BeamBasedBuilder, BuilderConfig, DatasetBuilder, GeneratorBasedBuilder
from .combine import concatenate_datasets, interleave_datasets
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download import *
from .features import *
from .fingerprint import disable_caching, enable_caching, is_caching_enabled, set_caching_enabled
from .info import DatasetInfo, MetricInfo
from .inspect import (
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_default_config_name,
    get_dataset_infos,
    get_dataset_split_names,
    inspect_dataset,
    inspect_metric,
    list_datasets,
    list_metrics,
)
from .iterable_dataset import IterableDataset
from .load import load_dataset, load_dataset_builder, load_from_disk, load_metric
from .metric import Metric
from .splits import (
    NamedSplit,
    NamedSplitAll,
    Split,
    SplitBase,
    SplitDict,
    SplitGenerator,
    SplitInfo,
    SubSplitInfo,
    percent,
)
from .tasks import *
from .utils import *
from .utils import logging


# isort: split

# Deprecated modules
from . import arrow_dataset as _arrow_dataset
from . import utils as _utils
from .exceptions import ExpectedMoreDownloadedFiles, ExpectedMoreSplits, UnexpectedDownloadedFile, UnexpectedSplits
from .utils import download_manager as _deprecated_download_manager
from .utils import info_utils as _deprecated_info_utils


_arrow_dataset.concatenate_datasets = concatenate_datasets
_utils.DownloadConfig = DownloadConfig
_utils.DownloadManager = DownloadManager
_utils.DownloadMode = DownloadMode
_deprecated_download_manager.DownloadConfig = DownloadConfig
_deprecated_download_manager.DownloadMode = DownloadMode
_deprecated_download_manager.DownloadManager = DownloadManager
_deprecated_info_utils.ExpectedMoreDownloadedFiles = ExpectedMoreDownloadedFiles
_deprecated_info_utils.ExpectedMoreSplits = ExpectedMoreSplits
_deprecated_info_utils.UnexpectedDownloadedFile = UnexpectedDownloadedFile
_deprecated_info_utils.UnexpectedSplits = UnexpectedSplits

del _arrow_dataset, _utils, _deprecated_download_manager
del _deprecated_info_utils, ExpectedMoreDownloadedFiles, ExpectedMoreSplits, UnexpectedDownloadedFile, UnexpectedSplits
