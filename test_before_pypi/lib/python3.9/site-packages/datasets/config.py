import importlib
import importlib.metadata
import logging
import os
import platform
from pathlib import Path
from typing import Optional

from huggingface_hub import constants
from packaging import version


logger = logging.getLogger(__name__.split(".", 1)[0])  # to avoid circular import from .utils.logging

# Datasets
S3_DATASETS_BUCKET_PREFIX = "https://s3.amazonaws.com/datasets.huggingface.co/datasets/datasets"
CLOUDFRONT_DATASETS_DISTRIB_PREFIX = "https://cdn-datasets.huggingface.co/datasets/datasets"
REPO_DATASETS_URL = "https://raw.githubusercontent.com/huggingface/datasets/{revision}/datasets/{path}/{name}"

# Metrics
S3_METRICS_BUCKET_PREFIX = "https://s3.amazonaws.com/datasets.huggingface.co/datasets/metrics"
CLOUDFRONT_METRICS_DISTRIB_PREFIX = "https://cdn-datasets.huggingface.co/datasets/metric"
REPO_METRICS_URL = "https://raw.githubusercontent.com/huggingface/datasets/{revision}/metrics/{path}/{name}"

# Hub
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HUB_DATASETS_URL = HF_ENDPOINT + "/datasets/{repo_id}/resolve/{revision}/{path}"
HUB_DATASETS_HFFS_URL = "hf://datasets/{repo_id}@{revision}/{path}"
HUB_DEFAULT_VERSION = "main"

PY_VERSION = version.parse(platform.python_version())

# General environment variables accepted values for booleans
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_FALSE_VALUES = {"0", "OFF", "NO", "FALSE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
ENV_VARS_FALSE_AND_AUTO_VALUES = ENV_VARS_FALSE_VALUES.union({"AUTO"})


# Imports
DILL_VERSION = version.parse(importlib.metadata.version("dill"))
FSSPEC_VERSION = version.parse(importlib.metadata.version("fsspec"))
PANDAS_VERSION = version.parse(importlib.metadata.version("pandas"))
PYARROW_VERSION = version.parse(importlib.metadata.version("pyarrow"))
HF_HUB_VERSION = version.parse(importlib.metadata.version("huggingface_hub"))

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_JAX", "AUTO").upper()

TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
    if TORCH_AVAILABLE:
        try:
            TORCH_VERSION = version.parse(importlib.metadata.version("torch"))
            logger.info(f"PyTorch version {TORCH_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling PyTorch because USE_TF is set")

POLARS_VERSION = "N/A"
POLARS_AVAILABLE = importlib.util.find_spec("polars") is not None

if POLARS_AVAILABLE:
    try:
        POLARS_VERSION = version.parse(importlib.metadata.version("polars"))
        logger.info(f"Polars version {POLARS_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass

TF_VERSION = "N/A"
TF_AVAILABLE = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
    if TF_AVAILABLE:
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for package in [
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        ]:
            try:
                TF_VERSION = version.parse(importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError:
                continue
            else:
                break
        else:
            TF_AVAILABLE = False
    if TF_AVAILABLE:
        if TF_VERSION.major < 2:
            logger.info(f"TensorFlow found but with version {TF_VERSION}. `datasets` requires version 2 minimum.")
            TF_AVAILABLE = False
        else:
            logger.info(f"TensorFlow version {TF_VERSION} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")


JAX_VERSION = "N/A"
JAX_AVAILABLE = False

if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    JAX_AVAILABLE = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
    if JAX_AVAILABLE:
        try:
            JAX_VERSION = version.parse(importlib.metadata.version("jax"))
            logger.info(f"JAX version {JAX_VERSION} available.")
        except importlib.metadata.PackageNotFoundError:
            pass
else:
    logger.info("Disabling JAX because USE_JAX is set to False")


USE_BEAM = os.environ.get("USE_BEAM", "AUTO").upper()
BEAM_VERSION = "N/A"
BEAM_AVAILABLE = False
if USE_BEAM in ENV_VARS_TRUE_AND_AUTO_VALUES:
    try:
        BEAM_VERSION = version.parse(importlib.metadata.version("apache_beam"))
        BEAM_AVAILABLE = True
        logger.info(f"Apache Beam version {BEAM_VERSION} available.")
    except importlib.metadata.PackageNotFoundError:
        pass
else:
    logger.info("Disabling Apache Beam because USE_BEAM is set to False")


# Optional tools for data loading
SQLALCHEMY_AVAILABLE = importlib.util.find_spec("sqlalchemy") is not None

# Optional tools for feature decoding
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
IS_OPUS_SUPPORTED = importlib.util.find_spec("soundfile") is not None and version.parse(
    importlib.import_module("soundfile").__libsndfile_version__
) >= version.parse("1.0.31")
IS_MP3_SUPPORTED = importlib.util.find_spec("soundfile") is not None and version.parse(
    importlib.import_module("soundfile").__libsndfile_version__
) >= version.parse("1.1.0")

# Optional compression tools
RARFILE_AVAILABLE = importlib.util.find_spec("rarfile") is not None
ZSTANDARD_AVAILABLE = importlib.util.find_spec("zstandard") is not None
LZ4_AVAILABLE = importlib.util.find_spec("lz4") is not None
PY7ZR_AVAILABLE = importlib.util.find_spec("py7zr") is not None

# Cache location
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))

DEFAULT_HF_DATASETS_CACHE = os.path.join(HF_CACHE_HOME, "datasets")
HF_DATASETS_CACHE = Path(os.getenv("HF_DATASETS_CACHE", DEFAULT_HF_DATASETS_CACHE))

DEFAULT_HF_METRICS_CACHE = os.path.join(HF_CACHE_HOME, "metrics")
HF_METRICS_CACHE = Path(os.getenv("HF_METRICS_CACHE", DEFAULT_HF_METRICS_CACHE))

DEFAULT_HF_MODULES_CACHE = os.path.join(HF_CACHE_HOME, "modules")
HF_MODULES_CACHE = Path(os.getenv("HF_MODULES_CACHE", DEFAULT_HF_MODULES_CACHE))

DOWNLOADED_DATASETS_DIR = "downloads"
DEFAULT_DOWNLOADED_DATASETS_PATH = os.path.join(HF_DATASETS_CACHE, DOWNLOADED_DATASETS_DIR)
DOWNLOADED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_DOWNLOADED_DATASETS_PATH", DEFAULT_DOWNLOADED_DATASETS_PATH))

EXTRACTED_DATASETS_DIR = "extracted"
DEFAULT_EXTRACTED_DATASETS_PATH = os.path.join(DEFAULT_DOWNLOADED_DATASETS_PATH, EXTRACTED_DATASETS_DIR)
EXTRACTED_DATASETS_PATH = Path(os.getenv("HF_DATASETS_EXTRACTED_DATASETS_PATH", DEFAULT_EXTRACTED_DATASETS_PATH))

# Download count for the website
HF_UPDATE_DOWNLOAD_COUNTS = (
    os.environ.get("HF_UPDATE_DOWNLOAD_COUNTS", "AUTO").upper() in ENV_VARS_TRUE_AND_AUTO_VALUES
)

# For downloads and to check remote files metadata
HF_DATASETS_MULTITHREADING_MAX_WORKERS = 16

# Remote dataset scripts support
__HF_DATASETS_TRUST_REMOTE_CODE = os.environ.get("HF_DATASETS_TRUST_REMOTE_CODE", "ask")
HF_DATASETS_TRUST_REMOTE_CODE: Optional[bool] = (
    True
    if __HF_DATASETS_TRUST_REMOTE_CODE.upper() in ENV_VARS_TRUE_VALUES
    else False
    if __HF_DATASETS_TRUST_REMOTE_CODE.upper() in ENV_VARS_FALSE_VALUES
    else None
)
TIME_OUT_REMOTE_CODE = 15

# Dataset viewer API
USE_PARQUET_EXPORT = True

# Batch size constants. For more info, see:
# https://github.com/apache/arrow/blob/master/docs/source/cpp/arrays.rst#size-limitations-and-recommendations)
DEFAULT_MAX_BATCH_SIZE = 1000

# Size of the preloaded record batch in `Dataset.__iter__`
ARROW_READER_BATCH_SIZE_IN_DATASET_ITER = 10

# Max shard size in bytes (e.g. to shard parquet datasets in push_to_hub or download_and_prepare)
MAX_SHARD_SIZE = "500MB"

# Parquet configuration
PARQUET_ROW_GROUP_SIZE_FOR_AUDIO_DATASETS = 100
PARQUET_ROW_GROUP_SIZE_FOR_IMAGE_DATASETS = 100
PARQUET_ROW_GROUP_SIZE_FOR_BINARY_DATASETS = 100

# Offline mode
_offline = os.environ.get("HF_DATASETS_OFFLINE")
HF_HUB_OFFLINE = constants.HF_HUB_OFFLINE if _offline is None else _offline.upper() in ENV_VARS_TRUE_VALUES
HF_DATASETS_OFFLINE = HF_HUB_OFFLINE  # kept for backward-compatibility

# Here, `True` will disable progress bars globally without possibility of enabling it
# programmatically. `False` will enable them without possibility of disabling them.
# If environment variable is not set (None), then the user is free to enable/disable
# them programmatically.
# TL;DR: env variable has priority over code
__HF_DATASETS_DISABLE_PROGRESS_BARS = os.environ.get("HF_DATASETS_DISABLE_PROGRESS_BARS")
HF_DATASETS_DISABLE_PROGRESS_BARS: Optional[bool] = (
    __HF_DATASETS_DISABLE_PROGRESS_BARS.upper() in ENV_VARS_TRUE_VALUES
    if __HF_DATASETS_DISABLE_PROGRESS_BARS is not None
    else None
)

# In-memory
DEFAULT_IN_MEMORY_MAX_SIZE = 0  # Disabled
IN_MEMORY_MAX_SIZE = float(os.environ.get("HF_DATASETS_IN_MEMORY_MAX_SIZE", DEFAULT_IN_MEMORY_MAX_SIZE))

# File names
DATASET_ARROW_FILENAME = "dataset.arrow"
DATASET_INDICES_FILENAME = "indices.arrow"
DATASET_STATE_JSON_FILENAME = "state.json"
DATASET_INFO_FILENAME = "dataset_info.json"
DATASETDICT_INFOS_FILENAME = "dataset_infos.json"
LICENSE_FILENAME = "LICENSE"
METRIC_INFO_FILENAME = "metric_info.json"
DATASETDICT_JSON_FILENAME = "dataset_dict.json"
METADATA_CONFIGS_FIELD = "configs"
REPOCARD_FILENAME = "README.md"
REPOYAML_FILENAME = ".huggingface.yaml"

MODULE_NAME_FOR_DYNAMIC_MODULES = "datasets_modules"

MAX_DATASET_CONFIG_ID_READABLE_LENGTH = 255

# Temporary cache directory prefix
TEMP_CACHE_DIR_PREFIX = "hf_datasets-"

# Streaming
STREAMING_READ_MAX_RETRIES = 20
STREAMING_READ_RETRY_INTERVAL = 5

# Datasets without script
DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE = 200
GLOBBED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE = 10
ARCHIVED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE = 200

# Progress bars
PBAR_REFRESH_TIME_INTERVAL = 0.05  # 20 progress updates per sec

# Maximum number of uploaded files per commit
UPLOADS_MAX_NUMBER_PER_COMMIT = 50

# Backward compatibiliy
MAX_TABLE_NBYTES_FOR_PICKLING = 4 << 30
