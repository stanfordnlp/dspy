import enum
import os
from typing import Optional

from huggingface_hub.utils import insecure_hashlib

from .. import config
from ..exceptions import (
    ExpectedMoreDownloadedFilesError,
    ExpectedMoreSplitsError,
    NonMatchingChecksumError,
    NonMatchingSplitsSizesError,
    UnexpectedDownloadedFileError,
    UnexpectedSplitsError,
)
from .logging import get_logger


logger = get_logger(__name__)


class VerificationMode(enum.Enum):
    """`Enum` that specifies which verification checks to run.

    The default mode is `BASIC_CHECKS`, which will perform only rudimentary checks to avoid slowdowns
    when generating/downloading a dataset for the first time.

    The verification modes:

    |                           | Verification checks                                                           |
    |---------------------------|------------------------------------------------------------------------------ |
    | `ALL_CHECKS`              | Split checks, uniqueness of the keys yielded in case of the GeneratorBuilder  |
    |                           | and the validity (number of files, checksums, etc.) of downloaded files       |
    | `BASIC_CHECKS` (default)  | Same as `ALL_CHECKS` but without checking downloaded files                    |
    | `NO_CHECKS`               | None                                                                          |

    """

    ALL_CHECKS = "all_checks"
    BASIC_CHECKS = "basic_checks"
    NO_CHECKS = "no_checks"


def verify_checksums(expected_checksums: Optional[dict], recorded_checksums: dict, verification_name=None):
    if expected_checksums is None:
        logger.info("Unable to verify checksums.")
        return
    if len(set(expected_checksums) - set(recorded_checksums)) > 0:
        raise ExpectedMoreDownloadedFilesError(str(set(expected_checksums) - set(recorded_checksums)))
    if len(set(recorded_checksums) - set(expected_checksums)) > 0:
        raise UnexpectedDownloadedFileError(str(set(recorded_checksums) - set(expected_checksums)))
    bad_urls = [url for url in expected_checksums if expected_checksums[url] != recorded_checksums[url]]
    for_verification_name = " for " + verification_name if verification_name is not None else ""
    if len(bad_urls) > 0:
        raise NonMatchingChecksumError(
            f"Checksums didn't match{for_verification_name}:\n"
            f"{bad_urls}\n"
            "Set `verification_mode='no_checks'` to skip checksums verification and ignore this error"
        )
    logger.info("All the checksums matched successfully" + for_verification_name)


def verify_splits(expected_splits: Optional[dict], recorded_splits: dict):
    if expected_splits is None:
        logger.info("Unable to verify splits sizes.")
        return
    if len(set(expected_splits) - set(recorded_splits)) > 0:
        raise ExpectedMoreSplitsError(str(set(expected_splits) - set(recorded_splits)))
    if len(set(recorded_splits) - set(expected_splits)) > 0:
        raise UnexpectedSplitsError(str(set(recorded_splits) - set(expected_splits)))
    bad_splits = [
        {"expected": expected_splits[name], "recorded": recorded_splits[name]}
        for name in expected_splits
        if expected_splits[name].num_examples != recorded_splits[name].num_examples
    ]
    if len(bad_splits) > 0:
        raise NonMatchingSplitsSizesError(str(bad_splits))
    logger.info("All the splits matched successfully.")


def get_size_checksum_dict(path: str, record_checksum: bool = True) -> dict:
    """Compute the file size and the sha256 checksum of a file"""
    if record_checksum:
        m = insecure_hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                m.update(chunk)
            checksum = m.hexdigest()
    else:
        checksum = None
    return {"num_bytes": os.path.getsize(path), "checksum": checksum}


def is_small_dataset(dataset_size):
    """Check if `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.

    Args:
        dataset_size (int): Dataset size in bytes.

    Returns:
        bool: Whether `dataset_size` is smaller than `config.IN_MEMORY_MAX_SIZE`.
    """
    if dataset_size and config.IN_MEMORY_MAX_SIZE:
        return dataset_size < config.IN_MEMORY_MAX_SIZE
    else:
        return False
