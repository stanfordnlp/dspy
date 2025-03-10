from typing import Any, Dict, List, Optional, Union

from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
    get_authentication_headers_for_url,
    http_get,
)
from .logging import get_logger


logger = get_logger(__name__)


class DatasetViewerError(DatasetsError):
    """Dataset viewer error.

    Raised when trying to use the dataset viewer HTTP API and when trying to access:
    - a missing dataset, or
    - a private/gated dataset and the user is not authenticated.
    - unavailable /parquet or /info responses
    """


def get_exported_parquet_files(dataset: str, revision: str, token: Optional[Union[str, bool]]) -> List[Dict[str, Any]]:
    """
    Get the dataset exported parquet files
    Docs: https://huggingface.co/docs/datasets-server/parquet
    """
    dataset_viewer_parquet_url = config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/parquet?dataset="
    try:
        parquet_data_files_response = http_get(
            url=dataset_viewer_parquet_url + dataset,
            temp_file=None,
            headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
            max_retries=3,
        )
        parquet_data_files_response.raise_for_status()
        if "X-Revision" in parquet_data_files_response.headers:
            if parquet_data_files_response.headers["X-Revision"] == revision or revision is None:
                parquet_data_files_response_json = parquet_data_files_response.json()
                if (
                    parquet_data_files_response_json.get("partial") is False
                    and not parquet_data_files_response_json.get("pending", True)
                    and not parquet_data_files_response_json.get("failed", True)
                    and "parquet_files" in parquet_data_files_response_json
                ):
                    return parquet_data_files_response_json["parquet_files"]
                else:
                    logger.debug(f"Parquet export for {dataset} is not completely ready yet.")
            else:
                logger.debug(
                    f"Parquet export for {dataset} is available but outdated (revision='{parquet_data_files_response.headers['X-Revision']}')"
                )
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the parquet export doesn't exist
        logger.debug(f"No parquet export for {dataset} available ({type(e).__name__}: {e})")
    raise DatasetViewerError("No exported Parquet files available.")


def get_exported_dataset_infos(
    dataset: str, revision: str, token: Optional[Union[str, bool]]
) -> Dict[str, Dict[str, Any]]:
    """
    Get the dataset information, can be useful to get e.g. the dataset features.
    Docs: https://huggingface.co/docs/datasets-server/info
    """
    dataset_viewer_info_url = config.HF_ENDPOINT.replace("://", "://datasets-server.") + "/info?dataset="
    try:
        info_response = http_get(
            url=dataset_viewer_info_url + dataset,
            temp_file=None,
            headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f"datasets/{dataset}", token=token),
            timeout=100.0,
            max_retries=3,
        )
        info_response.raise_for_status()
        if "X-Revision" in info_response.headers:
            if info_response.headers["X-Revision"] == revision or revision is None:
                info_response = info_response.json()
                if (
                    info_response.get("partial") is False
                    and not info_response.get("pending", True)
                    and not info_response.get("failed", True)
                    and "dataset_info" in info_response
                ):
                    return info_response["dataset_info"]
                else:
                    logger.debug(f"Dataset info for {dataset} is not completely ready yet.")
            else:
                logger.debug(
                    f"Dataset info for {dataset} is available but outdated (revision='{info_response.headers['X-Revision']}')"
                )
    except Exception as e:  # noqa catch any exception of the dataset viewer API and consider the dataset info doesn't exist
        logger.debug(f"No dataset info for {dataset} available ({type(e).__name__}: {e})")
    raise DatasetViewerError("No exported dataset infos available.")
