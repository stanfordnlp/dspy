from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from optuna.artifacts.exceptions import ArtifactNotFound


_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import BinaryIO

    from optuna.artifacts._protocol import ArtifactStore


class Backoff:
    """An artifact store's middleware for exponential backoff.

    Example:
       .. code-block:: python

           import optuna
           from optuna.artifacts import upload_artifact
           from optuna.artifacts import Boto3ArtifactStore
           from optuna.artifacts import Backoff


           artifact_store = Backoff(Boto3ArtifactStore("my-bucket"))


           def objective(trial: optuna.Trial) -> float:
               ... = trial.suggest_float("x", -10, 10)
               file_path = generate_example(...)
               upload_artifact(
                   artifact_store=artifact_store,
                   file_path=file_path,
                   study_or_trial=trial,
               )
               return ...
    """

    def __init__(
        self,
        backend: ArtifactStore,
        *,
        max_retries: int = 10,
        multiplier: float = 2,
        min_delay: float = 0.1,
        max_delay: float = 30,
    ) -> None:
        # Default sleep seconds:
        # 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 30
        self._backend = backend
        assert max_retries > 0
        assert multiplier > 0
        assert min_delay > 0
        assert max_delay > min_delay
        self._max_retries = max_retries
        self._multiplier = multiplier
        self._min_delay = min_delay
        self._max_delay = max_delay

    def _get_sleep_secs(self, n_retry: int) -> float:
        return min(self._min_delay * self._multiplier**n_retry, self._max_delay)

    def open_reader(self, artifact_id: str) -> BinaryIO:
        for i in range(self._max_retries):
            try:
                return self._backend.open_reader(artifact_id)
            except ArtifactNotFound:
                raise
            except Exception as e:
                if i == self._max_retries - 1:
                    raise
                else:
                    _logger.error(f"Failed to open artifact={artifact_id} n_retry={i}", exc_info=e)
            time.sleep(self._get_sleep_secs(i))
        assert False, "must not reach here"

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        for i in range(self._max_retries):
            try:
                self._backend.write(artifact_id, content_body)
                break
            except ArtifactNotFound:
                raise
            except Exception as e:
                if i == self._max_retries - 1:
                    raise
                else:
                    _logger.error(f"Failed to open artifact={artifact_id} n_retry={i}", exc_info=e)
            content_body.seek(0)
            time.sleep(self._get_sleep_secs(i))

    def remove(self, artifact_id: str) -> None:
        for i in range(self._max_retries):
            try:
                self._backend.remove(artifact_id)
            except ArtifactNotFound:
                raise
            except Exception as e:
                if i == self._max_retries - 1:
                    raise
                else:
                    _logger.error(f"Failed to delete artifact={artifact_id}", exc_info=e)
            time.sleep(self._get_sleep_secs(i))


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that the Backoff middleware implements
    # all abstract methods in ArtifactStore.
    from optuna.artifacts import FileSystemArtifactStore

    _: ArtifactStore = Backoff(FileSystemArtifactStore("."))
