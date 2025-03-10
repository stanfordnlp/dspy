from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO


class FileSystemArtifactStore:
    """An artifact store for file systems.

    Args:
        base_path:
            The base path to a directory to store artifacts.

    Example:
        .. code-block:: python

            import os

            import optuna
            from optuna.artifacts import FileSystemArtifactStore
            from optuna.artifacts import upload_artifact


            base_path = "./artifacts"
            os.makedirs(base_path, exist_ok=True)
            artifact_store = FileSystemArtifactStore(base_path=base_path)


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

    def __init__(self, base_path: str | Path) -> None:
        if isinstance(base_path, str):
            base_path = Path(base_path)
        # TODO(Shinichi): Check if the base_path is valid directory.
        self._base_path = base_path

    def open_reader(self, artifact_id: str) -> BinaryIO:
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            f = open(filepath, "rb")
        except FileNotFoundError as e:
            raise ArtifactNotFound("not found") from e
        return f

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        filepath = os.path.join(self._base_path, artifact_id)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(content_body, f)

    def remove(self, artifact_id: str) -> None:
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            raise ArtifactNotFound("not found") from e


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that LocalArtifactBackend
    # implements all abstract methods in ArtifactBackendProtocol.
    from optuna.artifacts._protocol import ArtifactStore

    _: ArtifactStore = FileSystemArtifactStore("")
