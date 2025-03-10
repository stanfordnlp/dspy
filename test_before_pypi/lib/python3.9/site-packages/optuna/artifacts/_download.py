from __future__ import annotations

import os
import shutil

from optuna.artifacts._protocol import ArtifactStore


def download_artifact(*, artifact_store: ArtifactStore, file_path: str, artifact_id: str) -> None:
    """Download an artifact from the artifact store.

    Args:
        artifact_store:
            An artifact store.
        file_path:
            A path to save the downloaded artifact.
        artifact_id:
            The identifier of the artifact to download.
    """
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")

    with artifact_store.open_reader(artifact_id) as reader, open(file_path, "wb") as writer:
        shutil.copyfileobj(reader, writer)
