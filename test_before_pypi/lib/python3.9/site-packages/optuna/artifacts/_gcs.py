from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO

with try_import() as _imports:
    import google.cloud.storage


@experimental_class("3.4.0")
class GCSArtifactStore:
    """An artifact backend for Google Cloud Storage (GCS).

    Args:
        bucket_name:
            The name of the bucket to store artifacts.

        client:
            A google-cloud-storage ``Client`` to use for storage operations. If not specified, a
            new client will be created with default settings.

    Example:
        .. code-block:: python

            import optuna
            from optuna.artifacts import GCSArtifactStore, upload_artifact


            artifact_backend = GCSArtifactStore("my-bucket")


            def objective(trial: optuna.Trial) -> float:
                ... = trial.suggest_float("x", -10, 10)
                file_path = generate_example(...)
                upload_artifact(
                    artifact_store=artifact_store,
                    file_path=file_path,
                    study_or_trial=trial,
                )
                return ...

        Before running this code, you will have to install ``gcloud`` and run

        .. code-block:: bash

            gcloud auth application-default login

        so that the Cloud Storage library can automatically find the credential.
    """

    def __init__(
        self,
        bucket_name: str,
        client: google.cloud.storage.Client | None = None,
    ) -> None:
        _imports.check()
        self.bucket_name = bucket_name
        self.client = client or google.cloud.storage.Client()
        self.bucket_obj = self.client.bucket(bucket_name)

    def open_reader(self, artifact_id: str) -> "BinaryIO":
        blob = self.bucket_obj.get_blob(artifact_id)

        if blob is None:
            raise ArtifactNotFound(
                f"Artifact storage with bucket: {self.bucket_name}, artifact_id: {artifact_id} was"
                " not found"
            )

        body = blob.download_as_bytes()
        return BytesIO(body)

    def write(self, artifact_id: str, content_body: "BinaryIO") -> None:
        blob = self.bucket_obj.blob(artifact_id)
        data = content_body.read()
        blob.upload_from_string(data)

    def remove(self, artifact_id: str) -> None:
        self.bucket_obj.delete_blob(artifact_id)


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that GCS3ArtifactStore implements all abstract methods
    # in ArtifactStore.
    from optuna.artifacts._protocol import ArtifactStore

    _: ArtifactStore = GCSArtifactStore("")
