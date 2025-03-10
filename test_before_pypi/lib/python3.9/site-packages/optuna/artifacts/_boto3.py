from __future__ import annotations

import io
import shutil
from typing import TYPE_CHECKING

from optuna._imports import try_import
from optuna.artifacts.exceptions import ArtifactNotFound


if TYPE_CHECKING:
    from typing import BinaryIO

    from mypy_boto3_s3 import S3Client

with try_import() as _imports:
    import boto3
    from botocore.exceptions import ClientError


class Boto3ArtifactStore:
    """An artifact backend for Boto3.

    Args:
        bucket_name:
            The name of the bucket to store artifacts.

        client:
            A Boto3 client to use for storage operations. If not specified, a new client will
            be created.

        avoid_buf_copy:
            If True, skip procedure to copy the content of the source file object to a buffer
            before uploading it to S3 ins. This is default to False because using
            ``upload_fileobj()`` method of Boto3 client might close the source file object.

    Example:
        .. code-block:: python

            import optuna
            from optuna.artifacts import upload_artifact
            from optuna.artifacts import Boto3ArtifactStore


            artifact_store = Boto3ArtifactStore("my-bucket")


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
        self, bucket_name: str, client: S3Client | None = None, *, avoid_buf_copy: bool = False
    ) -> None:
        _imports.check()
        self.bucket = bucket_name
        self.client = client or boto3.client("s3")
        # This flag is added to avoid that upload_fileobj() method of Boto3 client may close the
        # source file object. See https://github.com/boto/boto3/issues/929.
        self._avoid_buf_copy = avoid_buf_copy

    def open_reader(self, artifact_id: str) -> BinaryIO:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=artifact_id)
        except ClientError as e:
            if _is_not_found_error(e):
                raise ArtifactNotFound(
                    f"Artifact storage with bucket: {self.bucket}, artifact_id: {artifact_id} was"
                    " not found"
                ) from e
            raise
        body = obj.get("Body")
        assert body is not None
        return body

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        fsrc: BinaryIO = content_body
        if not self._avoid_buf_copy:
            buf = io.BytesIO()
            shutil.copyfileobj(content_body, buf)
            buf.seek(0)
            fsrc = buf
        self.client.upload_fileobj(fsrc, self.bucket, artifact_id)

    def remove(self, artifact_id: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=artifact_id)


def _is_not_found_error(e: ClientError) -> bool:
    error_code = e.response.get("Error", {}).get("Code")
    http_status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return error_code == "NoSuchKey" or http_status_code == 404


if TYPE_CHECKING:
    # A mypy-runtime assertion to ensure that Boto3ArtifactStore implements all abstract methods
    # in ArtifactStore.
    from optuna.artifacts._protocol import ArtifactStore

    _: ArtifactStore = Boto3ArtifactStore("")
