from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
import mimetypes
import os
import uuid

from optuna._convert_positional_args import convert_positional_args
from optuna.artifacts._protocol import ArtifactStore
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


ARTIFACTS_ATTR_PREFIX = "artifacts:"
DEFAULT_MIME_TYPE = "application/octet-stream"


@dataclass
class ArtifactMeta:
    """Meta information for an artifact.

    .. note::
        All the artifact meta linked to a study or trial can be listed by
        :func:`~optuna.artifacts.get_all_artifact_meta`.
        The artifact meta can be used for :func:`~optuna.artifacts.download_artifact`.

    Args:
        artifact_id:
            The identifier of the artifact.
        filename:
            The artifact file name used for the upload.
        mimetype:
            A MIME type of the artifact.
            If not specified, the MIME type is guessed from the file extension.
        encoding:
            An encoding of the artifact, which is suitable for use as a Content-Encoding header,
            e.g., gzip. If not specified, the encoding is guessed from the file extension.
    """

    artifact_id: str
    filename: str
    mimetype: str
    encoding: str | None


@convert_positional_args(
    previous_positional_arg_names=["study_or_trial", "file_path", "artifact_store"]
)
def upload_artifact(
    *,
    artifact_store: ArtifactStore,
    file_path: str,
    study_or_trial: Trial | FrozenTrial | Study,
    storage: BaseStorage | None = None,
    mimetype: str | None = None,
    encoding: str | None = None,
) -> str:
    """Upload an artifact to the artifact store.

    Args:
        artifact_store:
            An artifact store.
        file_path:
            A path to the file to be uploaded.
        study_or_trial:
            A :class:`~optuna.trial.Trial` object, a :class:`~optuna.trial.FrozenTrial`, or
            a :class:`~optuna.study.Study` object.
        storage:
            A storage object. This argument is required only if ``study_or_trial`` is
            :class:`~optuna.trial.FrozenTrial`.
        mimetype:
            A MIME type of the artifact. If not specified, the MIME type is guessed from the file
            extension.
        encoding:
            An encoding of the artifact, which is suitable for use as a ``Content-Encoding``
            header (e.g. gzip). If not specified, the encoding is guessed from the file extension.

    Returns:
        An artifact ID.
    """

    filename = os.path.basename(file_path)

    if isinstance(study_or_trial, Trial) and storage is None:
        storage = study_or_trial.storage
    elif isinstance(study_or_trial, Study) and storage is None:
        storage = study_or_trial._storage

    if storage is None:
        raise ValueError("storage is required for FrozenTrial.")

    artifact_id = str(uuid.uuid4())
    guess_mimetype, guess_encoding = mimetypes.guess_type(filename)
    artifact = ArtifactMeta(
        artifact_id=artifact_id,
        filename=filename,
        mimetype=mimetype or guess_mimetype or DEFAULT_MIME_TYPE,
        encoding=encoding or guess_encoding,
    )
    attr_key = ARTIFACTS_ATTR_PREFIX + artifact_id
    if isinstance(study_or_trial, (Trial, FrozenTrial)):
        trial_id = study_or_trial._trial_id
        storage.set_trial_system_attr(trial_id, attr_key, json.dumps(asdict(artifact)))
    else:
        study_id = study_or_trial._study_id
        storage.set_study_system_attr(study_id, attr_key, json.dumps(asdict(artifact)))

    with open(file_path, "rb") as f:
        artifact_store.write(artifact_id, f)
    return artifact_id
