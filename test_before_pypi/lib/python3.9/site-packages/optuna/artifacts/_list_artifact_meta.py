from __future__ import annotations

import json

from optuna.artifacts._upload import ArtifactMeta
from optuna.artifacts._upload import ARTIFACTS_ATTR_PREFIX
from optuna.storages import BaseStorage
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial


def get_all_artifact_meta(
    study_or_trial: Trial | FrozenTrial | Study, *, storage: BaseStorage | None = None
) -> list[ArtifactMeta]:
    """List the associated artifact information of the provided trial or study.

    Args:
        study_or_trial:
            A :class:`~optuna.trial.Trial` object, a :class:`~optuna.trial.FrozenTrial`, or
            a :class:`~optuna.study.Study` object.
        storage:
            A storage object. This argument is required only if ``study_or_trial`` is
            :class:`~optuna.trial.FrozenTrial`.

    Example:
        An example where this function is useful:

        .. code::

            import os

            import optuna


            # Get the storage that contains the study of interest.
            storage = optuna.storages.get_storage(storage=...)

            # Instantiate the artifact store used for the study.
            # Optuna does not provide the API that stores the used artifact store information, so
            # please manage the information in the user side.
            artifact_store = ...

            # Load study that contains the artifacts of interest.
            study = optuna.load_study(study_name=..., storage=storage)

            # Fetch the best trial.
            best_trial = study.best_trial

            # Fetch all the artifact meta connected to the best trial.
            artifact_metas = optuna.artifacts.get_all_artifact_meta(best_trial, storage=storage)

            download_dir_path = "./best_trial_artifacts/"
            os.makedirs(download_dir_path, exist_ok=True)

            for artifact_meta in artifact_metas:
                download_file_path = os.path.join(download_dir_path, artifact_meta.filename)
                # Download the artifacts to ``download_file_path``.
                optuna.artifacts.download_artifact(
                    artifact_store=artifact_store,
                    artifact_id=artifact_meta.artifact_id,
                    file_path=download_file_path,
                )

    Returns:
        The list of artifact meta in the trial or study.
        Each artifact meta includes ``artifact_id``, ``filename``, ``mimetype``, and ``encoding``.
        Note that if :class:`~optuna.study.Study` is provided, we return the information of the
        artifacts uploaded to ``study``, but not to all the trials in the study.
    """
    if isinstance(study_or_trial, Trial) and storage is None:
        storage = study_or_trial.storage
    elif isinstance(study_or_trial, Study) and storage is None:
        storage = study_or_trial._storage

    if storage is None:
        raise ValueError("storage is required for FrozenTrial.")

    if isinstance(study_or_trial, (Trial, FrozenTrial)):
        system_attrs = storage.get_trial_system_attrs(study_or_trial._trial_id)
    else:
        system_attrs = storage.get_study_system_attrs(study_or_trial._study_id)

    artifact_meta_list: list[ArtifactMeta] = []
    for attr_key, attr_json_string in system_attrs.items():
        if not attr_key.startswith(ARTIFACTS_ATTR_PREFIX):
            continue

        attr_content = json.loads(attr_json_string)
        artifact_meta = ArtifactMeta(
            artifact_id=attr_content["artifact_id"],
            filename=attr_content["filename"],
            mimetype=attr_content["mimetype"],
            encoding=attr_content["encoding"],
        )
        artifact_meta_list.append(artifact_meta)

    return artifact_meta_list
