from __future__ import annotations

from collections.abc import Callable
from collections.abc import Container
from collections.abc import Sequence
import copy
import threading
from typing import Any

import optuna
from optuna import distributions
from optuna._typing import JSONSerializable
from optuna.storages import BaseStorage
from optuna.storages._heartbeat import BaseHeartbeat
from optuna.storages._rdb.storage import RDBStorage
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class _StudyInfo:
    def __init__(self) -> None:
        # Trial number to corresponding FrozenTrial.
        self.trials: dict[int, FrozenTrial] = {}
        # A list of trials and the last trial number which require storage access to read latest
        # attributes.
        self.unfinished_trial_ids: set[int] = set()
        self.last_finished_trial_id: int = -1
        # Cache distributions to avoid storage access on distribution consistency check.
        self.param_distribution: dict[str, distributions.BaseDistribution] = {}
        self.directions: list[StudyDirection] | None = None
        self.name: str | None = None


class _CachedStorage(BaseStorage, BaseHeartbeat):
    """A wrapper class of storage backends.

    This class is used in :func:`~optuna.get_storage` function and automatically
    wraps :class:`~optuna.storages.RDBStorage` class.

    :class:`~optuna.storages._CachedStorage` meets the following **Data persistence** requirements.

    **Data persistence**

    :class:`~optuna.storages._CachedStorage` does not guarantee that write operations are logged
    into a persistent storage, even when write methods succeed.
    Thus, when process failure occurs, some writes might be lost.
    As exceptions, when a persistent storage is available, any writes on any attributes
    of `Study` and writes on `state` of `Trial` are guaranteed to be persistent.
    Additionally, any preceding writes on any attributes of `Trial` are guaranteed to
    be written into a persistent storage before writes on `state` of `Trial` succeed.
    The same applies for `param`, `user_attrs', 'system_attrs' and 'intermediate_values`
    attributes.

    Args:
        backend:
            :class:`~optuna.storages.RDBStorage` class instance to wrap.
    """

    def __init__(self, backend: RDBStorage) -> None:
        self._backend = backend
        self._studies: dict[int, _StudyInfo] = {}
        self._trial_id_to_study_id_and_number: dict[int, tuple[int, int]] = {}
        self._study_id_and_number_to_trial_id: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        study_id = self._backend.create_new_study(directions=directions, study_name=study_name)
        with self._lock:
            study = _StudyInfo()
            study.name = study_name
            study.directions = list(directions)
            self._studies[study_id] = study

        return study_id

    def delete_study(self, study_id: int) -> None:
        with self._lock:
            if study_id in self._studies:
                for trial_number in self._studies[study_id].trials:
                    trial_id = self._study_id_and_number_to_trial_id.get((study_id, trial_number))
                    if trial_id in self._trial_id_to_study_id_and_number:
                        del self._trial_id_to_study_id_and_number[trial_id]
                    if (study_id, trial_number) in self._study_id_and_number_to_trial_id:
                        del self._study_id_and_number_to_trial_id[(study_id, trial_number)]
                del self._studies[study_id]

        self._backend.delete_study(study_id)

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        self._backend.set_study_user_attr(study_id, key, value)

    def set_study_system_attr(self, study_id: int, key: str, value: JSONSerializable) -> None:
        self._backend.set_study_system_attr(study_id, key, value)

    def get_study_id_from_name(self, study_name: str) -> int:
        return self._backend.get_study_id_from_name(study_name)

    def get_study_name_from_id(self, study_id: int) -> str:
        with self._lock:
            if study_id in self._studies:
                name = self._studies[study_id].name
                if name is not None:
                    return name

        name = self._backend.get_study_name_from_id(study_id)
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            self._studies[study_id].name = name
        return name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        with self._lock:
            if study_id in self._studies:
                directions = self._studies[study_id].directions
                if directions is not None:
                    return directions

        directions = self._backend.get_study_directions(study_id)
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            self._studies[study_id].directions = directions
        return directions

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        return self._backend.get_study_user_attrs(study_id)

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        return self._backend.get_study_system_attrs(study_id)

    def get_all_studies(self) -> list[FrozenStudy]:
        return self._backend.get_all_studies()

    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        frozen_trial = self._backend._create_new_trial(study_id, template_trial)
        trial_id = frozen_trial._trial_id
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            study = self._studies[study_id]
            self._add_trials_to_cache(study_id, [frozen_trial])
            # Since finished trials will not be modified by any worker, we do not
            # need storage access for them.
            if frozen_trial.state.is_finished():
                study.last_finished_trial_id = max(study.last_finished_trial_id, trial_id)
            else:
                study.unfinished_trial_ids.add(trial_id)
        return trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:
        self._backend.set_trial_param(trial_id, param_name, param_value_internal, distribution)

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        key = (study_id, trial_number)
        with self._lock:
            if key in self._study_id_and_number_to_trial_id:
                return self._study_id_and_number_to_trial_id[key]

        return self._backend.get_trial_id_from_study_id_trial_number(study_id, trial_number)

    def get_best_trial(self, study_id: int) -> FrozenTrial:
        return self._backend.get_best_trial(study_id)

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        return self._backend.set_trial_state_values(trial_id, state=state, values=values)

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        self._backend.set_trial_intermediate_value(trial_id, step, intermediate_value)

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        self._backend.set_trial_user_attr(trial_id, key=key, value=value)

    def set_trial_system_attr(self, trial_id: int, key: str, value: JSONSerializable) -> None:
        self._backend.set_trial_system_attr(trial_id, key=key, value=value)

    def _get_cached_trial(self, trial_id: int) -> FrozenTrial | None:
        if trial_id not in self._trial_id_to_study_id_and_number:
            return None
        study_id, number = self._trial_id_to_study_id_and_number[trial_id]
        study = self._studies[study_id]
        return study.trials[number] if trial_id not in study.unfinished_trial_ids else None

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with self._lock:
            trial = self._get_cached_trial(trial_id)
            if trial is not None:
                return trial

        return self._backend.get_trial(trial_id)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        self._read_trials_from_remote_storage(study_id)

        with self._lock:
            study = self._studies[study_id]
            # We need to sort trials by their number because some samplers assume this behavior.
            # The following two lines are latency-sensitive.

            trials: dict[int, FrozenTrial] | list[FrozenTrial]

            if states is not None:
                trials = {number: t for number, t in study.trials.items() if t.state in states}
            else:
                trials = study.trials
            trials = list(sorted(trials.values(), key=lambda t: t.number))
            return copy.deepcopy(trials) if deepcopy else trials

    def _read_trials_from_remote_storage(self, study_id: int) -> None:
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            study = self._studies[study_id]
            trials = self._backend._get_trials(
                study_id,
                states=None,
                included_trial_ids=study.unfinished_trial_ids,
                trial_id_greater_than=study.last_finished_trial_id,
            )
            if not trials:
                return

            self._add_trials_to_cache(study_id, trials)
            for trial in trials:
                if not trial.state.is_finished():
                    study.unfinished_trial_ids.add(trial._trial_id)
                    continue

                study.last_finished_trial_id = max(study.last_finished_trial_id, trial._trial_id)
                if trial._trial_id in study.unfinished_trial_ids:
                    study.unfinished_trial_ids.remove(trial._trial_id)

    def _add_trials_to_cache(self, study_id: int, trials: list[FrozenTrial]) -> None:
        study = self._studies[study_id]
        for trial in trials:
            self._trial_id_to_study_id_and_number[trial._trial_id] = (
                study_id,
                trial.number,
            )
            self._study_id_and_number_to_trial_id[(study_id, trial.number)] = trial._trial_id
            study.trials[trial.number] = trial

    def record_heartbeat(self, trial_id: int) -> None:
        self._backend.record_heartbeat(trial_id)

    def _get_stale_trial_ids(self, study_id: int) -> list[int]:
        return self._backend._get_stale_trial_ids(study_id)

    def get_heartbeat_interval(self) -> int | None:
        return self._backend.get_heartbeat_interval()

    def get_failed_trial_callback(self) -> Callable[["optuna.Study", FrozenTrial], None] | None:
        return self._backend.get_failed_trial_callback()
