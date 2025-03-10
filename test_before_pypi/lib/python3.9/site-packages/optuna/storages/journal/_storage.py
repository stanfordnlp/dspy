from __future__ import annotations

from collections.abc import Container
from collections.abc import Sequence
import copy
import datetime
import enum
import pickle
import threading
from typing import Any
import uuid

import optuna
from optuna._typing import JSONSerializable
from optuna.distributions import BaseDistribution
from optuna.distributions import check_distribution_compatibility
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages.journal._base import BaseJournalBackend
from optuna.storages.journal._base import BaseJournalSnapshot
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)

NOT_FOUND_MSG = "Record does not exist."
# A heuristic interval number to dump snapshots
SNAPSHOT_INTERVAL = 100


class JournalOperation(enum.IntEnum):
    CREATE_STUDY = 0
    DELETE_STUDY = 1
    SET_STUDY_USER_ATTR = 2
    SET_STUDY_SYSTEM_ATTR = 3
    CREATE_TRIAL = 4
    SET_TRIAL_PARAM = 5
    SET_TRIAL_STATE_VALUES = 6
    SET_TRIAL_INTERMEDIATE_VALUE = 7
    SET_TRIAL_USER_ATTR = 8
    SET_TRIAL_SYSTEM_ATTR = 9


class JournalStorage(BaseStorage):
    """Storage class for Journal storage backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Journal storage writes a record of every operation to the database as it is executed and
    at the same time, keeps a latest snapshot of the database in-memory. If the database crashes
    for any reason, the storage can re-establish the contents in memory by replaying the
    operations stored from the beginning.

    Journal storage has several benefits over the conventional value logging storages.

    1. The number of IOs can be reduced because of larger granularity of logs.
    2. Journal storage has simpler backend API than value logging storage.
    3. Journal storage keeps a snapshot in-memory so no need to add more cache.

    Example:

        .. code::

            import optuna


            def objective(trial): ...


            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend("./optuna_journal_storage.log")
            )

            study = optuna.create_study(storage=storage)
            study.optimize(objective)

    In a Windows environment, an error message "A required privilege is not held by the
    client" may appear. In this case, you can solve the problem with creating storage
    by specifying :class:`~optuna.storages.journal.JournalFileOpenLock` as follows.

    .. code::

        file_path = "./optuna_journal_storage.log"
        lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj),
        )
    """

    def __init__(self, log_storage: BaseJournalBackend) -> None:
        self._worker_id_prefix = str(uuid.uuid4()) + "-"
        self._backend = log_storage
        self._thread_lock = threading.Lock()
        self._replay_result = JournalStorageReplayResult(self._worker_id_prefix)

        with self._thread_lock:
            if isinstance(self._backend, BaseJournalSnapshot):
                snapshot = self._backend.load_snapshot()
                if snapshot is not None:
                    self.restore_replay_result(snapshot)
            self._sync_with_backend()

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_worker_id_prefix"]
        del state["_replay_result"]
        del state["_thread_lock"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._worker_id_prefix = str(uuid.uuid4()) + "-"
        self._replay_result = JournalStorageReplayResult(self._worker_id_prefix)
        self._thread_lock = threading.Lock()

    def restore_replay_result(self, snapshot: bytes) -> None:
        try:
            r: JournalStorageReplayResult | None = pickle.loads(snapshot)
        except (pickle.UnpicklingError, KeyError):
            _logger.warning("Failed to restore `JournalStorageReplayResult`.")
            return
        if r is None:
            return
        if not isinstance(r, JournalStorageReplayResult):
            _logger.warning("The restored object is not `JournalStorageReplayResult`.")
            return
        r._worker_id_prefix = self._worker_id_prefix
        r._worker_id_to_owned_trial_id = {}
        r._last_created_trial_id_by_this_process = -1
        self._replay_result = r

    def _write_log(self, op_code: int, extra_fields: dict[str, Any]) -> None:
        worker_id = self._replay_result.worker_id
        self._backend.append_logs([{"op_code": op_code, "worker_id": worker_id, **extra_fields}])

    def _sync_with_backend(self) -> None:
        logs = self._backend.read_logs(self._replay_result.log_number_read)
        self._replay_result.apply_logs(logs)

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        study_name = study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())

        with self._thread_lock:
            self._write_log(
                JournalOperation.CREATE_STUDY, {"study_name": study_name, "directions": directions}
            )
            self._sync_with_backend()

            for frozen_study in self._replay_result.get_all_studies():
                if frozen_study.study_name != study_name:
                    continue

                _logger.info("A new study created in Journal with name: {}".format(study_name))
                study_id = frozen_study._study_id

                # Dump snapshot here.
                if (
                    isinstance(self._backend, BaseJournalSnapshot)
                    and study_id != 0
                    and study_id % SNAPSHOT_INTERVAL == 0
                ):
                    self._backend.save_snapshot(pickle.dumps(self._replay_result))

                return study_id
            assert False, "Should not reach."

    def delete_study(self, study_id: int) -> None:
        with self._thread_lock:
            self._write_log(JournalOperation.DELETE_STUDY, {"study_id": study_id})
            self._sync_with_backend()

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        log: dict[str, Any] = {"study_id": study_id, "user_attr": {key: value}}
        with self._thread_lock:
            self._write_log(JournalOperation.SET_STUDY_USER_ATTR, log)
            self._sync_with_backend()

    def set_study_system_attr(self, study_id: int, key: str, value: JSONSerializable) -> None:
        log: dict[str, Any] = {"study_id": study_id, "system_attr": {key: value}}
        with self._thread_lock:
            self._write_log(JournalOperation.SET_STUDY_SYSTEM_ATTR, log)
            self._sync_with_backend()

    def get_study_id_from_name(self, study_name: str) -> int:
        with self._thread_lock:
            self._sync_with_backend()
            for study in self._replay_result.get_all_studies():
                if study.study_name == study_name:
                    return study._study_id
            raise KeyError(NOT_FOUND_MSG)

    def get_study_name_from_id(self, study_id: int) -> str:
        with self._thread_lock:
            self._sync_with_backend()
            return self._replay_result.get_study(study_id).study_name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        with self._thread_lock:
            self._sync_with_backend()
            return self._replay_result.get_study(study_id).directions

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        with self._thread_lock:
            self._sync_with_backend()
            return self._replay_result.get_study(study_id).user_attrs

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        with self._thread_lock:
            self._sync_with_backend()
            return self._replay_result.get_study(study_id).system_attrs

    def get_all_studies(self) -> list[FrozenStudy]:
        with self._thread_lock:
            self._sync_with_backend()
            return copy.deepcopy(self._replay_result.get_all_studies())

    # Basic trial manipulation
    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        log: dict[str, Any] = {
            "study_id": study_id,
            "datetime_start": datetime.datetime.now().isoformat(timespec="microseconds"),
        }

        if template_trial:
            log["state"] = template_trial.state
            if template_trial.values is not None and len(template_trial.values) > 1:
                log["value"] = None
                log["values"] = template_trial.values
            else:
                log["value"] = template_trial.value
                log["values"] = None
            if template_trial.datetime_start:
                log["datetime_start"] = template_trial.datetime_start.isoformat(
                    timespec="microseconds"
                )
            else:
                log["datetime_start"] = None
            if template_trial.datetime_complete:
                log["datetime_complete"] = template_trial.datetime_complete.isoformat(
                    timespec="microseconds"
                )

            log["distributions"] = {
                k: distribution_to_json(dist) for k, dist in template_trial.distributions.items()
            }
            log["params"] = {
                k: template_trial.distributions[k].to_internal_repr(param)
                for k, param in template_trial.params.items()
            }
            log["user_attrs"] = template_trial.user_attrs
            log["system_attrs"] = template_trial.system_attrs
            log["intermediate_values"] = template_trial.intermediate_values

        with self._thread_lock:
            self._write_log(JournalOperation.CREATE_TRIAL, log)
            self._sync_with_backend()
            trial_id = self._replay_result._last_created_trial_id_by_this_process

        # Dump snapshot here.
        if (
            isinstance(self._backend, BaseJournalSnapshot)
            and trial_id != 0
            and trial_id % SNAPSHOT_INTERVAL == 0
        ):
            self._backend.save_snapshot(pickle.dumps(self._replay_result))
        return trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        log: dict[str, Any] = {
            "trial_id": trial_id,
            "param_name": param_name,
            "param_value_internal": param_value_internal,
            "distribution": distribution_to_json(distribution),
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_PARAM, log)
            self._sync_with_backend()

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        with self._thread_lock:
            self._sync_with_backend()
            if len(self._replay_result._study_id_to_trial_ids[study_id]) <= trial_number:
                raise KeyError(
                    "No trial with trial number {} exists in study with study_id {}.".format(
                        trial_number, study_id
                    )
                )
            return self._replay_result._study_id_to_trial_ids[study_id][trial_number]

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        log: dict[str, Any] = {
            "trial_id": trial_id,
            "state": state,
            "values": values,
        }

        if state == TrialState.RUNNING:
            log["datetime_start"] = datetime.datetime.now().isoformat(timespec="microseconds")
        elif state.is_finished():
            log["datetime_complete"] = datetime.datetime.now().isoformat(timespec="microseconds")

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_STATE_VALUES, log)
            self._sync_with_backend()

            if state == TrialState.RUNNING and trial_id != self._replay_result.owned_trial_id:
                return False
            else:
                return True

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        log: dict[str, Any] = {
            "trial_id": trial_id,
            "step": step,
            "intermediate_value": intermediate_value,
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE, log)
            self._sync_with_backend()

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        log: dict[str, Any] = {
            "trial_id": trial_id,
            "user_attr": {key: value},
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_USER_ATTR, log)
            self._sync_with_backend()

    def set_trial_system_attr(self, trial_id: int, key: str, value: JSONSerializable) -> None:
        log: dict[str, Any] = {
            "trial_id": trial_id,
            "system_attr": {key: value},
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_SYSTEM_ATTR, log)
            self._sync_with_backend()

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with self._thread_lock:
            self._sync_with_backend()
            return self._replay_result.get_trial(trial_id)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        with self._thread_lock:
            self._sync_with_backend()
            frozen_trials = self._replay_result.get_all_trials(study_id, states)
            if deepcopy:
                return copy.deepcopy(frozen_trials)
            return frozen_trials


class JournalStorageReplayResult:
    def __init__(self, worker_id_prefix: str) -> None:
        self.log_number_read = 0
        self._worker_id_prefix = worker_id_prefix
        self._studies: dict[int, FrozenStudy] = {}
        self._trials: dict[int, FrozenTrial] = {}

        self._study_id_to_trial_ids: dict[int, list[int]] = {}
        self._trial_id_to_study_id: dict[int, int] = {}
        self._next_study_id: int = 0
        self._worker_id_to_owned_trial_id: dict[str, int] = {}

    def apply_logs(self, logs: list[dict[str, Any]]) -> None:
        for log in logs:
            self.log_number_read += 1
            op = log["op_code"]
            if op == JournalOperation.CREATE_STUDY:
                self._apply_create_study(log)
            elif op == JournalOperation.DELETE_STUDY:
                self._apply_delete_study(log)
            elif op == JournalOperation.SET_STUDY_USER_ATTR:
                self._apply_set_study_user_attr(log)
            elif op == JournalOperation.SET_STUDY_SYSTEM_ATTR:
                self._apply_set_study_system_attr(log)
            elif op == JournalOperation.CREATE_TRIAL:
                self._apply_create_trial(log)
            elif op == JournalOperation.SET_TRIAL_PARAM:
                self._apply_set_trial_param(log)
            elif op == JournalOperation.SET_TRIAL_STATE_VALUES:
                self._apply_set_trial_state_values(log)
            elif op == JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE:
                self._apply_set_trial_intermediate_value(log)
            elif op == JournalOperation.SET_TRIAL_USER_ATTR:
                self._apply_set_trial_user_attr(log)
            elif op == JournalOperation.SET_TRIAL_SYSTEM_ATTR:
                self._apply_set_trial_system_attr(log)
            else:
                assert False, "Should not reach."

    def get_study(self, study_id: int) -> FrozenStudy:
        if study_id not in self._studies:
            raise KeyError(NOT_FOUND_MSG)
        return self._studies[study_id]

    def get_all_studies(self) -> list[FrozenStudy]:
        return list(self._studies.values())

    def get_trial(self, trial_id: int) -> FrozenTrial:
        if trial_id not in self._trials:
            raise KeyError(NOT_FOUND_MSG)
        return self._trials[trial_id]

    def get_all_trials(
        self, study_id: int, states: Container[TrialState] | None
    ) -> list[FrozenTrial]:
        if study_id not in self._studies:
            raise KeyError(NOT_FOUND_MSG)

        frozen_trials: list[FrozenTrial] = []
        for trial_id in self._study_id_to_trial_ids[study_id]:
            trial = self._trials[trial_id]
            if states is None or trial.state in states:
                frozen_trials.append(trial)
        return frozen_trials

    @property
    def worker_id(self) -> str:
        return self._worker_id_prefix + str(threading.get_ident())

    @property
    def owned_trial_id(self) -> int | None:
        return self._worker_id_to_owned_trial_id.get(self.worker_id)

    def _is_issued_by_this_worker(self, log: dict[str, Any]) -> bool:
        return log["worker_id"] == self.worker_id

    def _study_exists(self, study_id: int, log: dict[str, Any]) -> bool:
        if study_id in self._studies:
            return True
        if self._is_issued_by_this_worker(log):
            raise KeyError(NOT_FOUND_MSG)
        return False

    def _apply_create_study(self, log: dict[str, Any]) -> None:
        study_name = log["study_name"]
        directions = [StudyDirection(d) for d in log["directions"]]

        if study_name in [s.study_name for s in self._studies.values()]:
            if self._is_issued_by_this_worker(log):
                raise DuplicatedStudyError(
                    "Another study with name '{}' already exists. "
                    "Please specify a different name, or reuse the existing one "
                    "by setting `load_if_exists` (for Python API) or "
                    "`--skip-if-exists` flag (for CLI).".format(study_name)
                )
            return

        study_id = self._next_study_id
        self._next_study_id += 1

        self._studies[study_id] = FrozenStudy(
            study_name=study_name,
            direction=None,
            user_attrs={},
            system_attrs={},
            study_id=study_id,
            directions=directions,
        )
        self._study_id_to_trial_ids[study_id] = []

    def _apply_delete_study(self, log: dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            fs = self._studies.pop(study_id)
            assert fs._study_id == study_id

    def _apply_set_study_user_attr(self, log: dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            assert len(log["user_attr"]) == 1
            self._studies[study_id].user_attrs.update(log["user_attr"])

    def _apply_set_study_system_attr(self, log: dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            assert len(log["system_attr"]) == 1
            self._studies[study_id].system_attrs.update(log["system_attr"])

    def _apply_create_trial(self, log: dict[str, Any]) -> None:
        study_id = log["study_id"]

        if not self._study_exists(study_id, log):
            return

        trial_id = len(self._trials)
        distributions = {}
        if "distributions" in log:
            distributions = {k: json_to_distribution(v) for k, v in log["distributions"].items()}
        params = {}
        if "params" in log:
            params = {k: distributions[k].to_external_repr(p) for k, p in log["params"].items()}
        if log["datetime_start"] is not None:
            datetime_start = datetime.datetime.fromisoformat(log["datetime_start"])
        else:
            datetime_start = None
        if "datetime_complete" in log:
            datetime_complete = datetime.datetime.fromisoformat(log["datetime_complete"])
        else:
            datetime_complete = None

        self._trials[trial_id] = FrozenTrial(
            trial_id=trial_id,
            number=len(self._study_id_to_trial_ids[study_id]),
            state=TrialState(log.get("state", TrialState.RUNNING.value)),
            params=params,
            distributions=distributions,
            user_attrs=log.get("user_attrs", {}),
            system_attrs=log.get("system_attrs", {}),
            value=log.get("value", None),
            intermediate_values={int(k): v for k, v in log.get("intermediate_values", {}).items()},
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            values=log.get("values", None),
        )

        self._study_id_to_trial_ids[study_id].append(trial_id)
        self._trial_id_to_study_id[trial_id] = study_id

        if self._is_issued_by_this_worker(log):
            self._last_created_trial_id_by_this_process = trial_id
            if self._trials[trial_id].state == TrialState.RUNNING:
                self._worker_id_to_owned_trial_id[self.worker_id] = trial_id

    def _apply_set_trial_param(self, log: dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if not self._trial_exists_and_updatable(trial_id, log):
            return

        param_name = log["param_name"]
        param_value_internal = log["param_value_internal"]
        distribution = json_to_distribution(log["distribution"])

        study_id = self._trial_id_to_study_id[trial_id]

        for prev_trial_id in self._study_id_to_trial_ids[study_id]:
            prev_trial = self._trials[prev_trial_id]
            if param_name in prev_trial.params.keys():
                try:
                    check_distribution_compatibility(
                        prev_trial.distributions[param_name], distribution
                    )
                except Exception:
                    if self._is_issued_by_this_worker(log):
                        raise
                    return
                break

        trial = copy.copy(self._trials[trial_id])
        trial.params = {
            **copy.copy(trial.params),
            param_name: distribution.to_external_repr(param_value_internal),
        }
        trial.distributions = {**copy.copy(trial.distributions), param_name: distribution}
        self._trials[trial_id] = trial

    def _apply_set_trial_state_values(self, log: dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if not self._trial_exists_and_updatable(trial_id, log):
            return

        state = TrialState(log["state"])
        if state == self._trials[trial_id].state and state == TrialState.RUNNING:
            return

        trial = copy.copy(self._trials[trial_id])
        if state == TrialState.RUNNING:
            trial.datetime_start = datetime.datetime.fromisoformat(log["datetime_start"])
            if self._is_issued_by_this_worker(log):
                self._worker_id_to_owned_trial_id[self.worker_id] = trial_id
        if state.is_finished():
            trial.datetime_complete = datetime.datetime.fromisoformat(log["datetime_complete"])
        trial.state = state
        if log["values"] is not None:
            trial.values = log["values"]

        self._trials[trial_id] = trial

    def _apply_set_trial_intermediate_value(self, log: dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            trial = copy.copy(self._trials[trial_id])
            trial.intermediate_values = {
                **copy.copy(trial.intermediate_values),
                log["step"]: log["intermediate_value"],
            }
            self._trials[trial_id] = trial

    def _apply_set_trial_user_attr(self, log: dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            assert len(log["user_attr"]) == 1
            trial = copy.copy(self._trials[trial_id])
            trial.user_attrs = {**copy.copy(trial.user_attrs), **log["user_attr"]}
            self._trials[trial_id] = trial

    def _apply_set_trial_system_attr(self, log: dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            assert len(log["system_attr"]) == 1
            trial = copy.copy(self._trials[trial_id])
            trial.system_attrs = {
                **copy.copy(trial.system_attrs),
                **log["system_attr"],
            }
            self._trials[trial_id] = trial

    def _trial_exists_and_updatable(self, trial_id: int, log: dict[str, Any]) -> bool:
        if trial_id not in self._trials:
            if self._is_issued_by_this_worker(log):
                raise KeyError(NOT_FOUND_MSG)
            return False
        elif self._trials[trial_id].state.is_finished():
            if self._is_issued_by_this_worker(log):
                raise RuntimeError(
                    "Trial#{} has already finished and can not be updated.".format(
                        self._trials[trial_id].number
                    )
                )
            return False
        else:
            return True
