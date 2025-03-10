from __future__ import annotations

from collections.abc import Container
from collections.abc import Sequence
import copy
import json
import threading
from typing import Any
from typing import TYPE_CHECKING
import uuid

from optuna._experimental import experimental_class
from optuna._imports import _LazyImport
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.exceptions import DuplicatedStudyError
from optuna.storages._base import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    import grpc

    from optuna.storages._grpc import servicer as grpc_servicer
    from optuna.storages._grpc.auto_generated import api_pb2
    from optuna.storages._grpc.auto_generated import api_pb2_grpc
else:
    api_pb2 = _LazyImport("optuna.storages._grpc.auto_generated.api_pb2")
    api_pb2_grpc = _LazyImport("optuna.storages._grpc.auto_generated.api_pb2_grpc")
    grpc = _LazyImport("grpc")
    grpc_servicer = _LazyImport("optuna.storages._grpc.servicer")


@experimental_class("4.2.0")
class GrpcStorageProxy(BaseStorage):
    """gRPC client for :func:`~optuna.storages.run_grpc_proxy_server`.

    Example:

        This is a simple example of using :class:`~optuna.storages.GrpcStorageProxy` with
        :func:`~optuna.storages.run_grpc_proxy_server`.

        .. code::

            import optuna
            from optuna.storages import GrpcStorageProxy

            storage = GrpcStorageProxy(host="localhost", port=13000)
            study = optuna.create_study(storage=storage)

        Please refer to the example in :func:`~optuna.storages.run_grpc_proxy_server` for the
        server side code.

    Args:
        host: The hostname of the gRPC server.
        port: The port of the gRPC server.

    .. warning::

        Currently, gRPC storage proxy in combination with an SQLite3 database may cause unexpected
        behaviors when calling :func:`optuna.delete_study` due to non-invalidated cache.
    """

    def __init__(self, *, host: str = "localhost", port: int = 13000) -> None:
        self._stub = api_pb2_grpc.StorageServiceStub(
            grpc.insecure_channel(
                f"{host}:{port}",
                options=[("grpc.max_receive_message_length", -1)],
            )
        )  # type: ignore
        self._cache = GrpcClientCache(self._stub)
        self._host = host
        self._port = port

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_stub"]
        del state["_cache"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._stub = api_pb2_grpc.StorageServiceStub(
            grpc.insecure_channel(f"{self._host}:{self._port}")
        )  # type: ignore
        self._cache = GrpcClientCache(self._stub)

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        request = api_pb2.CreateNewStudyRequest(
            directions=[
                api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else api_pb2.MAXIMIZE
                for d in directions
            ],
            study_name=study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4()),
        )
        try:
            response = self._stub.CreateNewStudy(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                raise DuplicatedStudyError from e
            raise
        return response.study_id

    def delete_study(self, study_id: int) -> None:
        request = api_pb2.DeleteStudyRequest(study_id=study_id)
        try:
            self._stub.DeleteStudy(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        # TODO(c-bata): Fix a cache invalidation issue when using SQLite3
        # Please see https://github.com/optuna/optuna/pull/5872/files#r1893708995 for details.
        self._cache.delete_study_cache(study_id)

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetStudyUserAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudyUserAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetStudySystemAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudySystemAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

    def get_study_id_from_name(self, study_name: str) -> int:
        request = api_pb2.GetStudyIdFromNameRequest(study_name=study_name)
        try:
            response = self._stub.GetStudyIdFromName(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.study_id

    def get_study_name_from_id(self, study_id: int) -> str:
        request = api_pb2.GetStudyNameFromIdRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyNameFromId(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.study_name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        request = api_pb2.GetStudyDirectionsRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyDirections(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return [
            StudyDirection.MINIMIZE if d == api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
            for d in response.directions
        ]

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        request = api_pb2.GetStudyUserAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyUserAttributes(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return {key: json.loads(value) for key, value in response.user_attributes.items()}

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        request = api_pb2.GetStudySystemAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudySystemAttributes(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return {key: json.loads(value) for key, value in response.system_attributes.items()}

    def get_all_studies(self) -> list[FrozenStudy]:
        request = api_pb2.GetAllStudiesRequest()
        response = self._stub.GetAllStudies(request)
        return [
            FrozenStudy(
                study_id=study.study_id,
                study_name=study.study_name,
                direction=None,
                directions=[
                    StudyDirection.MINIMIZE if d == api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
                    for d in study.directions
                ],
                user_attrs={
                    key: json.loads(value) for key, value in study.user_attributes.items()
                },
                system_attrs={
                    key: json.loads(value) for key, value in study.system_attributes.items()
                },
            )
            for study in response.studies
        ]

    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        if template_trial is None:
            request = api_pb2.CreateNewTrialRequest(study_id=study_id, template_trial_is_none=True)
        else:
            request = api_pb2.CreateNewTrialRequest(
                study_id=study_id,
                template_trial=grpc_servicer._to_proto_trial(template_trial),
                template_trial_is_none=False,
            )
        try:
            response = self._stub.CreateNewTrial(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        request = api_pb2.SetTrialParameterRequest(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
        )
        try:
            self._stub.SetTrialParameter(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError from e
            else:
                raise

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        request = api_pb2.SetTrialStateValuesRequest(
            trial_id=trial_id,
            state=grpc_servicer._to_proto_trial_state(state),
            values=values,
        )
        try:
            response = self._stub.SetTrialStateValues(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

        return response.trial_updated

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        request = api_pb2.SetTrialIntermediateValueRequest(
            trial_id=trial_id, step=step, intermediate_value=intermediate_value
        )
        try:
            self._stub.SetTrialIntermediateValue(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetTrialUserAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialUserAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetTrialSystemAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialSystemAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        request = api_pb2.GetTrialIdFromStudyIdTrialNumberRequest(
            study_id=study_id, trial_number=trial_number
        )
        try:
            response = self._stub.GetTrialIdFromStudyIdTrialNumber(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.trial_id

    def get_trial(self, trial_id: int) -> FrozenTrial:
        request = api_pb2.GetTrialRequest(trial_id=trial_id)
        try:
            response = self._stub.GetTrial(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return grpc_servicer._from_proto_trial(response.trial)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        trials = self._cache.get_all_trials(study_id, states)
        return copy.deepcopy(trials) if deepcopy else trials


class GrpcClientCache:
    def __init__(self, grpc_client: api_pb2_grpc.StorageServiceStub) -> None:
        self.studies: dict[int, GrpcClientCacheEntry] = {}
        self.grpc_client = grpc_client
        self.lock = threading.Lock()

    def delete_study_cache(self, study_id: int) -> None:
        with self.lock:
            self.studies.pop(study_id, None)

    def get_all_trials(
        self, study_id: int, states: Container[TrialState] | None
    ) -> list[FrozenTrial]:
        with self.lock:
            self._read_trials_from_remote_storage(study_id)
            study = self.studies[study_id]
            trials: dict[int, FrozenTrial] | list[FrozenTrial]
            if states is not None:
                trials = {number: t for number, t in study.trials.items() if t.state in states}
            else:
                trials = study.trials
            trials = list(sorted(trials.values(), key=lambda t: t.number))
            return trials

    def _read_trials_from_remote_storage(self, study_id: int) -> None:
        if study_id not in self.studies:
            self.studies[study_id] = GrpcClientCacheEntry()
        study = self.studies[study_id]

        req = api_pb2.GetTrialsRequest(
            study_id=study_id,
            included_trial_ids=study.unfinished_trial_ids,
            trial_id_greater_than=study.last_finished_trial_id,
        )
        try:
            res = self.grpc_client.GetTrials(req)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                self.studies.pop(study_id, None)
                raise KeyError from e
            raise
        if not res.trials:
            return

        for trial_proto in res.trials:
            trial = grpc_servicer._from_proto_trial(trial_proto)
            self._add_trial_to_cache(study_id, trial)

    def _add_trial_to_cache(self, study_id: int, trial: FrozenTrial) -> None:
        study = self.studies[study_id]
        study.trials[trial.number] = trial

        if not trial.state.is_finished():
            study.unfinished_trial_ids.add(trial._trial_id)
            return

        study.last_finished_trial_id = max(study.last_finished_trial_id, trial._trial_id)
        study.unfinished_trial_ids.discard(trial._trial_id)


class GrpcClientCacheEntry:
    def __init__(self) -> None:
        self.trials: dict[int, FrozenTrial] = {}
        self.unfinished_trial_ids: set[int] = set()
        self.last_finished_trial_id: int = -1
