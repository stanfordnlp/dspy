from __future__ import annotations

from datetime import datetime
import json
import threading
from typing import TYPE_CHECKING

from optuna import logging
from optuna._imports import _LazyImport
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import BaseStorage
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if TYPE_CHECKING:
    import grpc

    from optuna.storages._grpc.auto_generated import api_pb2
    from optuna.storages._grpc.auto_generated import api_pb2_grpc
else:
    api_pb2 = _LazyImport("optuna.storages._grpc.auto_generated.api_pb2")
    api_pb2_grpc = _LazyImport("optuna.storages._grpc.auto_generated.api_pb2_grpc")
    grpc = _LazyImport("grpc")


_logger = logging.get_logger(__name__)
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class OptunaStorageProxyService(api_pb2_grpc.StorageServiceServicer):
    def __init__(self, storage: BaseStorage) -> None:
        self._backend = storage
        self._lock = threading.Lock()

    def CreateNewStudy(
        self,
        request: api_pb2.CreateNewStudyRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.CreateNewStudyReply:
        directions = [
            StudyDirection.MINIMIZE if d == api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
            for d in request.directions
        ]
        study_name = request.study_name

        try:
            study_id = self._backend.create_new_study(directions=directions, study_name=study_name)
        except DuplicatedStudyError as e:
            context.abort(code=grpc.StatusCode.ALREADY_EXISTS, details=str(e))
        return api_pb2.CreateNewStudyReply(study_id=study_id)

    def DeleteStudy(
        self,
        request: api_pb2.DeleteStudyRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.DeleteStudyReply:
        study_id = request.study_id
        try:
            self._backend.delete_study(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.DeleteStudyReply()

    def SetStudyUserAttribute(
        self,
        request: api_pb2.SetStudyUserAttributeRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetStudyUserAttributeReply:
        try:
            self._backend.set_study_user_attr(
                request.study_id, request.key, json.loads(request.value)
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.SetStudyUserAttributeReply()

    def SetStudySystemAttribute(
        self,
        request: api_pb2.SetStudySystemAttributeRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetStudySystemAttributeReply:
        try:
            self._backend.set_study_system_attr(
                request.study_id, request.key, json.loads(request.value)
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.SetStudySystemAttributeReply()

    def GetStudyIdFromName(
        self,
        request: api_pb2.GetStudyIdFromNameRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetStudyIdFromNameReply:
        try:
            study_id = self._backend.get_study_id_from_name(request.study_name)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.GetStudyIdFromNameReply(study_id=study_id)

    def GetStudyNameFromId(
        self,
        request: api_pb2.GetStudyNameFromIdRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetStudyNameFromIdReply:
        study_id = request.study_id

        try:
            name = self._backend.get_study_name_from_id(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        assert name is not None
        return api_pb2.GetStudyNameFromIdReply(study_name=name)

    def GetStudyDirections(
        self,
        request: api_pb2.GetStudyDirectionsRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetStudyDirectionsReply:
        study_id = request.study_id

        try:
            directions = self._backend.get_study_directions(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        assert directions is not None
        return api_pb2.GetStudyDirectionsReply(
            directions=[
                api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else api_pb2.MAXIMIZE
                for d in directions
            ]
        )

    def GetStudyUserAttributes(
        self,
        request: api_pb2.GetStudyUserAttributesRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetStudyUserAttributesReply:
        try:
            attributes = self._backend.get_study_user_attrs(request.study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.GetStudyUserAttributesReply(
            user_attributes={key: json.dumps(value) for key, value in attributes.items()}
        )

    def GetStudySystemAttributes(
        self,
        request: api_pb2.GetStudySystemAttributesRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetStudySystemAttributesReply:
        try:
            attributes = self._backend.get_study_system_attrs(request.study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.GetStudySystemAttributesReply(
            system_attributes={key: json.dumps(value) for key, value in attributes.items()}
        )

    def GetAllStudies(
        self,
        request: api_pb2.GetAllStudiesRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetAllStudiesReply:
        studies = self._backend.get_all_studies()
        return api_pb2.GetAllStudiesReply(
            studies=[
                api_pb2.Study(
                    study_id=study._study_id,
                    study_name=study.study_name,
                    directions=[
                        api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else api_pb2.MAXIMIZE
                        for d in study.directions
                    ],
                    user_attributes={
                        key: json.dumps(value) for key, value in study.user_attrs.items()
                    },
                    system_attributes={
                        key: json.dumps(value) for key, value in study.system_attrs.items()
                    },
                )
                for study in studies
            ]
        )

    def CreateNewTrial(
        self,
        request: api_pb2.CreateNewTrialRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.CreateNewTrialReply:
        study_id = request.study_id

        template_trial = None
        if not request.template_trial_is_none:
            template_trial = _from_proto_trial(request.template_trial)

        try:
            trial_id = self._backend.create_new_trial(study_id, template_trial)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        return api_pb2.CreateNewTrialReply(trial_id=trial_id)

    def SetTrialParameter(
        self,
        request: api_pb2.SetTrialParameterRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetTrialParameterReply:
        trial_id = request.trial_id
        param_name = request.param_name
        param_value_internal = request.param_value_internal
        distribution = json_to_distribution(request.distribution)
        try:
            self._backend.set_trial_param(trial_id, param_name, param_value_internal, distribution)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        except ValueError as e:
            context.abort(code=grpc.StatusCode.INVALID_ARGUMENT, details=str(e))
        return api_pb2.SetTrialParameterReply()

    def GetTrialIdFromStudyIdTrialNumber(
        self,
        request: api_pb2.GetTrialIdFromStudyIdTrialNumberRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetTrialIdFromStudyIdTrialNumberReply:
        study_id = request.study_id
        trial_number = request.trial_number

        try:
            trial_id = self._backend.get_trial_id_from_study_id_trial_number(
                study_id, trial_number
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return api_pb2.GetTrialIdFromStudyIdTrialNumberReply(trial_id=trial_id)

    def SetTrialStateValues(
        self,
        request: api_pb2.SetTrialStateValuesRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetTrialStateValuesReply:
        trial_id = request.trial_id
        state = request.state
        values = request.values
        try:
            trial_updated = self._backend.set_trial_state_values(
                trial_id, _from_proto_trial_state(state), values
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return api_pb2.SetTrialStateValuesReply(trial_updated=trial_updated)

    def SetTrialIntermediateValue(
        self,
        request: api_pb2.SetTrialIntermediateValueRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetTrialIntermediateValueReply:
        trial_id = request.trial_id
        step = request.step
        intermediate_value = request.intermediate_value
        try:
            self._backend.set_trial_intermediate_value(trial_id, step, intermediate_value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return api_pb2.SetTrialIntermediateValueReply()

    def SetTrialUserAttribute(
        self,
        request: api_pb2.SetTrialUserAttributeRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetTrialUserAttributeReply:
        trial_id = request.trial_id
        key = request.key
        value = json.loads(request.value)
        try:
            self._backend.set_trial_user_attr(trial_id, key, value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return api_pb2.SetTrialUserAttributeReply()

    def SetTrialSystemAttribute(
        self,
        request: api_pb2.SetTrialSystemAttributeRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.SetTrialSystemAttributeReply:
        trial_id = request.trial_id
        key = request.key
        value = json.loads(request.value)
        try:
            self._backend.set_trial_system_attr(trial_id, key, value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return api_pb2.SetTrialSystemAttributeReply()

    def GetTrial(
        self,
        request: api_pb2.GetTrialRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetTrialReply:
        trial_id = request.trial_id
        try:
            trial = self._backend.get_trial(trial_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        return api_pb2.GetTrialReply(trial=_to_proto_trial(trial))

    def GetTrials(
        self,
        request: api_pb2.GetTrialsRequest,
        context: grpc.ServicerContext,
    ) -> api_pb2.GetTrialsReply:
        study_id = request.study_id
        included_trial_ids = set(request.included_trial_ids)
        trial_id_greater_than = request.trial_id_greater_than
        try:
            trials = self._backend.get_all_trials(study_id, deepcopy=False)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        filtered_trials = [
            _to_proto_trial(t)
            for t in trials
            if t._trial_id > trial_id_greater_than or t._trial_id in included_trial_ids
        ]
        return api_pb2.GetTrialsReply(trials=filtered_trials)


def _to_proto_trial_state(state: TrialState) -> api_pb2.TrialState.ValueType:
    if state == TrialState.RUNNING:
        return api_pb2.RUNNING
    if state == TrialState.COMPLETE:
        return api_pb2.COMPLETE
    if state == TrialState.PRUNED:
        return api_pb2.PRUNED
    if state == TrialState.FAIL:
        return api_pb2.FAIL
    if state == TrialState.WAITING:
        return api_pb2.WAITING
    raise ValueError(f"Unknown TrialState: {state}")


def _from_proto_trial_state(state: api_pb2.TrialState.ValueType) -> TrialState:
    if state == api_pb2.RUNNING:
        return TrialState.RUNNING
    if state == api_pb2.COMPLETE:
        return TrialState.COMPLETE
    if state == api_pb2.PRUNED:
        return TrialState.PRUNED
    if state == api_pb2.FAIL:
        return TrialState.FAIL
    if state == api_pb2.WAITING:
        return TrialState.WAITING
    raise ValueError(f"Unknown api_pb2.TrialState: {state}")


def _to_proto_trial(trial: FrozenTrial) -> api_pb2.Trial:
    params = {}
    for key, value in trial.params.items():
        params[key] = trial.distributions[key].to_internal_repr(value)

    return api_pb2.Trial(
        trial_id=trial._trial_id,
        number=trial.number,
        state=_to_proto_trial_state(trial.state),
        values=trial.values,
        datetime_start=(
            trial.datetime_start.strftime(DATETIME_FORMAT) if trial.datetime_start else ""
        ),
        datetime_complete=(
            trial.datetime_complete.strftime(DATETIME_FORMAT) if trial.datetime_complete else ""
        ),
        distributions={
            key: distribution_to_json(distribution)
            for key, distribution in trial.distributions.items()
        },
        params=params,
        user_attributes={key: json.dumps(value) for key, value in trial.user_attrs.items()},
        system_attributes={key: json.dumps(value) for key, value in trial.system_attrs.items()},
        intermediate_values={step: value for step, value in trial.intermediate_values.items()},
    )


def _from_proto_trial(trial: api_pb2.Trial) -> FrozenTrial:
    datetime_start = (
        datetime.strptime(trial.datetime_start, DATETIME_FORMAT) if trial.datetime_start else None
    )
    datetime_complete = (
        datetime.strptime(trial.datetime_complete, DATETIME_FORMAT)
        if trial.datetime_complete
        else None
    )
    distributions = {
        key: json_to_distribution(value) for key, value in trial.distributions.items()
    }
    params = {}
    for key, value in trial.params.items():
        params[key] = distributions[key].to_external_repr(value)

    return FrozenTrial(
        trial_id=trial.trial_id,
        number=trial.number,
        state=_from_proto_trial_state(trial.state),
        value=None,
        values=trial.values if trial.values else None,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=params,
        distributions=distributions,
        user_attrs={key: json.loads(value) for key, value in trial.user_attributes.items()},
        system_attrs={key: json.loads(value) for key, value in trial.system_attributes.items()},
        intermediate_values={step: value for step, value in trial.intermediate_values.items()},
    )
