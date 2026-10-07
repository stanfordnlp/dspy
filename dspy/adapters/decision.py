"""Shared decision evidence and request translation for Predict backends."""

import copy
import json
import threading
import warnings
from contextlib import contextmanager
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, create_model

from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.decision import Choice, Noul, Probability, Score
from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example


@contextmanager
def record_evidence():
    """Collect decision evidence in this execution context and its DSPy workers.

    Yields a list that receives one ``(predictor, output name, evidence)`` entry per decoded
    output. ``predictor`` is the innermost calling module. Nested collectors are independent.
    """
    log = []
    with settings.context(_decision_evidence=(log, threading.Lock())):
        yield log


@contextmanager
def replay_answers():
    """Reuse raw answers to repeated decision requests in this execution context and its DSPy workers.

    A request repeats when the client and its defaults, adapter settings, signature, question
    settings, demos, inputs, and LM arguments all match an earlier call. Its stored answers are
    decoded under the current thresholds, cuts, and weights, which change how answers are read
    but not what is asked.
    Only clients that cache take part, since an uncached client would answer a repeat afresh.
    Custom chat adapters and unsupported request values use the normal request path.
    Replayed calls skip the client, so they add no history or usage. Concurrent first calls
    may each reach the client before an answer is stored.
    """
    with settings.context(_decision_replay=({}, threading.Lock())):
        yield


def _frozen(value):
    """Snapshot supported request values without merging different types or dict orders.

    Hashability alone is insufficient: arbitrary objects may change their rendered value
    while keeping the same hash. Unsupported values must use the normal request path.
    """
    kind = type(value)
    if kind is dict:
        return (kind, tuple((_frozen(k), _frozen(v)) for k, v in value.items()))
    if kind in (list, tuple):
        return (kind, tuple(_frozen(v) for v in value))
    if kind is float:
        return (kind, value.hex())  # Preserve signed zero, which compares equal in Python.
    if kind in (str, int, bool, type(None)):
        return (kind, value)
    if isinstance(value, BaseModel):
        return (kind, _frozen(value.model_dump(mode="json")))
    if isinstance(value, Example):
        return (kind, _frozen(value.toDict()))
    raise TypeError(f"Cannot snapshot {kind.__name__} for decision replay.")


@lru_cache(maxsize=256)
def evidence_type(kind):
    """Closed schemas work with JSON structured outputs as well as chat adapters."""
    name = "Noul" if issubclass(kind, Noul) else "Score" if issubclass(kind, Score) else "Choice"
    if issubclass(kind, Noul):
        fields = {"noul": (Probability, Field(description="Probability that the answer is true."))}
    else:
        criteria = kind.criteria()
        labels = range(len(criteria)) if issubclass(kind, Score) else criteria
        probabilities = create_model(
            f"{name}Probabilities",
            __config__=ConfigDict(extra="forbid"),
            **{f"option_{i}": (Probability, Field(alias=str(label))) for i, label in enumerate(labels)},
        )
        fields = {
            "probabilities": (probabilities, Field(description="Probability of each option; sum to one.")),
            "confidence": (Probability, Field(description="Confidence in the decision, from 0 to 1.")),
        }
    return create_model(f"{name}Evidence", __config__=ConfigDict(extra="forbid"), **fields)


def resolve_adapter(lm, adapter, signature, fields, declared_signature=None):
    """Resolve backend translation before a chat adapter starts capability planning."""
    system_one = getattr(lm, "supports_decision_requests", False) is True
    state = DecisionState(signature, fields, system_one=system_one, declared_signature=declared_signature)
    for name, field in signature.output_fields.items():
        if name not in state.types and any(
            kind.extract_custom_type_from_annotation(field.rebuild_annotation()) for kind in (Noul, Choice, Score)
        ):
            warnings.warn(
                f"Decision evidence decoding is not implemented for nested output {name!r}. "
                "The LM generates values and confidence directly; thresholds, cuts, and weights are not applied. "
                "Use top-level decision output fields instead.",
                UserWarning,
                stacklevel=2,
            )
    return DecisionAdapter(adapter, state, system_one) if state.types else adapter


class DecisionAdapter:
    """Translate decision outputs once, independently of the chosen chat format."""

    def __init__(self, adapter, state, system_one):
        self.adapter = adapter
        self.state = state
        self.system_one = system_one

    def _prepare(self, signature, demos, inputs, lm_kwargs):
        from dspy.adapters.utils import get_field_description_string
        from dspy.predict.predict import serialize_object

        if settings.send_stream is not None:
            raise NotImplementedError("Streaming decision evidence is not supported.")
        types = self.state.types
        questions = {
            name: self.state._question(name, signature.output_fields[name], kind) for name, kind in types.items()
        }
        if self.system_one:
            if lm_kwargs:
                raise ValueError(f"Unsupported TypeSafe generation settings: {sorted(lm_kwargs)}.")
            state = {
                "instructions": signature.instructions,
                "input_fields": get_field_description_string(signature.input_fields),
                "inputs": serialize_object({k: v for k, v in inputs.items() if k in signature.input_fields}),
            }
            if demos:
                state["demos"] = [
                    serialize_object({k: v for k, v in demo.items() if k in signature.fields}) for demo in demos
                ]
            return {"state": state, "questions": questions}
        for name, kind in types.items():
            signature = signature.with_updated_fields(
                name, type_=evidence_type(kind), desc="\n" + json.dumps(questions[name], ensure_ascii=False, indent=2)
            )
        # Labeled demonstrations need not contain distributions. Keep them as
        # task examples in the instructions, rather than fabricating evidence.
        if demos:
            examples = [serialize_object({k: v for k, v in demo.items() if k in signature.fields}) for demo in demos]
            signature = signature.with_instructions(
                signature.instructions + "\n\nTask examples (labels or evidence):\n" + json.dumps(examples)
            )
        return {"lm_kwargs": lm_kwargs, "signature": signature, "demos": [], "inputs": inputs}

    def _decode(self, completions):
        types = self.state.types
        results = []
        for completion in completions:
            answers = {}
            for name, kind in types.items():
                answer = completion[name]
                if isinstance(answer, BaseModel):
                    answer = answer.model_dump(by_alias=True)
                answer = copy.deepcopy(answer)
                if issubclass(kind, Score):
                    keys = answer["probabilities"]
                    labels = {str(i) for i in range(len(kind.criteria()))}
                    if any(type(k) not in (int, str) or str(k) not in labels for k in keys) or len(keys) != len(labels):
                        raise ValueError(f"Invalid Score distribution for {name!r}.")
                    answer["probabilities"] = {int(k): v for k, v in answer["probabilities"].items()}
                answers[name] = answer
            recorder = settings.get("_decision_evidence")
            if recorder is not None:
                log, lock = recorder
                caller = (settings.caller_modules or [None])[-1]
                with lock:
                    log.extend((caller, name, copy.deepcopy(answer)) for name, answer in answers.items())
            results.append({**completion, **self.state._decode(answers)})
        return results

    def _replay_key(self, lm, lm_kwargs, signature, demos, inputs):
        """This request's key in the open replay store, or None when it cannot be replayed."""
        if (
            settings.get("_decision_replay") is None
            or not lm_kwargs.get("cache", getattr(lm, "cache", False))
            or not getattr(lm, "_cache_responses", True)
        ):
            return None
        questions = {
            name: {k: v for k, v in config.items() if k in ("instructions", "criteria")}
            for name, config in self.state.fields.items()
        }
        try:
            adapter = None
            if not self.system_one:
                from dspy.adapters.chat_adapter import ChatAdapter
                from dspy.adapters.json_adapter import JSONAdapter

                # Custom adapters can depend on state outside their attributes. Leave
                # their request construction and cache lookup to the normal path.
                if type(self.adapter) not in (ChatAdapter, JSONAdapter):
                    return None
                adapter = (
                    type(self.adapter),
                    tuple(self.adapter.native_response_types),
                    _frozen({k: v for k, v in vars(self.adapter).items() if k != "native_response_types"}),
                )
            client = {
                name: getattr(lm, name, None)
                for name in ("model", "model_type", "kwargs", "base_url", "use_developer_role")
            }
            key = (
                id(lm),
                self.system_one,
                _frozen(client),
                adapter,
                signature,
                _frozen(signature.dump_state()),
                _frozen(questions),
                _frozen(demos),
                _frozen(inputs),
                _frozen(lm_kwargs),
            )
            hash(key)
        except (TypeError, ValueError, RecursionError):
            return None
        return key

    @staticmethod
    def _store(key, lm, completions):
        answers, lock = settings.get("_decision_replay")
        stored = copy.deepcopy(completions)
        with lock:
            # Keep runtime-selected clients alive so their ids cannot be reused in this context.
            answers[key] = (lm, stored)

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        key = self._replay_key(lm, lm_kwargs, signature, demos, inputs)
        if key is not None and key in settings.get("_decision_replay")[0]:
            return self._decode(copy.deepcopy(settings.get("_decision_replay")[0][key][1]))
        request = self._prepare(signature, demos, inputs, lm_kwargs)
        completions = [lm(**request)] if self.system_one else self.adapter(lm, **request)
        if key is not None:
            self._store(key, lm, completions)
        return self._decode(completions)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        key = self._replay_key(lm, lm_kwargs, signature, demos, inputs)
        if key is not None and key in settings.get("_decision_replay")[0]:
            return self._decode(copy.deepcopy(settings.get("_decision_replay")[0][key][1]))
        request = self._prepare(signature, demos, inputs, lm_kwargs)
        completions = [await lm.acall(**request)] if self.system_one else await self.adapter.acall(lm, **request)
        if key is not None:
            self._store(key, lm, completions)
        return self._decode(completions)
