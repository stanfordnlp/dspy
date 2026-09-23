"""A closed-set predictor backed by a System One model."""

import copy
import itertools
import json
import math

from pydantic import JsonValue, TypeAdapter

from dspy.adapters.base import Adapter
from dspy.adapters.types.decision import Choice, Noul, Score, decision_type
from dspy.adapters.utils import get_field_description_string
from dspy.clients.base_lm import sanitize_lm_state
from dspy.clients.typesafe import TypeSafe
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import serialize_object
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental

_JSON_ADAPTER = TypeAdapter(JsonValue)


@experimental
class DecisionAdapter(Adapter):
    """Answer a signature's closed-set outputs in one System One request.

    Declare outputs using Noul, Score["low", "medium", "high"], and
    Choice[(value, description), ...]. Native bool, Literal,
    Annotated[bool, Noul[...]], and Annotated[float, Score[...]] return only the
    value. Bare float has no rubric.
    Unsupported outputs raise; Decide never falls back to a generative LM.

    Each instance owns ``fields[name]`` configuration: a Boolean ``threshold``
    (initially 0.5), or ``cuts`` for Score level selection, halfway between level
    indices. Score values average those indices; cuts affect only .level.
    Choice weights are nonnegative probability multipliers keyed by string
    option labels, initially 1.0. Missing option weights also default to 1.0.
    Weighted Choice selection preserves raw probabilities and provider confidence;
    that confidence does not describe a newly selected option. Changing parameters
    reuses cached provider distributions without mutating the signature types.

    Use ``Predict(signature, adapter=DecisionAdapter(), backend=TypeSafe(...))``.
    Predict binds an independent adapter to each signature. The backend defaults
    to ``settings.system_one``, never ``settings.lm``. Module callbacks, tracing,
    and Prediction construction remain Predict's responsibility.
    """

    def __init__(self, *, fields=None, **kwargs):
        super().__init__(**kwargs)
        self.fields = copy.deepcopy(fields)

    def dump_state(self):
        state = super().dump_state()
        state["fields"] = copy.deepcopy(self.fields)
        return state

    def bind(self, signature):
        """Return independent per-predictor configuration for this signature."""
        bound = copy.copy(self)
        types = self._output_types(signature)
        if self.fields is not None:
            bound.fields = copy.deepcopy(self.fields)
            bound._validate_parameters(types)
            return bound
        bound.fields = {}
        for name, kind in types.items():
            if issubclass(kind, Noul):
                bound.fields[name] = {"threshold": 0.5}
            elif issubclass(kind, Score):
                bound.fields[name] = {"cuts": [i + 0.5 for i in range(len(kind.options) - 1)]}
            else:
                bound.fields[name] = {"weights": {str(v): 1.0 for v, _ in kind.options}}
        return bound

    def get_criteria(self, signature, field: str):
        """Return an independent copy of the effective criteria for an output field.

        Explicit module overrides take precedence over type-declared defaults.
        Bare Noul/bool has no default criteria and returns None.
        """
        kind = self._output_types(signature)[field]
        return self._question(field, signature.output_fields[field], kind).get("criteria")

    def set_criteria(self, signature, field: str, criteria):
        """Validate and copy a criteria override without changing the signature.

        Noul accepts a true/false map or None; Choice requires its option-label
        map; Score requires an array matching its declared levels. Descriptions
        may be strings, objects, arrays, or None. None explicitly sends null Noul
        criteria; it does not restore the type defaults. Unknown fields raise
        KeyError, invalid criteria raise ValueError, and neither changes state.
        """
        kind = self._output_types(signature)[field]
        self._validate_question_config(field, kind, {"criteria": criteria})
        self.fields[field]["criteria"] = copy.deepcopy(criteria)

    @staticmethod
    def _output_types(signature):
        if not signature.output_fields:
            raise ValueError("Decide requires at least one output field.")
        types = {}
        for name, field in signature.output_fields.items():
            kind = decision_type(field)
            if kind is None:
                raise ValueError(
                    f"Unsupported Decide output {name!r}. Use Noul, Score[...], Choice[...], bool, Literal[...], "
                    "Annotated[bool, Noul[...]], or Annotated[float, Score[...]]. Bare float requires a rubric."
                )
            types[name] = kind
        return types

    def _validate_parameters(self, types):
        if not isinstance(self.fields, dict) or self.fields.keys() != types.keys():
            raise ValueError("Decide fields must match its declared output fields.")
        for name, kind in types.items():
            config = self.fields[name]
            parameter = "threshold" if issubclass(kind, Noul) else "cuts" if issubclass(kind, Score) else "weights"
            if (
                not isinstance(config, dict)
                or parameter not in config
                or not config.keys() <= {"instructions", "criteria", parameter}
            ):
                raise ValueError(
                    f"Invalid configuration for {name!r}: require {parameter!r}, with optional instructions and criteria."
                )
            self._validate_question_config(name, kind, config)
            if issubclass(kind, Noul):
                threshold = config["threshold"]
                if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                    raise ValueError(f"Threshold for {name!r} must be in [0, 1].")
            elif issubclass(kind, Score):
                cuts = config["cuts"]
                if (
                    not 2 <= len(kind.options) <= 10
                    or not isinstance(cuts, list)
                    or len(cuts) != len(kind.options) - 1
                    or any(type(c) not in (int, float) or not 0 < c < len(kind.options) - 1 for c in cuts)
                    or any(a >= b for a, b in itertools.pairwise(cuts))
                ):
                    raise ValueError(
                        f"Invalid cuts for {name!r}: require ordered boundaries inside the Score index range."
                    )
            else:
                weights = config["weights"]
                labels = {str(v) for v, _ in kind.options}
                if (
                    not isinstance(weights, dict)
                    or not set(weights) <= labels
                    or any(type(w) not in (int, float) or not 0 <= w < math.inf for w in weights.values())
                    or not any(weights.get(label, 1.0) > 0 for label in labels)
                ):
                    raise ValueError(
                        f"Choice weights for {name!r} must map known string labels to finite, nonnegative numbers "
                        "with at least one positive effective weight."
                    )

    @staticmethod
    def _validate_question_config(name, kind, config):
        for label in ("instructions", "criteria"):
            if label in config:
                entry = config[label]
                if not isinstance(entry, (str, dict, list, type(None))):
                    raise ValueError(f"Invalid {label} for {name!r}: expected a string, object, array, or null.")
                try:
                    _JSON_ADAPTER.validate_python(entry, strict=True)
                    json.dumps(entry, allow_nan=False)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid JSON in {label} for {name!r}.") from error
        if "criteria" in config:
            criteria = config["criteria"]
            if issubclass(kind, Noul):
                valid = criteria is None or (isinstance(criteria, dict) and criteria.keys() <= {"true", "false"})
            elif issubclass(kind, Choice):
                valid = isinstance(criteria, dict) and criteria.keys() == {str(v) for v, _ in kind.options}
            else:
                valid = isinstance(criteria, list) and len(criteria) == len(kind.options) and 2 <= len(criteria) <= 10
            if not valid:
                raise ValueError(f"Invalid criteria for {name!r}: must match the declared decision type and options.")
            entries = criteria.values() if isinstance(criteria, dict) else criteria or []
            if any(not isinstance(entry, (str, dict, list, type(None))) for entry in entries):
                raise ValueError(f"Invalid criteria description for {name!r}.")

    def _question(self, name, field, kind):
        config = self.fields[name]
        desc = field.json_schema_extra.get("desc", "")
        if desc == f"${{{name}}}":
            desc = ""
        question = {"instructions": copy.deepcopy(config.get("instructions", desc or f"Decide `{name}`."))}
        if issubclass(kind, Noul):
            question["type"] = "noul"
            if kind.options:
                question["criteria"] = {str(v).lower(): desc for v, desc in kind.options}
        elif issubclass(kind, Score):
            question.update(type="score", criteria=list(kind.options))
        else:
            question.update(type="choice", criteria={str(v): desc or None for v, desc in kind.options})
        if "criteria" in config:
            question["criteria"] = copy.deepcopy(config["criteria"])
        return question

    def prepare_call(self, signature, backend, config, demos, kwargs):
        """Resolve the backend and validate inputs without generative LM assumptions."""
        if config:
            raise ValueError("DecisionAdapter does not accept generative LM configuration.")
        trace = kwargs.pop("_trace", True)
        declared_signature = signature
        signature = ensure_signature(kwargs.pop("signature", signature))
        client = backend if backend is not None else settings.system_one
        if client is None:
            raise ValueError(
                "Configure a System One client with dspy.configure(system_one=dspy.experimental.TypeSafe(...))."
            )
        types = self._output_types(signature)
        if signature is not declared_signature:
            declared = self._output_types(declared_signature)
            if types.keys() != declared.keys() or any(
                kind.model_fields["value"].annotation != declared[name].model_fields["value"].annotation
                or kind.options != declared[name].options
                for name, kind in types.items()
            ):
                raise ValueError(
                    "Decide signature override must preserve output names, value types, and declared options/rubrics. "
                    "Construct a new Decide for a different answer space."
                )
        self._validate_parameters(types)
        inputs = {}
        for name, field in signature.input_fields.items():
            if name in kwargs:
                value = kwargs.pop(name)
            elif not field.is_required():
                value = field.get_default(call_default_factory=True)
            else:
                raise ValueError(f"Missing Decide input: {name!r}.")
            inputs[name] = TypeAdapter(field.rebuild_annotation()).validate_python(value)
        if kwargs:
            raise ValueError(f"Unexpected Decide inputs: {sorted(kwargs)}.")
        return client, {}, signature, demos, {**inputs, "_trace": trace}

    def format(self, signature, demos, inputs):
        types = self._output_types(signature)
        questions = {name: self._question(name, field, types[name]) for name, field in signature.output_fields.items()}
        state = {
            "instructions": signature.instructions,
            "input_fields": get_field_description_string(signature.input_fields),
            "inputs": serialize_object({name: inputs[name] for name in signature.input_fields}),
        }
        if demos:
            state["demos"] = [
                serialize_object({name: demo[name] for name in signature.fields if name in demo})
                for demo in demos
            ]
        return state, questions

    def parse(self, signature, answers):
        types = self._output_types(signature)
        outputs = {}
        for name, field in signature.output_fields.items():
            kind = types[name]
            answer = answers[name]
            if issubclass(kind, Noul):
                probability = answer["noul"]
                threshold = self.fields[name]["threshold"]
                result = kind(
                    value=probability >= threshold,
                    probability=probability,
                    confidence=abs(probability - threshold) / max(threshold, 1 - threshold),
                )
            elif issubclass(kind, Score):
                probabilities = answer["probabilities"]
                if set(probabilities) != set(range(len(kind.options))) or sum(probabilities.values()) <= 0:
                    raise ValueError(f"Invalid Score distribution for {name!r}.")
                value = sum(i * p for i, p in probabilities.items()) / sum(probabilities.values())
                result = kind(
                    value=value,
                    confidence=answer["confidence"],
                    probabilities=probabilities,
                    level=sum(value >= cut for cut in self.fields[name]["cuts"]),
                )
            else:
                options = {str(v): v for v, _ in kind.options}
                if (
                    answer["choice"] not in options
                    or set(answer["probabilities"]) != set(options)
                    or sum(answer["probabilities"].values()) <= 0
                ):
                    raise ValueError(f"Invalid Choice answer for {name!r}.")
                selected = answer["choice"]
                weights = self.fields[name]["weights"]
                if any(w != 1 for w in weights.values()):
                    scores = {label: answer["probabilities"][label] * weights.get(label, 1.0) for label in options}
                    # Preserve the provider choice on ties; otherwise use declaration order.
                    selected = max(scores, key=lambda label: (scores[label], label == answer["choice"]))
                    if scores[selected] <= 0:
                        raise ValueError(f"Choice weights for {name!r} leave no positive probability mass.")
                result = kind(
                    value=options[selected],
                    confidence=answer["confidence"],
                    probabilities=answer["probabilities"],
                )
            outputs[name] = result if field.annotation is kind else result.value
        return outputs

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        state, questions = self.format(signature, demos, inputs)
        return [self.parse(signature, lm(state=state, questions=questions))]

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        state, questions = self.format(signature, demos, inputs)
        return [self.parse(signature, await lm.acall(state=state, questions=questions))]

    def dump_predict_state(self, signature, backend, json_mode=True):
        self._validate_parameters(self._output_types(signature))
        if backend is not None and not isinstance(backend, TypeSafe):
            raise TypeError(
                "Saving an explicit Decide client requires dspy.experimental.TypeSafe; configure custom clients in settings."
            )
        return {
            "client": backend.dump_state() if backend is not None else None,
        }

    def load_predict_state(self, signature, state, *, allow_unsafe_lm_state=False):
        state = copy.deepcopy(state)
        client_state = state.pop("client", None)
        restored = copy.copy(self)
        signature = signature.load_state(state["signature"])
        if "adapter" not in state:
            restored.fields = state["fields"]
        backend = TypeSafe(**sanitize_lm_state(client_state, allow_unsafe_lm_state)) if client_state else None
        restored._validate_parameters(self._output_types(signature))
        self.__dict__.update(restored.__dict__)
        return signature, backend
