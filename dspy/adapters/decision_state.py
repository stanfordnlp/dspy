"""Configuration and decoding shared by decision backends."""

import copy
import itertools
import json
import math

from pydantic import JsonValue, TypeAdapter

from dspy.adapters.types.decision import Choice, Noul, Score, decision_type

_JSON_ADAPTER = TypeAdapter(JsonValue)


class DecisionState:
    """Per-operation configuration snapshot and backend-independent decoding.

    Predict owns the persisted overrides. This snapshot resolves output types
    and copies overrides over defaults without modifying the predictor or signature.
    """

    def __init__(self, signature, fields, *, system_one=False, declared_signature=None):
        if not isinstance(fields, dict):
            raise ValueError("Decision fields must be a mapping of output names to configuration.")
        self.signature = signature
        self.types = {}
        self.fields = {}
        for name, field in signature.output_fields.items():
            if not system_one and name not in fields and not any(
                isinstance(annotation, type) and issubclass(annotation, (Noul, Score, Choice))
                for annotation in (field.annotation, *field.metadata)
            ):
                continue
            kind = decision_type(field)
            if kind is None:
                if system_one:
                    raise ValueError(f"Unsupported System One output {name!r}; use a decision type or native equivalent.")
                continue
            if declared_signature is not None and name in fields:
                declared = declared_signature.output_fields.get(name)
                original = decision_type(declared) if declared is not None else None
                if original is None or kind.options != original.options or (
                    kind.model_fields["value"].annotation != original.model_fields["value"].annotation
                ):
                    raise ValueError(f"Signature override must preserve the answer space of configured output {name!r}.")
            self.types[name] = kind
            if issubclass(kind, Noul):
                self.fields[name] = {"threshold": 0.5}
            elif issubclass(kind, Score):
                self.fields[name] = {"cuts": [i + 0.5 for i in range(len(kind.options) - 1)]}
            else:
                self.fields[name] = {"weights": {str(v): 1.0 for v, _ in kind.options}}
        if fields.keys() - self.types.keys():
            raise ValueError(f"Decision configuration refers to unsupported outputs: {sorted(fields.keys() - self.types.keys())}.")
        for name, config in fields.items():
            if not isinstance(config, dict):
                raise ValueError(f"Decision configuration for {name!r} must be a mapping.")
            self.fields[name].update(copy.deepcopy(config))
        self._validate_parameters()

    def get_criteria(self, field: str):
        """Return an independent copy of the effective criteria for an output field.

        Explicit module overrides take precedence over type-declared defaults.
        Bare Noul/bool has no default criteria and returns None.
        """
        kind = self.types[field]
        return self._question(field, self.signature.output_fields[field], kind).get("criteria")

    def set_criteria(self, field: str, criteria):
        """Validate and copy a criteria override without changing the signature.

        Noul accepts a true/false map or None; Choice requires its option-label
        map; Score requires an array matching its declared levels. Descriptions
        may be strings, objects, arrays, or None. None explicitly sends null Noul
        criteria; it does not restore the type defaults. Unknown fields raise
        KeyError, invalid criteria raise ValueError, and neither changes state.
        """
        kind = self.types[field]
        self._validate_question_config(field, kind, {"criteria": criteria})
        self.fields[field]["criteria"] = copy.deepcopy(criteria)

    def _validate_parameters(self):
        for name, kind in self.types.items():
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
        if "instructions" not in config and not (desc and desc.strip()):
            raise ValueError(f"Decision output {name!r} requires an OutputField(desc=...) or explicit instructions.")
        question = {"instructions": copy.deepcopy(config.get("instructions", desc))}
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

    def _decode(self, answers):
        outputs = {}
        for name, kind in self.types.items():
            field = self.signature.output_fields[name]
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
