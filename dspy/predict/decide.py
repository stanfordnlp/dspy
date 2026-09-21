"""A closed-set predictor backed by a System One model."""

import copy
import itertools
import math

from pydantic import TypeAdapter

from dspy.adapters.types.decision import Choice, Noul, Score, decision_type
from dspy.adapters.utils import get_field_description_string
from dspy.clients.typesafe import TypeSafe
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.predict.predict import _sanitize_lm_state, serialize_object
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.callback import BaseCallback


@experimental
class Decide(Module, Parameter):
    """Answer a signature's closed-set outputs in one System One request.

    Declare outputs using Noul, Score[(number, description), ...], and
    Choice[(value, description), ...]. Native bool, Literal, and
    Annotated[float, Score[...]] return only the value. Bare float has no rubric.
    Unsupported outputs raise; Decide never falls back to a generative LM.

    Each instance owns ``thresholds`` (one per Boolean output, initially 0.5)
    and ``weights``. Score weights are numeric option values, initially the
    declared anchors, and must increase strictly within the declared range.
    Choice weights are nonnegative probability multipliers keyed by string
    option labels, initially 1.0. Missing option weights also default to 1.0.
    Weighted Choice selection preserves raw probabilities and provider confidence;
    that confidence does not describe a newly selected option. Changing parameters
    reuses cached provider distributions without mutating the signature types.

    Args:
        signature: A DSPy signature with supported output annotations.
        client: A dspy.experimental.TypeSafe client; defaults to settings.system_one.
        callbacks: DSPy module callbacks.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        *,
        client: TypeSafe | None = None,
        callbacks: list[BaseCallback] | None = None,
    ):
        super().__init__(callbacks=callbacks)
        self.signature = ensure_signature(signature)
        self.client = client
        types = self._output_types(self.signature)
        self.thresholds = {name: 0.5 for name, kind in types.items() if kind is Noul}
        self.weights = {name: [v for v, _ in kind.options] for name, kind in types.items() if issubclass(kind, Score)}
        self.weights.update(
            {name: {str(v): 1.0 for v, _ in kind.options} for name, kind in types.items() if issubclass(kind, Choice)}
        )

    def reset(self):
        """Keep configuration intact: Decide has no demonstration or training state to reset."""

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
                    "or Annotated[float, Score[...]]. Bare float requires a rubric."
                )
            types[name] = kind
        return types

    def _validate_parameters(self, types):
        if set(self.thresholds) != {name for name, kind in types.items() if kind is Noul}:
            raise ValueError("Decide thresholds must match its Boolean output fields.")
        for name, threshold in self.thresholds.items():
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                raise ValueError(f"Threshold for {name!r} must be in [0, 1].")
        if set(self.weights) != types.keys() - self.thresholds.keys():
            raise ValueError("Decide weights must match its Score and Choice output fields.")
        for name, weights in self.weights.items():
            options = types[name].options
            if issubclass(types[name], Choice):
                labels = {str(v) for v, _ in options}
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
                continue
            if (
                len(weights) != len(options)
                or any(type(w) not in (int, float) or not options[0][0] <= w <= options[-1][0] for w in weights)
                or any(a >= b for a, b in itertools.pairwise(weights))
            ):
                raise ValueError(f"Weights for {name!r} must increase strictly within the declared Score range.")

    def _questions(self, signature, types):
        questions = {}
        for name, field in signature.output_fields.items():
            kind = types[name]
            desc = field.json_schema_extra.get("desc", "")
            if desc == f"${{{name}}}":
                desc = ""
            instructions = {
                "question": desc or f"Decide `{name}`.",
                "task": signature.instructions,
                "inputs": get_field_description_string(signature.input_fields),
            }
            question = {"instructions": instructions}
            if kind is Noul:
                question["type"] = "noul"
            elif issubclass(kind, Score):
                question.update(type="score", criteria=[desc for _, desc in kind.options])
            else:
                question.update(type="choice", criteria={str(v): desc or None for v, desc in kind.options})
            questions[name] = question
        return questions

    def _prepare(self, kwargs):
        trace = kwargs.pop("_trace", True)
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        client = self.client if self.client is not None else settings.system_one
        if client is None:
            raise ValueError(
                "Configure a System One client with dspy.configure(system_one=dspy.experimental.TypeSafe(...))."
            )
        types = self._output_types(signature)
        if signature is not self.signature:
            declared = self._output_types(self.signature)
            if types.keys() != declared.keys() or any(
                kind.model_fields["value"].annotation != declared[name].model_fields["value"].annotation
                or getattr(kind, "options", ()) != getattr(declared[name], "options", ())
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
        questions = self._questions(signature, types)
        return client, signature, types, inputs, questions, trace

    def _decode(self, answers, signature, types, inputs, trace):
        outputs = {}
        for name, field in signature.output_fields.items():
            kind = types[name]
            answer = answers[name]
            if kind is Noul:
                probability = answer["noul"]
                threshold = self.thresholds[name]
                result = Noul(
                    value=probability >= threshold,
                    probability=probability,
                    confidence=abs(probability - threshold) / max(threshold, 1 - threshold),
                )
            elif issubclass(kind, Score):
                probabilities = answer["probabilities"]
                weights = self.weights[name]
                if set(probabilities) != set(range(len(weights))) or sum(probabilities.values()) <= 0:
                    raise ValueError(f"Invalid Score distribution for {name!r}.")
                value = sum(weights[i] * p for i, p in probabilities.items()) / sum(probabilities.values())
                result = kind(value=value, confidence=answer["confidence"], probabilities=probabilities)
            else:
                options = {str(v): v for v, _ in kind.options}
                if (
                    answer["choice"] not in options
                    or set(answer["probabilities"]) != set(options)
                    or sum(answer["probabilities"].values()) <= 0
                ):
                    raise ValueError(f"Invalid Choice answer for {name!r}.")
                selected = answer["choice"]
                weights = self.weights[name]
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
        prediction = Prediction.from_completions([outputs], signature=signature)
        if trace and settings.trace is not None and settings.max_trace_size > 0:
            if len(settings.trace) >= settings.max_trace_size:
                settings.trace.pop(0)
            settings.trace.append((self, inputs, prediction))
        return prediction

    def forward(self, **kwargs):
        client, signature, types, inputs, questions, trace = self._prepare(kwargs)
        answers = client(state=serialize_object(inputs), questions=questions)
        return self._decode(answers, signature, types, inputs, trace)

    async def aforward(self, **kwargs):
        client, signature, types, inputs, questions, trace = self._prepare(kwargs)
        answers = await client.acall(state=serialize_object(inputs), questions=questions)
        return self._decode(answers, signature, types, inputs, trace)

    def dump_state(self, json_mode=True):
        if self.client is not None and not isinstance(self.client, TypeSafe):
            raise TypeError(
                "Saving an explicit Decide client requires dspy.experimental.TypeSafe; configure custom clients in settings."
            )
        return {
            "signature": self.signature.dump_state(),
            "thresholds": copy.deepcopy(self.thresholds),
            "weights": copy.deepcopy(self.weights),
            "client": self.client.dump_state() if self.client is not None else None,
        }

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        state = copy.deepcopy(state)
        client_state = state.pop("client", None)
        self.signature = self.signature.load_state(state["signature"])
        self.thresholds = state["thresholds"]
        self.weights = state["weights"]
        self.client = TypeSafe(**_sanitize_lm_state(client_state, allow_unsafe_lm_state)) if client_state else None
        self._validate_parameters(self._output_types(self.signature))
        return self
