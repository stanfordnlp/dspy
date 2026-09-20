"""A closed-set predictor backed by a System One model."""

import copy
import itertools
import math
from typing import Literal, get_origin

from pydantic import TypeAdapter

from dspy.adapters.types.decision import Noul, Score, _Decision, decision_type
from dspy.adapters.utils import get_field_description_string
from dspy.clients.typesafe import TypeSafe
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict, _sanitize_lm_state, serialize_object
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.annotation import experimental
from dspy.utils.callback import BaseCallback


@experimental
class Decide(Predict):
    """Answer a signature's closed-set outputs in one System One request.

    Declare outputs using Noul, Score[(number, description), ...], and
    Choice[(value, description), ...]. Native bool, Literal, and
    Annotated[float, Score[...]] return only the value. Bare float has no rubric.
    Unsupported outputs raise; Decide never falls back to a generative LM.

    Each instance owns ``thresholds`` (one per Boolean output, initially 0.5)
    and ``weights`` (one list per Score output, initially the declared anchors).
    Weights are numeric option values, not probability multipliers. They must
    increase strictly within the declared range. Changing these parameters
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
        super().__init__(signature, callbacks=callbacks)
        self.client = client
        types = self._output_types(self.signature)
        self.thresholds = {name: 0.5 for name, kind in types.items() if kind is Noul}
        self.weights = {name: [v for v, _ in kind.options] for name, kind in types.items() if issubclass(kind, Score)}

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
        if set(self.weights) != {name for name, kind in types.items() if issubclass(kind, Score)}:
            raise ValueError("Decide weights must match its Score output fields.")
        for name, weights in self.weights.items():
            options = types[name].options
            if (
                len(weights) != len(options)
                or any(type(w) not in (int, float) or not math.isfinite(w) for w in weights)
                or any(a >= b for a, b in itertools.pairwise(weights))
                or weights[0] < options[0][0]
                or weights[-1] > options[-1][0]
            ):
                raise ValueError(f"Weights for {name!r} must increase strictly within the declared Score range.")

    def _questions(self, signature, types, demos):
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
            examples = []
            for demo in demos:
                if name in demo:
                    answer = demo[name]
                    if isinstance(answer, _Decision):
                        answer = answer.value
                    examples.append(
                        {
                            "inputs": {k: serialize_object(demo[k]) for k in signature.input_fields if k in demo},
                            "answer": serialize_object(answer),
                        }
                    )
            if examples:
                instructions["examples"] = examples
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
        kwargs = dict(kwargs)
        trace = kwargs.pop("_trace", True)
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        client = self.client if self.client is not None else settings.system_one
        if client is None:
            raise ValueError(
                "Configure a System One client with dspy.configure(system_one=dspy.experimental.TypeSafe(...))."
            )
        types = self._output_types(signature)
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
        questions = self._questions(signature, types, demos)
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
                result = kind(
                    value=options[answer["choice"]],
                    confidence=answer["confidence"],
                    probabilities=answer["probabilities"],
                )
            native = field.annotation in (bool, float) or get_origin(field.annotation) is Literal
            outputs[name] = result.value if native else result
        return self._forward_postprocess([outputs], signature, _trace=trace, **inputs)

    def forward(self, **kwargs):
        client, signature, types, inputs, questions, trace = self._prepare(kwargs)
        answers = client(state=serialize_object(inputs), questions=questions)
        return self._decode(answers, signature, types, inputs, trace)

    async def aforward(self, **kwargs):
        client, signature, types, inputs, questions, trace = self._prepare(kwargs)
        answers = await client.acall(state=serialize_object(inputs), questions=questions)
        return self._decode(answers, signature, types, inputs, trace)

    def dump_state(self, json_mode=True):
        state = super().dump_state(json_mode=json_mode)
        state["thresholds"] = copy.deepcopy(self.thresholds)
        state["weights"] = copy.deepcopy(self.weights)
        if self.client is not None and not isinstance(self.client, TypeSafe):
            raise TypeError(
                "Saving an explicit Decide client requires dspy.experimental.TypeSafe; configure custom clients in settings."
            )
        state["client"] = self.client.dump_state() if self.client is not None else None
        return state

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        state = copy.deepcopy(state)
        client_state = state.pop("client", None)
        super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)
        self.client = TypeSafe(**_sanitize_lm_state(client_state, allow_unsafe_lm_state)) if client_state else None
        self._validate_parameters(self._output_types(self.signature))
        for demo in self.demos:
            for name, field in self.signature.fields.items():
                if name in demo and isinstance(field.annotation, type) and issubclass(field.annotation, _Decision):
                    demo[name] = field.annotation.model_validate(demo[name])
        return self
