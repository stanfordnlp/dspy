"""Shared decision evidence and request translation for Predict backends."""

import copy
import json
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, create_model

from dspy.adapters.decision_state import DecisionState
from dspy.adapters.types.decision import Choice, Noul, Probability, Score


@lru_cache(maxsize=256)
def evidence_type(kind):
    """Closed schemas work with JSON structured outputs as well as chat adapters."""
    if issubclass(kind, Noul):
        fields = {"noul": (Probability, Field(description="Probability that the answer is true."))}
    else:
        labels = range(len(kind.options)) if issubclass(kind, Score) else [str(v) for v, _ in kind.options]
        probabilities = create_model(
            f"{kind.__name__}Probabilities",
            __config__=ConfigDict(extra="forbid"),
            **{f"option_{i}": (Probability, Field(alias=str(label))) for i, label in enumerate(labels)},
        )
        fields = {
            "probabilities": (probabilities, Field(description="Probability of each option; sum to one.")),
            "confidence": (Probability, Field(description="Confidence in the decision, from 0 to 1.")),
        }
    return create_model(f"{kind.__name__}Evidence", __config__=ConfigDict(extra="forbid"), **fields)


def resolve_adapter(lm, adapter, signature, fields, declared_signature=None):
    """Resolve backend translation before a chat adapter starts capability planning."""
    system_one = getattr(lm, "supports_decision_requests", False) is True
    state = DecisionState(signature, fields, system_one=system_one, declared_signature=declared_signature)
    return DecisionAdapter(adapter, state, system_one) if state.types else adapter


class DecisionAdapter:
    """Translate decision outputs once, independently of the chosen chat format."""

    def __init__(self, adapter, state, system_one):
        self.adapter = adapter
        self.state = state
        self.system_one = system_one

    def _prepare(self, signature, demos, inputs, lm_kwargs):
        from dspy.adapters.utils import get_field_description_string
        from dspy.dsp.utils.settings import settings
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
                "demos": [
                    serialize_object({k: v for k, v in demo.items() if k in signature.fields}) for demo in demos
                ],
            }
            return {"state": state, "questions": questions}
        for name, kind in types.items():
            signature = signature.with_updated_fields(
                name, type_=evidence_type(kind), desc=json.dumps(questions[name], ensure_ascii=False)
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
                    answer["probabilities"] = {int(k): v for k, v in answer["probabilities"].items()}
                elif issubclass(kind, Choice):
                    probabilities = answer["probabilities"]
                    labels = [str(value) for value, _ in kind.options]
                    if set(probabilities) != set(labels):
                        raise ValueError(f"Invalid Choice distribution for {name!r}.")
                    # Selection is derived for both backends. Ties use declared
                    # option order, never response-object key order.
                    answer["choice"] = max(labels, key=probabilities.get)
                answers[name] = answer
            results.append({**completion, **self.state._decode(answers)})
        return results

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        request = self._prepare(signature, demos, inputs, lm_kwargs)
        completions = [lm(**request)] if self.system_one else self.adapter(lm, **request)
        return self._decode(completions)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        request = self._prepare(signature, demos, inputs, lm_kwargs)
        completions = [await lm.acall(**request)] if self.system_one else await self.adapter.acall(lm, **request)
        return self._decode(completions)
