"""Compatibility entry point for the experimental standalone decision module."""

import copy

from pydantic import TypeAdapter

from dspy.adapters.decision_state import DecisionState
from dspy.adapters.utils import get_field_description_string
from dspy.clients.base_lm import BaseLM
from dspy.clients.typesafe import TypeSafe
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.predict.predict import _sanitize_lm_state, serialize_object
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.annotation import experimental


@experimental
class Decide(DecisionState, Module, Parameter):
    """Execute a closed-set System One signature using a dedicated client."""

    def __init__(self, signature, *, client=None, callbacks=None):
        Module.__init__(self, callbacks=callbacks)
        DecisionState.__init__(self, signature)
        self.client = client

    def reset(self):
        """Keep decision parameters intact."""

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
        questions = {name: self._question(name, field, types[name]) for name, field in signature.output_fields.items()}
        state = {
            "instructions": signature.instructions,
            "input_fields": get_field_description_string(signature.input_fields),
            "inputs": serialize_object(inputs),
        }
        return client, signature, types, inputs, state, questions, trace

    def _decode(self, answers, signature, types, inputs, trace):
        prediction = super()._decode(answers, signature, types)
        if trace and settings.trace is not None and settings.max_trace_size > 0:
            if len(settings.trace) >= settings.max_trace_size:
                settings.trace.pop(0)
            settings.trace.append((self, inputs, prediction))
        return prediction

    def forward(self, **kwargs):
        client, signature, types, inputs, state, questions, trace = self._prepare(kwargs)
        return self._decode(client(state=state, questions=questions), signature, types, inputs, trace)

    async def aforward(self, **kwargs):
        client, signature, types, inputs, state, questions, trace = self._prepare(kwargs)
        return self._decode(await client.acall(state=state, questions=questions), signature, types, inputs, trace)

    def dump_state(self, json_mode=True):
        self._validate_parameters(self._output_types(self.signature))
        if self.client is not None and not isinstance(self.client, TypeSafe):
            raise TypeError(
                "Saving an explicit Decide client requires dspy.experimental.TypeSafe; configure custom clients in settings."
            )
        return {
            "signature": self.signature.dump_state(),
            "fields": copy.deepcopy(self.fields),
            "client": self.client.dump_state() if self.client is not None else None,
        }

    def load_state(self, state, *, allow_unsafe_lm_state=False):
        state = copy.deepcopy(state)
        client_state = state.pop("client", None)
        restored = copy.copy(self)
        restored.signature = self.signature.load_state(state["signature"])
        restored.fields = state["fields"]
        if client_state:
            client_state = _sanitize_lm_state(client_state, allow_unsafe_lm_state)
            client_state.setdefault("_dspy_lm_class", "dspy.clients.typesafe.TypeSafe")
            restored.client = BaseLM.load_state(client_state, allow_custom_lm_class=allow_unsafe_lm_state)
        else:
            restored.client = None
        restored._validate_parameters(self._output_types(restored.signature))
        self.__dict__.update(restored.__dict__)
        return self
