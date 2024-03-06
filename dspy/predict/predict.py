import dspy
import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction

from dspy.signatures.signature import ensure_signature, signature_to_template


class Predict(Parameter):
    def __init__(self, signature, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.reset()

    def reset(self):
        self.backend = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self):
        state_keys = ["backend", "traces", "train", "demos"]
        state = {k: getattr(self, k) for k in state_keys}

        # Cache the signature instructions and the last field's name.
        state["signature_instructions"] = self.signature.instructions

        *_, last_key = self.signature.fields.keys()
        state["signature_prefix"] = self.signature.fields[last_key].json_schema_extra[
            "prefix"
        ]

        return state

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

        # Reconstruct the signature.
        if "signature_instructions" in state:
            instructions = state["signature_instructions"]
            self.signature = self.signature.with_instructions(instructions)

        if "signature_prefix" in state:
            prefix = state["signature_prefix"]
            *_, last_key = self.signature.fields.keys()
            self.signature = self.signature.with_updated_fields(last_key, prefix=prefix)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # Extract the three privileged keyword arguments.
        signature = kwargs.pop("new_signature", kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}), **kwargs)

        # Get the right Backend to use.
        backend = kwargs.pop("backend", self.backend) or dspy.settings.get(
            "backend", None
        )
        assert backend is not None, "No Backend is configured."

        x = dspy.Example(demos=demos, **kwargs)

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(
                f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}."
            )

        completions = backend(signature, **config)

        # TODO: What purpose does stage play here?
        # assert self.stage in x, "The generated (input, output) example was not stored"

        pred = Prediction.from_completions(completions)

        trace = dspy.settings.get("trace")
        if trace is not None and kwargs.pop("_trace", True):
            trace.append((self, {**kwargs}, pred))

        return pred

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"


# TODO: get some defaults during init from the context window?
# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
