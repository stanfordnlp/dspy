import random

from pydantic import BaseModel

import dsp
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
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self):
        state_keys = ["lm", "traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                if isinstance(demo[field], BaseModel):
                    demo[field] = demo[field].model_dump_json()

            state["demos"].append(demo)

        # Cache the signature instructions and the last field's name.
        state["signature_instructions"] = self.signature.instructions

        *_, last_key = self.signature.fields.keys()
        state["signature_prefix"] = self.signature.fields[last_key].json_schema_extra["prefix"]

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
        new_signature = ensure_signature(kwargs.pop("new_signature", None))
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or dsp.settings.lm
        assert lm is not None, "No LM is loaded."

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature")
        temperature = lm.kwargs["temperature"] if temperature is None else temperature

        num_generations = config.get("n")
        if num_generations is None:
            num_generations = lm.kwargs.get("n", lm.kwargs.get("num_generations", None))

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7
            # print(f"#> Setting temperature to 0.7 since n={num_generations} and prior temperature={temperature}.")

        # All of the other kwargs are presumed to fit a prefix of the signature.
        # That is, they are input variables for the bottom most generation, so
        # we place them inside the input - x - together with the demos.
        x = dsp.Example(demos=demos, **kwargs)

        if new_signature is not None:
            signature = new_signature

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.")

        # Switch to legacy format for dsp.generate
        template = signature_to_template(signature)

        if self.lm is None:
            x, C = dsp.generate(template, **config)(x, stage=self.stage)
        else:
            # Note: query_only=True means the instructions and examples are not included.
            # I'm not really sure why we'd want to do that, but it's there.
            with dsp.settings.context(lm=self.lm, query_only=True):
                x, C = dsp.generate(template, **config)(x, stage=self.stage)

        assert self.stage in x, "The generated (input, output) example was not stored"

        completions = []

        for c in C:
            completions.append({})
            for field in template.fields:
                if field.output_variable not in kwargs.keys():
                    completions[-1][field.output_variable] = getattr(
                        c,
                        field.output_variable,
                    )

        pred = Prediction.from_completions(completions, signature=signature)

        if kwargs.pop("_trace", True) and dsp.settings.trace is not None:
            trace = dsp.settings.trace
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
