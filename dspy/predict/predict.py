import logging
import random
from functools import lru_cache

from pydantic import BaseModel

from dspy.adapters.image_utils import Image
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from dspy.utils.callback import with_callbacks


@lru_cache(maxsize=None)
def warn_once(msg: str):
    logging.warning(msg)


class Predict(Module, Parameter):
    def __init__(self, signature, _parse_values=True, callbacks=None, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.callbacks = callbacks or []
        self._parse_values = _parse_values
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
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                # It does matter for images
                if isinstance(demo[field], Image):
                    demo[field] = demo[field].model_dump()
                elif isinstance(demo[field], BaseModel):
                    demo[field] = demo[field].model_dump_json()

            state["demos"].append(demo)

        state["signature"] = self.signature.dump_state()
        # `extended_signature` is a special field for `Predict`s like CoT.
        if hasattr(self, "extended_signature"):
            state["extended_signature"] = self.extended_signature.dump_state()
        return state

    def load_state(self, state):
        """Load the saved state of a `Predict` object.

        Args:
            state (dict): The saved state of a `Predict` object.

        Returns:
            self: Returns self to allow method chaining
        """
        excluded_keys = ["signature", "extended_signature"]
        for name, value in state.items():
            # `excluded_keys` are fields that go through special handling.
            if name not in excluded_keys:
                setattr(self, name, value)

        # FIXME: Images are getting special treatment, but all basemodels initialized from json should be converted back to objects
        for demo in self.demos:
            for field in demo:
                if isinstance(demo[field], dict) and "url" in demo[field]:
                    url = demo[field]["url"]
                    if not isinstance(url, str):
                        raise ValueError(f"Image URL must be a string, got {type(url)}")
                    demo[field] = Image(url=url)

        self.signature = self.signature.load_state(state["signature"])

        if "extended_signature" in state:
            self.extended_signature = self.extended_signature.load_state(state["extended_signature"])

        return self

    @with_callbacks
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        import dspy
        assert not dspy.settings.compiling, "It's no longer ever the case that .compiling is True"

        # Extract the three privileged keyword arguments.
        new_signature = ensure_signature(kwargs.pop("new_signature", None))
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or dspy.settings.lm
        assert lm is not None, "No LM is loaded."

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature")
        temperature = lm.kwargs["temperature"] if temperature is None else temperature
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if new_signature is not None:
            signature = new_signature

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.")

        assert isinstance(lm, dspy.LM)
        completions = v2_5_generate(lm, config, signature, demos, kwargs, _parse_values=self._parse_values)

        pred = Prediction.from_completions(completions, signature=signature)

        if kwargs.pop("_trace", True) and dspy.settings.trace is not None:
            trace = dspy.settings.trace
            trace.append((self, {**kwargs}, pred))

        return pred

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"


def v2_5_generate(lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    import dspy

    adapter = dspy.settings.adapter or dspy.ChatAdapter()

    return adapter(
        lm, lm_kwargs=lm_kwargs, signature=signature, demos=demos, inputs=inputs, _parse_values=_parse_values
    )


# TODO: get some defaults during init from the context window?
# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
