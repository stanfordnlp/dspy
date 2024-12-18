from functools import lru_cache
import random

import logging

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
    def __init__(self, signature, callbacks=None, **config):
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.callbacks = callbacks or []
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

        if "extended_signature" in state:  # legacy, up to and including 2.5, for CoT.
            raise NotImplementedError("Loading extended_signature is no longer supported in DSPy 2.6+")

        return self

    def _load_state_legacy(self, state):
        """Legacy state loading for backwards compatibility.

        This method is used to load the saved state of a `Predict` object from a version of DSPy prior to v2.5.3.
        Returns:
            self: Returns self to allow method chaining
        """
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

        # Some special stuff for CoT.
        if "extended_signature_instructions" in state:
            instructions = state["extended_signature_instructions"]
            self.extended_signature = self.extended_signature.with_instructions(instructions)

        if "extended_signature_prefix" in state:
            prefix = state["extended_signature_prefix"]
            *_, last_key = self.extended_signature.fields.keys()
            self.extended_signature = self.extended_signature.with_updated_fields(last_key, prefix=prefix)

        return self

    def load(self, path, return_self=False):
        """Load a saved state from a file.

        Args:
            path (str): Path to the saved state file
            return_self (bool): If True, returns self to allow method chaining. Default is False for backwards compatibility.

        Returns:
            Union[None, Predict]: Returns None if return_self is False (default), returns self if return_self is True
        """
        super().load(path)
        return self if return_self else None

    @with_callbacks
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        import dspy

        # Extract the three privileged keyword arguments.
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or dspy.settings.lm
        assert isinstance(lm, dspy.LM), "No LM is loaded."

        # If temperature is 0.0 but its n > 1, set temperature to 0.7.
        temperature = config.get("temperature")
        temperature = lm.kwargs["temperature"] if temperature is None else temperature
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            print(f"WARNING: Not all input fields were provided to module. Present: {present}. Missing: {missing}.")

        _trace = kwargs.pop("_trace", True)
        inputs = kwargs

        import dspy

        if hasattr(self, "retrieve_demos") and callable(self.retrieve_demos):
            demos = demos[:] + self.retrieve_demos(**inputs)
            random.Random(self.random_seed).shuffle(demos)

        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        pred = Prediction.from_completions(completions, signature=signature)

        if _trace and dspy.settings.trace is not None:
            trace = dspy.settings.trace
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
