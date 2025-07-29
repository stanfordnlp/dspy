import logging
import random
from typing import Union

from pydantic import BaseModel

from ..clients.base_lm import BaseLM
from ..clients.lm import LM
from ..utils.settings import settings
from ..primitives.module import Module
from ..primitives.prediction import Prediction
from ..signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)


class Predict(Module):
    def __init__(self, signature: Union[str, type[Signature]], callbacks=None, instructions=None, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature, instructions)
        self.config = config
        self.demos = []  # Store optimized demonstrations
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def _get_positional_args_error_message(self):
        input_fields = list(self.signature.input_fields.keys())
        return (
            "Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments "
            f"that match your signature input fields: '{', '.join(input_fields)}'. For example: "
            f"`predict({input_fields[0]}=input_value, ...)`."
        )

    def __call__(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        return super().__call__(**kwargs)

    async def acall(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        return await super().acall(**kwargs)

    def _forward_preprocess(self, **kwargs):
        # Extract the three privileged keyword arguments.
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or settings.lm

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            raise ValueError(
                f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
                f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
            )
        elif not isinstance(lm, BaseLM):
            raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

        # If temperature is unset or <=0.15, and n > 1, set temperature to 0.7 to keep randomness.
        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        # DEBUG LOGGING
        logger.debug(f"[DEBUG] _forward_preprocess called with kwargs: {kwargs}")
        logger.debug(f"[DEBUG] signature.input_fields: {list(signature.input_fields.keys())}")
        present = [k for k in signature.input_fields if k in kwargs]
        missing = [k for k in signature.input_fields if k not in kwargs]
        logger.debug(f"[DEBUG] Present: {present}")
        logger.debug(f"[DEBUG] Missing: {missing}")
        if not all(k in kwargs for k in signature.input_fields):
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, signature, demos, kwargs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None:
            trace = settings.trace
            trace.append((self, {**kwargs}, pred))
        return pred

    def forward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        # Use ChatAdapter for proper signature-based formatting
        from ..adapters.chat_adapter import ChatAdapter
        adapter = ChatAdapter()
        completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    async def aforward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        # Use ChatAdapter for proper signature-based formatting
        from ..adapters.chat_adapter import ChatAdapter
        adapter = ChatAdapter()
        completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"

    def dump_state(self):
        """Dump the state of the Predict object."""
        state = super().dump_state()
        # Store the signature string representation, not just the class name
        if hasattr(self.signature, 'signature'):
            # If it's a Signature class, get its string representation
            state["signature"] = self.signature.signature
        elif isinstance(self.signature, str):
            # If it's already a string, store it as is
            state["signature"] = self.signature
        else:
            # Fallback to string representation
            state["signature"] = str(self.signature)
        
        # Store the instructions separately
        if hasattr(self.signature, 'instructions'):
            state["instructions"] = self.signature.instructions
        
        state["config"] = self.config
        state["demos"] = self.demos  # Store the optimized demos
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    def load_state(self, state):
        """Load the state of the Predict object."""
        # Load base state
        super().load_state(state)
        
        # Load signature - reconstruct it from the stored string
        signature_str = state.get("signature", "input -> output")
        instructions = state.get("instructions")
        
        if isinstance(signature_str, str):
            # Reconstruct the signature from the string with instructions
            self.signature = ensure_signature(signature_str, instructions)
        else:
            self.signature = signature_str
        
        # Load config
        self.config = state.get("config", {})
        
        # Load demos
        self.demos = state.get("demos", [])
        
        # Load LM (simplified - would need proper LM reconstruction)
        lm_state = state.get("lm")
        if lm_state and self.lm:
            self.lm.load_state(lm_state)
        
        return self


def serialize_object(obj):
    """Simple serialization for objects."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj) 