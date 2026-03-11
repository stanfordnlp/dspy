import logging
import random
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from typeguard import TypeCheckError, check_type

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback
from dspy.utils.constants import IS_TYPE_UNDEFINED

logger = logging.getLogger(__name__)

UNSAFE_LM_STATE_KEYS = {"api_base", "base_url", "model_list"}


def _sanitize_lm_state(lm_state: dict, allow_unsafe_lm_state: bool) -> dict:
    if allow_unsafe_lm_state:
        return lm_state

    unsafe_keys = sorted(UNSAFE_LM_STATE_KEYS.intersection(lm_state))

    if not unsafe_keys:
        return lm_state

    sanitized_lm_state = {k: v for k, v in lm_state.items() if k not in UNSAFE_LM_STATE_KEYS}
    logger.warning(
        "Ignoring unsafe LM config key(s) during state load: %s. "
        "Pass allow_unsafe_lm_state=True to preserve these keys for trusted files.",
        unsafe_keys,
    )
    return sanitized_lm_state


def _resolve_signature_override(
    signature_override: str | type[Signature] | None,
    inputs: dict[str, Any],
    input_field_names: set[str],
) -> str | type[Signature] | None:
    legacy_signature = None
    if "signature" in inputs and "signature" not in input_field_names:
        legacy_signature = inputs.pop("signature")

    if signature_override is not None and legacy_signature is not None:
        raise TypeError("Pass either `signature_override=` or legacy `signature=`, not both.")

    return signature_override if signature_override is not None else legacy_signature


class Predict(Module, Parameter):
    """Basic DSPy module that maps inputs to outputs using a language model.

    Args:
        signature: The input/output signature describing the task.
        callbacks: Optional list of callbacks for instrumentation.
        **config: Default keyword arguments forwarded to the underlying
            language model. These values can be overridden for a single
            invocation by passing a ``config`` dictionary when calling the
            module. For example::

                predict = dspy.Predict("q -> a", rollout_id=1, temperature=1.0)
                predict(q="What is 1 + 52?", config={"rollout_id": 2, "temperature": 1.0})
    """

    def __init__(self, signature: str | type[Signature], callbacks: list[BaseCallback] | None = None, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self, json_mode=True):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                demo[field] = serialize_object(demo[field])

            if isinstance(demo, dict) or not json_mode:
                state["demos"].append(demo)
            else:
                state["demos"].append(demo.toDict())

        state["signature"] = self.signature.dump_state()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    def load_state(self, state: dict, *, allow_unsafe_lm_state: bool = False) -> "Predict":
        """Load the saved state of a `Predict` object.

        Args:
            state: The saved state of a `Predict` object.
            allow_unsafe_lm_state: If True, preserves `api_base`, `base_url`, and `model_list` from
                serialized LM state. Enable only when loading trusted files.

        Returns:
            Self to allow method chaining.
        """
        excluded_keys = ["signature", "extended_signature", "lm"]
        for name, value in state.items():
            # `excluded_keys` are fields that go through special handling.
            if name not in excluded_keys:
                setattr(self, name, value)

        self.signature = self.signature.load_state(state["signature"])
        sanitized_lm_state = _sanitize_lm_state(state["lm"], allow_unsafe_lm_state) if state["lm"] else None
        self.lm = LM(**sanitized_lm_state) if sanitized_lm_state else None

        if "extended_signature" in state:  # legacy, up to and including 2.5, for CoT.
            raise NotImplementedError("Loading extended_signature is no longer supported in DSPy 2.6+")

        return self

    def _get_positional_args_error_message(self):
        input_fields = list(self.signature.input_fields.keys())
        return (
            "Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments "
            f"that match your signature input fields: '{', '.join(input_fields)}'. For example: "
            f"`predict({input_fields[0]}=input_value, ...)`."
        )

    def __call__(
        self,
        *args,
        config: dict | None = None,
        signature_override: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        signature_override = _resolve_signature_override(
            signature_override=signature_override,
            inputs=inputs,
            input_field_names=set(self.signature.input_fields),
        )

        call_kwargs = dict(inputs)
        if config is not None:
            call_kwargs["config"] = config
        if signature_override is not None:
            call_kwargs["signature_override"] = signature_override
        if demos is not None:
            call_kwargs["demos"] = demos
        if lm is not None:
            call_kwargs["lm"] = lm

        return super().__call__(**call_kwargs)

    async def acall(
        self,
        *args,
        config: dict | None = None,
        signature_override: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        signature_override = _resolve_signature_override(
            signature_override=signature_override,
            inputs=inputs,
            input_field_names=set(self.signature.input_fields),
        )

        call_kwargs = dict(inputs)
        if config is not None:
            call_kwargs["config"] = config
        if signature_override is not None:
            call_kwargs["signature_override"] = signature_override
        if demos is not None:
            call_kwargs["demos"] = demos
        if lm is not None:
            call_kwargs["lm"] = lm

        return await super().acall(**call_kwargs)

    def _forward_preprocess(
        self,
        *,
        config: dict | None = None,
        signature_override: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        # Extract the privileged keyword arguments.
        assert "new_signature" not in inputs, "new_signature is no longer a valid keyword argument."
        signature_override = _resolve_signature_override(
            signature_override=signature_override,
            inputs=inputs,
            input_field_names=set(self.signature.input_fields),
        )
        effective_signature = ensure_signature(self.signature if signature_override is None else signature_override)
        demos = self.demos if demos is None else demos
        config = {**self.config, **({} if config is None else config)}

        # Get the right LM to use.
        lm = self.lm if lm is None else lm
        lm = lm or settings.lm

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            # Many users mistakenly use `dspy.configure(lm="openai/gpt-4o-mini")` instead of
            # `dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))`, so we are providing a specific error message.
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

        if "prediction" in inputs:
            if (
                isinstance(inputs["prediction"], dict)
                and inputs["prediction"].get("type") == "content"
                and "content" in inputs["prediction"]
            ):
                # If the `prediction` is the standard predicted outputs format
                # (https://platform.openai.com/docs/guides/predicted-outputs), we remove it from input kwargs and add it
                # to the lm kwargs.
                config["prediction"] = inputs.pop("prediction")

        # Populate default values for missing input fields.
        for k, v in effective_signature.input_fields.items():
            if k not in inputs and v.default is not PydanticUndefined:
                inputs[k] = v.default

        # Check and warn for extra fields not in signature
        extra_fields = [k for k in inputs if k not in effective_signature.input_fields]
        if extra_fields:
            logger.warning(
                "Input contains fields not in signature. These fields will be ignored: %s. "
                "Expected fields: %s.",
                extra_fields,
                list(effective_signature.input_fields.keys()),
            )

        # Validate input field types match signature
        if settings.warn_on_type_mismatch:
            for field_name, field_info in effective_signature.input_fields.items():
                if field_name in inputs:
                    value = inputs[field_name]
                    expected_type: type = field_info.annotation
                    json_schema_extra = field_info.json_schema_extra or {}

                    if value is None or json_schema_extra.get(IS_TYPE_UNDEFINED, False):
                        continue

                    if not _is_value_compatible_with_type(value, expected_type):
                        logger.warning(
                            "Type mismatch for field '%s': expected %s based on given Signature, "
                            "but the provided value is incompatible: %s.",
                            field_name,
                            _get_type_name(expected_type),
                            value,
                        )

        if not all(k in inputs for k in effective_signature.input_fields):
            present = [k for k in effective_signature.input_fields if k in inputs]
            missing = [k for k in effective_signature.input_fields if k not in inputs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, effective_signature, demos, inputs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None and settings.max_trace_size > 0:
            trace = settings.trace
            if len(trace) >= settings.max_trace_size:
                trace.pop(0)
            trace.append((self, {**kwargs}, pred))
        return pred

    def _should_stream(self):
        stream_listeners = settings.stream_listeners or []
        should_stream = settings.send_stream is not None
        if should_stream and len(stream_listeners) > 0:
            should_stream = any(stream_listener.predict == self for stream_listener in stream_listeners)

        return should_stream

    def forward(
        self,
        *,
        config: dict | None = None,
        signature_override: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        lm, config, effective_signature, demos, inputs = self._forward_preprocess(
            config=config,
            signature_override=signature_override,
            demos=demos,
            lm=lm,
            **inputs,
        )

        adapter = settings.adapter or ChatAdapter()

        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = adapter(lm, lm_kwargs=config, signature=effective_signature, demos=demos, inputs=inputs)
        else:
            with settings.context(send_stream=None):
                completions = adapter(lm, lm_kwargs=config, signature=effective_signature, demos=demos, inputs=inputs)

        return self._forward_postprocess(completions, effective_signature, **inputs)

    async def aforward(
        self,
        *,
        config: dict | None = None,
        signature_override: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        lm, config, effective_signature, demos, inputs = self._forward_preprocess(
            config=config,
            signature_override=signature_override,
            demos=demos,
            lm=lm,
            **inputs,
        )

        adapter = settings.adapter or ChatAdapter()
        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = await adapter.acall(
                    lm,
                    lm_kwargs=config,
                    signature=effective_signature,
                    demos=demos,
                    inputs=inputs,
                )
        else:
            with settings.context(send_stream=None):
                completions = await adapter.acall(
                    lm,
                    lm_kwargs=config,
                    signature=effective_signature,
                    demos=demos,
                    inputs=inputs,
                )

        return self._forward_postprocess(completions, effective_signature, **inputs)

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"

def _get_type_name(type_annotation) -> str:
    """Helper method to get the name for a type annotation."""

    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    if origin is None:
        # Primitives like str, int, etc.
        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__
        return str(type_annotation)

    # Handle Literal types
    if origin is Literal:
        literal_values = ", ".join(repr(arg) for arg in args)
        return f"Literal[{literal_values}]"

    # Types like list[str], dict[str, int], generics, etc.
    if args:
        # Handle Ellipsis in tuples (e.g., tuple[int, ...])
        args_str = ", ".join("..." if arg is ... else _get_type_name(arg) for arg in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"

    return getattr(origin, "__name__", str(origin))

def _is_value_compatible_with_type(value: Any, expected: type) -> bool:
    """Return True if the value matches the expected type hint."""
    try:
        # Special handle list[str] because we allow setting input type to str, however, invoking with a list thereof.
        if expected is str and isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return True

        check_type(value, expected)
        return True
    except TypeCheckError:
        return False

def serialize_object(obj):
    """
    Recursively serialize a given object into a JSON-compatible format.
    Supports Pydantic models, lists, dicts, and primitive types.
    """
    if isinstance(obj, BaseModel):
        # Use model_dump with mode="json" to ensure all fields (including HttpUrl, datetime, etc.)
        # are converted to JSON-serializable types (strings)
        return obj.model_dump(mode="json")
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    else:
        return obj


# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
