import inspect
import logging
import json
from pathlib import Path

from ..utils.settings import settings, thread_local_overrides
from ..utils.usage_tracker import track_usage

logger = logging.getLogger(__name__)


class ProgramMeta(type):
    """Metaclass ensuring every ``dspy.Module`` instance is properly initialised."""

    def __call__(cls, *args, **kwargs):
        # Create the instance without invoking ``__init__`` so we can inject
        # the base initialization beforehand.
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            # ``_base_init`` sets attributes that should exist on all modules
            # even when a subclass forgets to call ``super().__init__``.
            Module._base_init(obj)
            cls.__init__(obj, *args, **kwargs)

            # Guarantee existence of critical attributes if ``__init__`` didn't
            # create them.
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj


class Module(metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        # LM calling history of the module.
        self.history = []

    def __call__(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = self.forward(*args, **kwargs)
                output.set_lm_usage(usage_tracker.get_total_tokens())
                return output

            return self.forward(*args, **kwargs)

    async def acall(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = await self.aforward(*args, **kwargs)
                    output.set_lm_usage(usage_tracker.get_total_tokens())
                    return output

            return await self.aforward(*args, **kwargs)

    def named_predictors(self):
        from ..predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def named_parameters(self):
        """Get named parameters of the module."""
        # Simplified implementation - return empty list for now
        return []

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def inspect_history(self, n: int = 1):
        from ..utils.inspect_history import pretty_print_history
        return pretty_print_history(self.history, n)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name == "forward" and callable(attr):
            # Check if forward is called through __call__ or directly
            stack = inspect.stack()
            forward_called_directly = len(stack) <= 1 or stack[1].function != "__call__"

            if forward_called_directly:
                logger.warning(
                    f"Calling {self.__class__.__name__}.forward() directly is discouraged. "
                    f"Please use {self.__class__.__name__}() instead."
                )

        return attr

    def dump_state(self):
        """Dump the state of the module."""
        # Simplified implementation - return basic state
        state = {
            "traces": getattr(self, "traces", []),
            "train": getattr(self, "train", []),
            "demos": getattr(self, "demos", []),
            "history": getattr(self, "history", []),
        }
        return state

    def load_state(self, state):
        """Load the state of the module."""
        # Simplified implementation - load basic state
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save(self, path, save_program=False, modules_to_serialize=None):
        """Save the module state to a file.
        
        Args:
            path (str): Path to save the state file (.json or .pkl)
            save_program (bool): Not supported in minimal version
            modules_to_serialize: Not supported in minimal version
        """
        if save_program:
            raise NotImplementedError("save_program=True is not supported in dspy_minimal")
        
        metadata = {}
        metadata["dependency_versions"] = get_dependency_versions()
        path = Path(path)

        state = self.dump_state()
        state["metadata"] = metadata
        
        if path.suffix == ".json":
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non "
                    "json-serializable objects."
                )
        elif path.suffix == ".pkl":
            raise NotImplementedError(".pkl saving is not supported in dspy_minimal")
        else:
            raise ValueError(f"`path` must end with `.json` when `save_program=False`, but received: {path}")

    def load(self, path):
        """Load the saved module state.
        
        Args:
            path (str): Path to the saved state file (.json)
        """
        path = Path(path)

        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
        else:
            raise ValueError(f"`path` must end with `.json`, but received: {path}")

        # Check dependency versions
        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state["metadata"]["dependency_versions"]
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions.get(key) != saved_version:
                logger.warning(f"Version mismatch for {key}: saved={saved_version}, current={dependency_versions.get(key)}")

        # Remove metadata before loading state
        state.pop("metadata", None)
        return self.load_state(state)


def set_attribute_by_name(obj, name, value):
    # Simplified version without magicattr dependency
    setattr(obj, name, value)


def get_dependency_versions():
    """Get dependency versions for saving/loading."""
    return {
        "dspy_minimal": "1.0.0",  # Simplified version tracking
        "pydantic": "2.8.0",
        "requests": "2.31.0",
    } 