import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.utils.exceptions import AdapterParseError


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None

class TraceData(TypedDict):
    example_ind: int
    example: Example
    prediction: Prediction
    trace: list[tuple[Any, dict[str, Any], Prediction]]
    score: float | None


class ProgramWrapper(Module):
    """
    A transparent wrapper around a dspy Module that:
    - Inherits from Module.
    - Delegates behavior to the wrapped program, with an optional call_wrapper(program, **kwargs)
      used when calling.
    - Avoids recursion during Module's metaclass-driven initialization by handling attribute
      setting carefully before _program exists.
    """

    def __init__(self, program: Module, call_wrapper: Callable[[Module,], Any] = None):
        # Initialize Module (this sets callbacks/history on the wrapper instance).
        super().__init__(callbacks=getattr(program, "callbacks", None))
        # Set internal fields directly.
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "_call_wrapper", call_wrapper)
        # Make wrapper's callbacks/history reference the wrapped program's lists for consistency.
        try:
            object.__setattr__(self, "history", program.history)
        except Exception:
            pass
        try:
            object.__setattr__(self, "callbacks", program.callbacks)
        except Exception:
            pass

    def __call__(self, *args, **kwargs):
        """
        If call_wrapper is provided, try to adapt the call to kwargs and call
        call_wrapper(program, **kwargs). Otherwise, or if adaptation fails, call the
        wrapped program directly to preserve behavior.
        """
        # If single dict-like positional arg, treat it as kwargs.
        if args and not kwargs and len(args) == 1 and isinstance(args[0], dict):
            kwargs = dict(args[0])
            args = ()

        # Try binding args to program signature so we can pass only kwargs to call_wrapper.
        if args:
            try:
                sig = inspect.signature(self._program)
                bound = sig.bind_partial(*args, **kwargs)
                kwargs = dict(bound.arguments)
                args = ()
            except Exception:
                # Can't safely adapt; preserve exact behavior by calling program directly.
                return self._program(*args, **kwargs)

        return self._call_wrapper(self._program, **kwargs)

    async def acall(self, *args, **kwargs):
        return await self._program.acall(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._program.forward(*args, **kwargs)

    async def aforward(self, *args, **kwargs):
        return await self._program.aforward(*args, **kwargs)

    # Predictor/LM APIs: delegate to wrapped program to preserve fidelity.
    def named_predictors(self):
        return self._program.named_predictors()

    def predictors(self):
        return self._program.predictors()

    def set_lm(self, lm):
        return self._program.set_lm(lm)

    def get_lm(self):
        return self._program.get_lm()

    def map_named_predictors(self, func):
        self._program.map_named_predictors(func)
        return self

    def inspect_history(self, n: int = 1):
        return self._program.inspect_history(n)

    def batch(
        self,
        examples,
        num_threads=None,
        max_errors=None,
        return_failed_examples=False,
        provide_traceback=None,
        disable_progress_bar=False,
    ):
        return self._program.batch(
            examples,
            num_threads=num_threads,
                max_errors=max_errors,
                return_failed_examples=return_failed_examples,
                provide_traceback=provide_traceback,
                disable_progress_bar=disable_progress_bar,
            )

    # Transparent attribute access with init-safe guards
    def __getattr__(self, name):
        # Only called if normal lookup fails. Avoid recursion before _program exists.
        if "_program" not in self.__dict__:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._program, name)

    def __setattr__(self, name, value):
        # During metaclass-initialization, _program isn't set yet; keep attributes local.
        if name in {"_program", "_call_wrapper"} or "_program" not in self.__dict__:
            object.__setattr__(self, name, value)
            return

        # If setting a data attribute that already lives on the wrapper instance, set locally.
        if name in getattr(self, "__dict__", {}):
            object.__setattr__(self, name, value)
            return

        # Delegate attribute setting to the wrapped program by default.
        setattr(self._program, name, value)

    def __delattr__(self, name):
        if name in {"_program", "_call_wrapper"} or "_program" not in self.__dict__:
            object.__delattr__(self, name)
            return

        if name in getattr(self, "__dict__", {}):
            object.__delattr__(self, name)
        else:
            delattr(self._program, name)

    # Introspection and representation
    def __dir__(self):
        own = set(dir(type(self))) | set(getattr(self, "__dict__", {}).keys())
        try:
            prog = set(dir(self._program))
        except Exception:
            prog = set()
        return sorted(own | prog)

    def __repr__(self):
        return repr(self._program)

    def __str__(self):
        return str(self._program)

    # Iteration support if the wrapped program is iterable
    def __iter__(self):
        return iter(self._program)

    # Equality and hashing delegate to the wrapped object
    def __eq__(self, other):
        return self._program == (other._program if isinstance(other, ProgramWrapper) else other)

    def __hash__(self):
        return hash(self._program)

    # Pickle support
    def __getstate__(self):
        # Let Module's __getstate__ do its normal pruning on the wrapper's dict,
        # but also include the wrapped program and call_wrapper explicitly.
        state = super().__getstate__()
        state["_program"] = self._program
        state["_call_wrapper"] = self._call_wrapper
        return state

    def __setstate__(self, state):
        # Restore wrapper internals first.
        object.__setattr__(self, "_program", state.pop("_program"))
        object.__setattr__(self, "_call_wrapper", state.pop("_call_wrapper"))
        # Restore the rest via Module's machinery.
        super().__setstate__(state)


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
) -> list[TraceData]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    def wrapped_program_callable(program_to_use: Module, **kwargs):
        with dspy.context(trace=[]):
            try:
                return program_to_use(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program_to_use.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting."
                    )

                return failed_pred, trace

    wrapped_program = ProgramWrapper(program, wrapped_program_callable)

    results = evaluator(wrapped_program, metric=wrapped_metric).results

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace = prediction
        except ValueError as ve:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting. This leads to a value error.
            # To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace, and pass it to in the GRPO group
            # with a high-negative user-configurable score.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data
