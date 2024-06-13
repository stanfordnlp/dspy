import inspect
import uuid
from typing import Any

import dsp
import dspy

#################### Assertion Helpers ####################


def _build_error_msg(feedback_msgs):
    """Build an error message from a list of feedback messages."""
    return "\n".join([msg for msg in feedback_msgs])


#################### Assertion Exceptions ####################


class DSPyAssertionError(AssertionError):
    """Custom exception raised when a DSPy `Assert` fails."""

    def __init__(
        self,
        id: str,
        msg: str,
        target_module: Any = None,
        state: Any = None,
        is_metric: bool = False,
    ) -> None:
        super().__init__(msg)
        self.id = id
        self.msg = msg
        self.target_module = target_module
        self.state = state
        self.is_metric = is_metric


class DSPySuggestionError(AssertionError):
    """Custom exception raised when a DSPy `Suggest` fails."""

    def __init__(
        self,
        id: str,
        msg: str,
        target_module: Any = None,
        state: Any = None,
        is_metric: bool = False,
    ) -> None:
        super().__init__(msg)
        self.id = id
        self.msg = msg
        self.target_module = target_module
        self.state = state
        self.is_metric = is_metric


#################### Assertion Primitives ####################


class Constraint:
    def __init__(
        self,
        result: bool,
        msg: str = "",
        target_module=None,
        is_metric: bool = False,
    ):
        self.id = str(uuid.uuid4())
        self.result = result
        self.msg = msg
        self.target_module = target_module
        self.is_metric = is_metric

        self.__call__()


class Assert(Constraint):
    """DSPy Assertion"""

    def __call__(self) -> bool:
        if isinstance(self.result, bool):
            if self.result:
                return True
            elif dspy.settings.bypass_assert:
                dspy.logger.error(f"AssertionError: {self.msg}")
                return True
            else:
                dspy.logger.error(f"AssertionError: {self.msg}")
                raise DSPyAssertionError(
                    id=self.id,
                    msg=self.msg,
                    target_module=self.target_module,
                    state=dsp.settings.trace,
                    is_metric=self.is_metric,
                )
        else:
            raise ValueError("Assertion function should always return [bool]")


class Suggest(Constraint):
    """DSPy Suggestion"""

    def __call__(self) -> Any:
        if isinstance(self.result, bool):
            if self.result:
                return True
            elif dspy.settings.bypass_suggest:
                dspy.logger.info(f"SuggestionFailed: {self.msg}")
                return True
            else:
                dspy.logger.info(f"SuggestionFailed: {self.msg}")
                raise DSPySuggestionError(
                    id=self.id,
                    msg=self.msg,
                    target_module=self.target_module,
                    state=dsp.settings.trace,
                    is_metric=self.is_metric,
                )
        else:
            raise ValueError("Suggestion function should always return [bool]")


#################### Assertion Handlers ####################


def noop_handler(func):
    """Handler to bypass assertions and suggestions.

    Now both assertions and suggestions will become noops.
    """

    def wrapper(*args, **kwargs):
        with dspy.settings.context(bypass_assert=True, bypass_suggest=True):
            return func(*args, **kwargs)

    return wrapper


def bypass_suggest_handler(func):
    """Handler to bypass suggest only.

    If a suggestion fails, it will be logged but not raised.
    And If an assertion fails, it will be raised.
    """

    def wrapper(*args, **kwargs):
        with dspy.settings.context(bypass_suggest=True, bypass_assert=False):
            return func(*args, **kwargs)

    return wrapper


def bypass_assert_handler(func):
    """Handler to bypass assertion only.

    If a assertion fails, it will be logged but not raised.
    And If an assertion fails, it will be raised.
    """

    def wrapper(*args, **kwargs):
        with dspy.settings.context(bypass_assert=True):
            return func(*args, **kwargs)

    return wrapper


def assert_no_except_handler(func):
    """Handler to ignore assertion failure and return None."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DSPyAssertionError:
            return None

    return wrapper


def backtrack_handler(func, bypass_suggest=True, max_backtracks=2):
    """Handler for backtracking suggestion and assertion.

    Re-run the latest predictor up to `max_backtracks` times,
    with updated signature if an assertion fails. updated signature adds a new
    input field to the signature, which is the feedback.
    """

    def wrapper(*args, **kwargs):
        error_msg, result = None, None
        with dspy.settings.lock:
            dspy.settings.backtrack_to = None
            dspy.settings.suggest_failures = 0
            dspy.settings.assert_failures = 0

            # Predictor -> List[feedback_msg]
            dspy.settings.predictor_feedbacks = {}

            current_error = None
            for i in range(max_backtracks + 1):
                if i > 0 and dspy.settings.backtrack_to is not None:
                    # generate values for new fields
                    feedback_msg = _build_error_msg(
                        dspy.settings.predictor_feedbacks[dspy.settings.backtrack_to],
                    )

                    dspy.settings.backtrack_to_args = {
                        "feedback": feedback_msg,
                        "past_outputs": past_outputs,
                    }

                # if last backtrack: ignore suggestion errors
                if i == max_backtracks:
                    if isinstance(current_error, DSPyAssertionError):
                        raise current_error
                    dsp.settings.trace.clear()
                    result = bypass_suggest_handler(func)(*args, **kwargs) if bypass_suggest else None
                    break
                else:
                    try:
                        dsp.settings.trace.clear()
                        result = func(*args, **kwargs)
                        break
                    except (DSPySuggestionError, DSPyAssertionError) as e:
                        if not current_error:
                            current_error = e
                        error_id, error_msg, error_target_module, error_state = (
                            e.id,
                            e.msg,
                            e.target_module,
                            e.state[-1],
                        )

                        # increment failure count depending on type of error
                        if isinstance(e, DSPySuggestionError) and e.is_metric:
                            dspy.settings.suggest_failures += 1
                        elif isinstance(e, DSPyAssertionError) and e.is_metric:
                            dspy.settings.assert_failures += 1

                        if dsp.settings.trace:
                            if error_target_module:
                                for i in range(len(dsp.settings.trace) - 1, -1, -1):
                                    trace_element = dsp.settings.trace[i]
                                    mod = trace_element[0]
                                    if mod.signature == error_target_module:
                                        error_state = e.state[i]
                                        dspy.settings.backtrack_to = mod
                                        break
                            else:
                                dspy.settings.backtrack_to = dsp.settings.trace[-1][0]

                            if dspy.settings.backtrack_to is None:
                                dspy.logger.error("Specified module not found in trace")

                            # save unique feedback message for predictor
                            if error_msg not in dspy.settings.predictor_feedbacks.setdefault(
                                dspy.settings.backtrack_to,
                                [],
                            ):
                                dspy.settings.predictor_feedbacks[dspy.settings.backtrack_to].append(error_msg)

                            output_fields = error_state[0].new_signature.output_fields
                            past_outputs = {}
                            for field_name in output_fields.keys():
                                past_outputs[field_name] = getattr(
                                    error_state[2],
                                    field_name,
                                    None,
                                )

                            # save latest failure trace for predictor per suggestion
                            error_ip = error_state[1]
                            error_op = error_state[2].__dict__["_store"]
                            error_op.pop("_assert_feedback", None)
                            error_op.pop("_assert_traces", None)

                        else:
                            dspy.logger.error(
                                "UNREACHABLE: No trace available, this should not happen. Is this run time?",
                            )

            return result

    return wrapper


def handle_assert_forward(assertion_handler, **handler_args):
    def forward(self, *args, **kwargs):
        args_to_vals = inspect.getcallargs(self._forward, *args, **kwargs)

        # if user has specified a bypass_assert flag, set it
        if "bypass_assert" in args_to_vals:
            dspy.settings.configure(bypass_assert=args_to_vals["bypass_assert"])

        wrapped_forward = assertion_handler(self._forward, **handler_args)
        return wrapped_forward(*args, **kwargs)

    return forward


default_assertion_handler = backtrack_handler


def assert_transform_module(
    module,
    assertion_handler=default_assertion_handler,
    **handler_args,
):
    """
    Transform a module to handle assertions.
    """
    if not getattr(module, "forward", False):
        raise ValueError(
            "Module must have a forward method to have assertions handled.",
        )
    if getattr(module, "_forward", False):
        dspy.logger.info(
            f"Module {module.__class__.__name__} already has a _forward method. Skipping...",
        )
        pass  # TODO warning: might be overwriting a previous _forward method

    module._forward = module.forward
    module.forward = handle_assert_forward(assertion_handler, **handler_args).__get__(
        module,
    )

    if all(
        map(lambda p: isinstance(p[1], dspy.retry.Retry), module.named_predictors()),
    ):
        pass  # we already applied the Retry mapping outside
    elif all(
        map(lambda p: not isinstance(p[1], dspy.retry.Retry), module.named_predictors()),
    ):
        module.map_named_predictors(dspy.retry.Retry)
    else:
        raise RuntimeError("Module has mixed predictors, can't apply Retry mapping.")

    module._assert_transformed = True

    return module
