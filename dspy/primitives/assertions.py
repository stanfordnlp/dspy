import inspect
from typing import Any, Callable
import dsp
import dspy

import logging
import uuid
from ..predict.retry import Retry

#################### Assertion Helpers ####################


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("assertion.log")
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    return logger


logger = setup_logger()


def _build_error_msg(feedback_msgs):
    """Build an error message from a list of feedback messages."""
    return "\n".join([msg for msg in feedback_msgs])


def _build_trace_passages(failure_traces):
    """Build a list of trace passages from a list of failure traces."""

    # print("Traces:")
    # for id, (ip, op, msg) in failure_traces.items():
    #     print(id, op, msg)
    # print()

    # TODO complete this
    return failure_traces


def _extend_predictor_signature(predictor, **kwargs):
    """Update the signature of a predictor instance with specified fields."""
    old_signature = predictor.extended_signature
    old_keys = list(old_signature.kwargs.keys())

    # include other input fields after question
    position = old_keys.index("question") + 1
    for key in reversed(kwargs):
        old_keys.insert(position, key)

    extended_kwargs = {
        key: kwargs.get(key, old_signature.kwargs.get(key)) for key in old_keys
    }

    new_signature = dsp.Template(old_signature.instructions, **extended_kwargs)
    predictor.extended_signature = new_signature


def _revert_predictor_signature(predictor, *args):
    """Revert the signature of a predictor by removing specified fields."""
    old_signature_kwargs = predictor.extended_signature.kwargs
    for key in args:
        old_signature_kwargs.pop(key, None)
    new_signature = dsp.Template(
        predictor.extended_signature.instructions, **old_signature_kwargs
    )
    predictor.extended_signature = new_signature


#################### Assertion Exceptions ####################


class DSPyAssertionError(AssertionError):
    """Custom exception raised when a DSPy `Assert` fails."""

    def __init__(self, id: str, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.id = id
        self.msg = msg
        self.state = state


class DSPySuggestionError(AssertionError):
    """Custom exception raised when a DSPy `Suggest` fails."""

    def __init__(self, id: str, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.id = id
        self.msg = msg
        self.state = state


#################### Assertion Primitives ####################


class Constraint:
    def __init__(self, result: bool, msg: str = "", target_module=None):
        self.id = str(uuid.uuid4())
        self.result = result
        self.msg = msg
        self.target_module = target_module

        self.__call__()


class Assert(Constraint):
    """DSPy Assertion"""

    def __call__(self) -> bool:
        if isinstance(self.result, bool):
            if self.result:
                return True
            elif dspy.settings.bypass_assert:
                logger.error(f"AssertionError: {self.msg}")
                return True
            else:
                logger.error(f"AssertionError: {self.msg}")
                raise DSPyAssertionError(
                    id=self.id, msg=self.msg, state=dsp.settings.trace
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
                logger.error(f"SuggestionFailed: {self.msg}")
                return True
            else:
                logger.error(f"SuggestionFailed: {self.msg}")
                raise DSPySuggestionError(
                    id=self.id, msg=self.msg, state=dsp.settings.trace
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
        except DSPyAssertionError as e:
            return None

    return wrapper


def suggest_backtrack_handler(func, max_backtracks=10, **handler_args):
    """Handler for backtracking suggestion.

    Re-run the latest predictor up to `max_backtracks` times,
    with updated signature if a suggestion fails. updated signature adds a new
    input field to the signature, which is the feedback.
    """

    def wrapper(*args, **kwargs):
        target_module = handler_args.get("target_module", None)
        backtrack_to = None
        error_msg = None
        result = None
        extended_predictors_to_original_forward = {}
        # predictor_feedback: Predictor -> List[feedback_msg]
        predictor_feedbacks = {}

        # failure_traces: Predictor -> Dict[assertion, (input, op, msg)]
        failure_traces = {}

        for i in range(max_backtracks + 1):
            if i > 0 and backtrack_to is not None:
<<<<<<< HEAD
                retry_module = Retry(backtrack_to)

                # set feedback field for predictor's forward function
=======
                # create a new retry module that wraps backtrack_to
                retry_module = dspy.Retry(backtrack_to)

                # save the original forward function to revert back to
                predictors_to_original_forward[backtrack_to] = backtrack_to.forward

                # set the new fields
>>>>>>> 759d62ae604e5924cd2df6ffeb94a0d692939488
                feedback_msg = _build_error_msg(predictor_feedbacks[backtrack_to])

                # set traces field for predictor's forward function
                trace_passages = _build_trace_passages(failure_traces[backtrack_to])
                kwargs["feedback"] = feedback_msg
                kwargs["traces"] = trace_passages

<<<<<<< HEAD
                kwargs['feedback'] = feedback_msg
                kwargs['traces'] = trace_passages
=======
                # FIXME: need to replace backtrack_to.forward
                # in the user's program with retry_module.forward
                # NOTE: this replacement needs to be thread safe if possible
>>>>>>> 759d62ae604e5924cd2df6ffeb94a0d692939488

            # if last backtrack: ignore suggestion errors
            if i == max_backtracks:
                result = bypass_suggest_handler(func)(*args, **kwargs)
                break

            else:
                try:
                    result = retry_module(*args, **kwargs)
                    break
                except DSPySuggestionError as e:
                    suggest_id, error_msg = e.id, e.msg
                    error_ip = e.state[-1][1]
                    error_op = e.state[-1][2].__dict__["_store"]

                    if dsp.settings.trace:
                        if target_module:
                            for i in range(len(dsp.settings.trace) - 1, -1, -1):
                                trace_element = dsp.settings.trace[i]
                                mod = trace_element[0]
                                if mod.signature == target_module:
                                    backtrack_to = mod
                                    break
                        else:
                            backtrack_to = dsp.settings.trace[-1][0]

                        if backtrack_to is None:
                            logger.error("Specified module not found in trace")

                        # save unique feedback message for predictor
                        if error_msg not in predictor_feedbacks.setdefault(
                            backtrack_to, []
                        ):
                            predictor_feedbacks[backtrack_to].append(error_msg)

                        # save latest failure trace for predictor per suggestion
                        failure_traces.setdefault(backtrack_to, {})[suggest_id] = (
                            error_ip,
                            error_op,
                            error_msg,
                        )

                    else:
                        logger.error(
                            f"UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )

        # revert any extended predictors to their originals
        if extended_predictors_to_original_forward:
            for (
                predictor,
                original_forward,
            ) in extended_predictors_to_original_forward.items():
                _revert_predictor_signature(predictor, "feedback")
                predictor.forward = original_forward

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


default_assertion_handler = suggest_backtrack_handler


def assert_transform_module(
    module, assertion_handler=default_assertion_handler, **handler_args
):
    """
    Transform a module to handle assertions.
    """
    if not getattr(module, "forward", False):
        raise ValueError(
            "Module must have a forward method to have assertions handled."
        )
    if getattr(module, "_forward", False):
        logger.info(
            f"Module {module.__class__.__name__} already has a _forward method. Skipping..."
        )
        pass  # TODO warning: might be overwriting a previous _forward method

    module._forward = module.forward
    module.forward = handle_assert_forward(assertion_handler, **handler_args).__get__(
        module
    )

    return module