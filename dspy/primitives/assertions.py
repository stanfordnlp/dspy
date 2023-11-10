import inspect
from typing import Any, Callable
import dsp
import dspy

import logging


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


class DSPyAssertionError(AssertionError):
    """Custom exception raised when a DSPy `Assert` fails."""

    def __init__(self, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.msg = msg
        self.state = state


class DSPySuggestionError(AssertionError):
    """Custom exception raised when a DSPy `Suggest` fails."""

    def __init__(self, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.msg = msg
        self.state = state


class Constraint:

    def __init__(self, result: bool, msg: str = ""):
        self.result = result
        self.msg = msg

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
                raise DSPyAssertionError(msg=self.msg, state=dsp.settings.trace)
        else:
            raise ValueError("Assertion function should always return [bool]")


class Suggest(Constraint):
    """DSPy Suggestion"""

    def __call__(self) -> Any:
        if isinstance(self.result, bool):
            if self.result:
                return True
            elif dspy.settings.bypass_assert:
                logger.error(f"SuggestionFailed: {self.msg}")
                return True
            else:
                logger.error(f"SuggestionFailed: {self.msg}")
                raise DSPyAssertionError(msg=self.msg, state=dsp.settings.trace)


def noop_handler(func):
    """ "Handler to bypass assertions."""

    def wrapper(*args, **kwargs):
        with dspy.settings.context(bypass_assert=True):
            return func(*args, **kwargs)

    return wrapper


def assert_backtrack_handler(func, max_backtracks=2):
    """Handler for backtracking assertions.

    Re-run the latest predictor up to `max_backtracks` times,
    with updated signature if assertion fails. updated signature adds a new
    input field to the signature, which is the feedback.
    """

    def wrapper(*args, **kwargs):
        backtrack_to = None
        error_msg = None
        result = None
        extended_predictors_to_original_forward = {}
        # predictor_feedback: Predictor -> List[feedback_msg]
        # this will contain all assertion failure msgs for each predictor
        # including msgs for the same assertion failure
        predictor_feedbacks = {}

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

        def _build_error_msg(feedback_msgs):
            """Build an error message from a list of feedback messages."""
            return "\n".join([msg for msg in feedback_msgs])

        def _revert_predictor_signature(predictor, *args):
            """Revert the signature of a predictor by removing specified fields."""
            old_signature_kwargs = predictor.extended_signature.kwargs
            for key in args:
                old_signature_kwargs.pop(key, None)
            new_signature = dsp.Template(
                predictor.extended_signature.instructions, **old_signature_kwargs
            )
            predictor.extended_signature = new_signature

        def _wrap_forward_with_set_fields(predictor, default_args):
            """Wrap the forward method of a predictor instance to enforce default arguments."""
            original_forward = predictor.forward

            def new_forward(**kwargs):
                for arg, value in default_args.items():
                    kwargs.setdefault(arg, value)

                return original_forward(**kwargs)

            predictor.forward = new_forward

        for i in range(max_backtracks + 1):
            if i > 0 and backtrack_to is not None:
                # create a new feedback field
                feedback = dspy.InputField(
                    prefix="Instruction:",
                    desc="Some instructions you must satisfy",
                    format=str,
                )

                # save the original forward function to revert back to
                extended_predictors_to_original_forward[
                    backtrack_to
                ] = backtrack_to.forward

                # extend signature with feedback field
                _extend_predictor_signature(backtrack_to, feedback=feedback)

                # set feedback field as a default kwarg to the predictor's forward function
                feedback_msg = _build_error_msg(predictor_feedbacks[backtrack_to])
                _wrap_forward_with_set_fields(
                    backtrack_to, default_args={"feedback": feedback_msg}
                )

            # if user set bypass_assert: ignore assertion errors
            if dsp.settings.bypass_assert:
                result = func(*args, **kwargs)
                break

            # if last backtrack: ignore assertion errors
            elif i == max_backtracks and dspy.settings.bypass_assert is False:
                dspy.settings.configure(bypass_assert=True)
                result = func(*args, **kwargs)
                dspy.settings.configure(bypass_assert=False)
                break

            else:
                try:
                    result = func(*args, **kwargs)
                    break
                except DSPyAssertionError as e:
                    error_msg = e.msg

                    if dsp.settings.trace:
                        backtrack_to = dsp.settings.trace[-1][0]
                        if error_msg not in predictor_feedbacks.setdefault(
                            backtrack_to, []
                        ):
                            predictor_feedbacks[backtrack_to].append(error_msg)
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


def handle_assert_forward(assertion_handler):
    def forward(self, *args, **kwargs):
        args_to_vals = inspect.getcallargs(self._forward, *args, **kwargs)

        # if user has specified a bypass_assert flag, set it
        if "bypass_assert" in args_to_vals:
            dspy.settings.configure(bypass_assert=args_to_vals["bypass_assert"])

        wrapped_forward = assertion_handler(self._forward)
        return wrapped_forward(*args, **kwargs)

    return forward


default_assertion_handler = assert_backtrack_handler


def assert_transform_module(module, assertion_handler=default_assertion_handler):
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
    module.forward = handle_assert_forward(assertion_handler).__get__(module)

    return module
