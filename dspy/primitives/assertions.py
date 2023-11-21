import inspect
from typing import Any, Callable
import dsp
import dspy
import logging
import uuid

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
    trace_template = """Failed Instruction: {} | Output: {}"""
    trace_passages = []

    for id, (ip, op, msg) in failure_traces.items():
        trace_passages.append(trace_template.format(msg, op))

    return trace_passages


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


def _wrap_forward_with_set_fields(predictor, default_args):
    original_forward = predictor.forward

    # Create a copy of default_args to use inside the closure
    closure_args = default_args.copy()

    def new_forward(*args, **kwargs):
        for arg, value in closure_args.items():
            kwargs.setdefault(arg, value)

        return original_forward(**kwargs)

    predictor.forward = new_forward


def _pprintdict(d):
    """Pretty print a dictionary"""
    import json
    from termcolor import colored

    formatted_dict = json.dumps(d, indent=4)
    colored_lines = [
        colored(line, "blue")
        if ":" not in line
        else ": ".join(
            colored(part.strip(), "blue" if i == 0 else "green")
            for i, part in enumerate(line.split(":", 1))
        )
        for line in formatted_dict.splitlines()
    ]
    return "\n".join(colored_lines) + "\n"


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

    def __init__(
        self, id: str, msg: str, target_module: Any = None, state: Any = None
    ) -> None:
        super().__init__(msg)
        self.id = id
        self.msg = msg
        self.target_module = target_module
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
                    id=self.id,
                    msg=self.msg,
                    target_module=self.target_module,
                    state=dsp.settings.trace,
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


def suggest_backtrack_handler(func, max_backtracks=2):
    """Handler for backtracking suggestion.

    Re-run the latest predictor up to `max_backtracks` times,
    with updated signature if a suggestion fails. updated signature adds a new
    input field to the signature, which is the feedback.
    """
    from ..predict.retry import Retry

    def wrapper(*args, **kwargs):
        backtrack_to, error_msg, result = None, None, None

        # Predictor -> List[feedback_msg]
        dspy.settings.predictor_feedbacks = {}

        # Predictor -> Predictor.forward
        dspy.settings.predictors_to_original_forward = {}

        for i in range(max_backtracks + 1):
            if i > 0 and backtrack_to is not None:
                # revert to original forward function if modified
                if backtrack_to in dspy.settings.predictors_to_original_forward:
                    backtrack_to.forward = dspy.settings.predictors_to_original_forward[
                        backtrack_to
                    ]

                # create a new retry module wrapping backtrack_to
                retry_module = Retry(backtrack_to)

                # save original forward function to revert back to
                dspy.settings.predictors_to_original_forward[
                    backtrack_to
                ] = backtrack_to.forward

                # generate values for new fields
                feedback_msg = _build_error_msg(
                    dspy.settings.predictor_feedbacks[backtrack_to]
                )

                # set values as default for the new fields
                _wrap_forward_with_set_fields(
                    retry_module,
                    default_args={
                        "feedback": feedback_msg,
                        "past_outputs": past_outputs,
                    },
                )

                # point backtrack_to.forward to retry_module.forward
                # TODO: verify and make sure is thread safe
                backtrack_to.forward = retry_module.forward.__get__(
                    retry_module, retry_module.__class__
                )

            # if last backtrack: ignore suggestion errors
            if i == max_backtracks:
                result = bypass_suggest_handler(func)(*args, **kwargs)
                break

            else:
                try:
                    result = func(*args, **kwargs)
                    break
                except DSPySuggestionError as e:
                    suggest_id, error_msg, suggest_target_module, error_state = (
                        e.id,
                        e.msg,
                        e.target_module,
                        e.state[-1],
                    )

                    if dsp.settings.trace:
                        if suggest_target_module:
                            for i in range(len(dsp.settings.trace) - 1, -1, -1):
                                trace_element = dsp.settings.trace[i]
                                mod = trace_element[0]
                                if mod.signature == suggest_target_module:
                                    backtrack_to = mod
                                    break
                        else:
                            backtrack_to = dsp.settings.trace[-1][0]

                        if backtrack_to is None:
                            logger.error("Specified module not found in trace")

                        # save unique feedback message for predictor
                        if (
                            error_msg
                            not in dspy.settings.predictor_feedbacks.setdefault(
                                backtrack_to, []
                            )
                        ):
                            dspy.settings.predictor_feedbacks[backtrack_to].append(
                                error_msg
                            )

                        output_fields = vars(error_state[0].signature.signature)
                        past_outputs = {}
                        for field_name, field_obj in output_fields.items():
                            if isinstance(field_obj, dspy.OutputField):
                                past_outputs[field_name] = getattr(
                                    error_state[2], field_name, None
                                )

                        # save latest failure trace for predictor per suggestion
                        error_ip = error_state[1]
                        error_op = error_state[2].__dict__["_store"]
                        error_op.pop("_assert_feedback", None)
                        error_op.pop("_assert_traces", None)

                    else:
                        logger.error(
                            f"UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )

        # cleanup: after all tries revert any leftover predictors to their originals
        if dspy.settings.predictors_to_original_forward:
            for (
                predictor,
                original_forward,
            ) in dspy.settings.predictors_to_original_forward.items():
                _revert_predictor_signature(predictor, "feedback")
                setattr(predictor, "forward", original_forward.__get__(predictor))

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
