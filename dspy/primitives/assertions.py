from typing import Any, Callable
import dsp
import dspy


class DSPyAssertionError(AssertionError):
    """Custom exception raised when a DSPy `Assert` fails.
    """

    def __init__(self, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.msg = msg
        self.state = state


class Assert:
    """DSPy Assertion"""

    def __init__(self, assert_fun: Callable, *args, **kwargs):
        self.assert_fun = assert_fun
        self.args = args

        if "msg" in kwargs:
            self.msg = kwargs["msg"]
            del kwargs["msg"]
        else:
            self.msg = ""

        self.kwargs = kwargs
        self.__call__()

    def __call__(self) -> bool:
        result = self.assert_fun(*self.args, **self.kwargs)
        if isinstance(result, bool):
            if result:
                return True
            else:
                raise DSPyAssertionError(msg=self.msg, state=dsp.settings.trace)
        else:
            raise ValueError("Assertion function should always return [bool]")


def assert_latest_feedback_transform(max_backtracks=2):
    """Decorator that defines the backtracking policy for assertions.
    
    current policy: re-run the latest predictor up to `max_backtracks` times,
    with updated signature if assertion fails. updated signature adds a new 
    input field to the signature, which is the feedback.
    """

    def wrapper(func):
        def inner(*args, **kwargs):
            backtrack_to = None
            error_msg = None
            result = None
            extended_predictors_to_original_forward = {}

            def _extend_predictor_signature(predictor, **kwargs):
                """Update the signature of a predictor instance with specified fields."""
                old_signature = predictor.extended_signature
                new_signature_kwargs = {
                    k: v
                    for (k, v) in old_signature.kwargs.items()
                    if isinstance(v, dspy.InputField)
                }
                new_signature_kwargs.update(kwargs)
                new_signature_kwargs.update(
                    {
                        k: v
                        for (k, v) in old_signature.kwargs.items()
                        if isinstance(v, (dsp.Type, dspy.OutputField))
                    }
                )

                new_signature = dsp.Template(
                    old_signature.instructions, **new_signature_kwargs
                )

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
                        prefix="Instruction:", desc="Some instructions you must satisfy"
                    )

                    # save the original forward function to revert back to
                    extended_predictors_to_original_forward[
                        backtrack_to
                    ] = backtrack_to.forward

                    # extend signature with feedback field
                    _extend_predictor_signature(backtrack_to, feedback=feedback)

                    # set feedback field as a default kwarg to the predictor's forward function
                    _wrap_forward_with_set_fields(
                        backtrack_to, default_args={"feedback": error_msg}
                    )

                try:
                    result = func(*args, **kwargs)
                    break
                except DSPyAssertionError as e:
                    print(f"AssertionError: {e.msg}")
                    error_msg = e.msg

                    if dsp.settings.trace:
                        backtrack_to = dsp.settings.trace[-1][0]
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
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

        return inner

    return wrapper
