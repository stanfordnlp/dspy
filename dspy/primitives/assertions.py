# Note(shangyin): What abstractions we want to make about assertions?
# things to consider: what we want to assert, and what would be the syntax?
# One possible starting point would be constraints on the output of a module
from typing import Any
import dsp
import dspy


class DSPyAssertionError(AssertionError):
    """DSPy custom error message."""

    def __init__(self, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.msg = f"Assertion Error: {msg}"
        self.state = state


class Assert:
    """Compile time assertion."""

    def __init__(self, assert_fun, *args, **kwargs):
        self.assert_fun = assert_fun
        self.args = args

        if "msg" in kwargs:
            self.msg = kwargs["msg"]
            del kwargs["msg"]
        else:
            self.msg = ""

        self.kwargs = kwargs
        self.__call__()

    # assert fun should always return bool
    def __call__(self) -> bool:
        result = self.assert_fun(*self.args, **self.kwargs)
        if isinstance(result, bool):
            if result:
                return True
            else:
                raise DSPyAssertionError(msg=self.msg, state=dsp.settings.trace)
        else:
            raise ValueError("Assertion function should always return [bool]")


# we could possibly to have runtime assertions as well
# class RuntimeAssert(Assert): ...


############################# ASSERTION AND BACKTRACKING POLICIES #############################


def assert_transform(backtrack=2):
    """Decorator that simply re-runs the function if assertion fails,
    up to `backtrack` times."""

    def wrapper(func):
        def inner(*args, **kwargs):
            for i in range(backtrack):
                try:
                    return func(*args, **kwargs)
                except DSPyAssertionError as e:
                    print(f"{e.msg}")
                    if dsp.settings.trace:
                        pass
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )

        return inner

    return wrapper


def assert_update_transform(backtrack=2):
    """Decorator that simply re-runs the function with updated temperature
    if assertion fails, up to `backtrack` times."""

    def wrapper(func):
        def inner(*args, **kwargs):
            for i in range(backtrack):
                lm = dsp.settings.lm

                if i > 0 and backtrack_to is not None:
                    # print(f"updating temperature to {0.71 + 0.002 * i}")
                    lm = lm.copy(temperature=0.71 + 0.002 * i)

                new_settings = dict(lm=lm) if i > 0 else {}

                try:
                    with dsp.settings.context(**new_settings):
                        return func(*args, **kwargs)
                except DSPyAssertionError as e:
                    print(f"{e.msg}")

                    if dsp.settings.trace:
                        backtrack_to = id(dsp.settings.trace[-1][0])
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )

        return inner

    return wrapper


def assert_latest_transform(backtrack=2):
    """Decorator that simply re-runs the function but updates the temperature
    only for the latest predictor in the trace if assertion fails, up to `backtrack` times.

    # NOTE (@manish): the previous assert_update_transform applies temperature change to all predictors in the pipeline.
    # Here, we udpate/pass the new temperature to the *backtrack_to* predictor *only* --> cause a cache miss --> re-run
    # For other predictors, settings remain the same --> cache hit --> no re-run
    """

    def wrapper(func):
        def inner(*args, **kwargs):
            backtrack_to = None

            for i in range(backtrack):
                if i > 0 and backtrack_to is not None:
                    print(f"rewinding to {id(backtrack_to)}")

                    # udpate temperature via "config" attribute of the `backtrack_to` predictor
                    predictor_id = id(backtrack_to)
                    print(
                        f"prev temp @ {predictor_id}: {backtrack_to.get_config().get('temperature', 'default')}"
                    )
                    backtrack_to.update_config(temperature=0.7 + 0.001 * i)
                    print(
                        f"new temp @ {predictor_id}: {backtrack_to.get_config()['temperature']}"
                    )
                try:
                    return func(*args, **kwargs)
                except DSPyAssertionError as e:
                    print(f"{e.msg}")

                    if dsp.settings.trace:
                        backtrack_to = dsp.settings.trace[-1][0]
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )

        return inner

    return wrapper


def assert_latest_feedback_transform(backtrack=2):
    """ "Decorator that simply re-runs the function but updates the signature of
    the latest predictor in the trace if assertion fails, up to `backtrack` times.

    The updated signature passes a new input to the predictor, which is the feedback
    from the failed assertion. This feedback will be used by the predictor to self-repair.
    """

    def wrapper(func):
        def inner(*args, **kwargs):
            backtrack_to = None
            error_msg = None
            extended_backtrack_to = None

            def _extend_predictor_signature(predictor, **kwargs):
                """Update the signature of a predictor instance with a new field."""
                old_signature = predictor.extended_signature
                *keys, last_key = old_signature.kwargs.keys()

                extended_kwargs = {key: kwargs[key] for key in kwargs}
                extended_kwargs.update({key: old_signature.kwargs[key] for key in keys})
                extended_kwargs.update({last_key: old_signature.kwargs[last_key]})

                new_signature = dsp.Template(old_signature.instructions, **extended_kwargs)
                predictor.extended_signature = new_signature

            def _remove_predictor_signature(predictor, *args):
                """Remove the signature of a predictor."""
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

            for i in range(backtrack):
                if i > 0 and backtrack_to is not None:
                    print(f"rewinding to {id(backtrack_to)}")

                    # udpate the signature of the `backtrack_to` predictor with a feedback field
                    feedback = dspy.InputField(
                        prefix="Note:", desc="common mistakes to avoid"
                    )
                    extended_backtrack_to = backtrack_to
                    original_forward = extended_backtrack_to.forward
                    _extend_predictor_signature(backtrack_to, feedback=feedback)

                    # set the feedback field as a default kwarg to the predictor's forward function
                    _wrap_forward_with_set_fields(
                        backtrack_to, default_args={"feedback": error_msg}
                    )

                try:
                    return func(*args, **kwargs)
                except DSPyAssertionError as e:
                    print(f"{e.msg}")
                    error_msg = e.msg

                    if dsp.settings.trace:
                        backtrack_to = dsp.settings.trace[-1][0]
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )
                finally:
                    if extended_backtrack_to:
                        _remove_predictor_signature(extended_backtrack_to, "feedback")
                        extended_backtrack_to.forward = original_forward

        return inner

    return wrapper
