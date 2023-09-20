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
        self.msg = msg
        self.state = state


class Assert:
    """Compile time assertion."""

    def __init__(self, assert_fun, *args, **kwargs):
        self.assert_fun = assert_fun
        self.args = args
        self.kwargs = kwargs
        self.__call__()

    # assert fun should always return bool
    def __call__(self) -> bool:
        result = self.assert_fun(*self.args, **self.kwargs)
        if isinstance(result, bool):
            if result:
                return True
            else:
                raise DSPyAssertionError(
                    f"Assertion {self.assert_fun} failed", state=dsp.settings.trace
                )
        else:
            raise ValueError("Assertion function should always return [bool]")


# we could possibly to have runtime assertions as well
# class RuntimeAssert(Assert): ...


############################# ASSERTION AND BACKTRACKING POLICIES #############################

def assert_transform(backtrack=2):
    """Decorator that simply re-runs the function if assertion fails, 
    up to [backtrack] times."""

    def wrapper(func):
        def inner(*args, **kwargs):
            for i in range(backtrack):
                try:
                    return func(*args, **kwargs)
                except DSPyAssertionError as e:
                    print(f"{e.msg}")
                    if dsp.settings.trace:
                        print(dsp.settings.trace[-1])
                    else:
                        print(
                            "UNREACHABLE: No trace available, this should not happen. Is this run time?"
                        )
                else:
                    break

        return inner

    return wrapper


def assert_update_transform(backtrack=2):
    """Decorator that simply re-runs the function with updated temperature 
    if assertion fails, up to [backtrack] times."""

    def wrapper(func):
        def inner(*args, **kwargs):
            for i in range(backtrack):
                lm = dsp.settings.lm
                
                if i > 0 and backtrack_to is not None:
                    print(f"updating temperature to {0.7 + 0.001 * i}")
                    lm = lm.copy(temperature=0.7 + 0.001 * i)
                
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
                else:
                    break

        return inner

    return wrapper


def assert_latest_transform(backtrack=3):
    """Decorator that simply re-runs the function but updates the temperature
    only for the latest predictor in ther trace if assertion fails, up to [backtrack] times.
    
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

                    # udpate temperature via "config" attribute of the backtrack_to predictor

                    print(f"prev temp @ {id(backtrack_to)}: {backtrack_to.get_config().get('temperature', 'default')}")
                    backtrack_to.update_config(temperature=0.7 + 0.001 * i)
                    print(f"new temp @ {id(backtrack_to)}: {backtrack_to.get_config()['temperature']}")
                    
                    # @manish: works quite well w/ current example!!!
                    # TODO (@manish): check if this makes sense w/ more examples

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