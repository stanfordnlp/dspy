# Note(shangyin): What abstractions we want to make about assertions?
# things to consider: what we want to assert, and what would be the syntax?
# One possible starting point would be constraints on the output of a module
from typing import Any
import dspy


class DSPyAssertionError(AssertionError):
    """DSPy custom error message."""
    def __init__(self, msg: str, state: Any = None) -> None:
        super().__init__(msg)
        self.state = state


class Assert:
    """Compile time assertion."""
    def __init__(self, assert_fun, *args, **kwargs):
        self.assert_fun = assert_fun
        self.args = args
        self.kwargs = kwargs

    # assert fun should always return bool
    def __call__(self) -> bool:
        result = self.assert_fun(*self.args, **self.kwargs)
        if isinstance(result, bool):
            if result:
                return True
            else:
                raise DSPyAssertionError(f"Assertion {self.assert_fun} failed")
        else:
            raise ValueError("Assertion function should always return [bool]")


# we could possibly to have runtime assertions as well
# class RuntimeAssert(Assert): ...


def assert_transform(backtrack=2):
    """Decorator for transforming a function into a rewindable function."""
    def wrapper(func):
        def inner(*args, **kwargs):
            for i in range(backtrack):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    # Add metadata to state
                    pass

        return inner

    return wrapper
