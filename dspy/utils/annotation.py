import inspect
import re
import types
from typing import Callable, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

@overload
def experimental(f: Callable[P, R], version: str | None = None) -> Callable[P, R]: ...

@overload
def experimental(f: None = None, version: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def experimental(
    f: Callable[P, R] | None = None,
    version: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator / decorator creator for marking APIs experimental in the docstring.

    Args:
        f: The function to be decorated.
        version: The version in which the API was introduced as experimental.
            The version is used to determine whether the API should be considered
            as stable or not when releasing a new version of DSPy.

    Returns:
        A decorator that adds a note to the docstring of the decorated API.
    """
    if f:
        return _experimental(f, version)
    else:
        def decorator(f: Callable[P, R]) -> Callable[P, R]:
            return _experimental(f, version)
        return decorator


def _experimental(api: Callable[P, R], version: str | None = None) -> Callable[P, R]:
    """Add experimental notice to the API's docstring."""
    if inspect.isclass(api):
        api_type = "class"
    elif inspect.isfunction(api):
        api_type = "function"
    elif isinstance(api, property):
        api_type = "property"
    elif isinstance(api, types.MethodType):
        api_type = "method"
    else:
        api_type = str(type(api))

    indent = _get_min_indent_of_docstring(api.__doc__) if api.__doc__ else ""

    version_text = f" (introduced in v{version})" if version else ""
    notice = (
        indent + f"Experimental: This {api_type} may change or "
        f"be removed in a future release without warning{version_text}."
    )

    if api_type == "property":
        api.__doc__ = api.__doc__ + "\n\n" + notice if api.__doc__ else notice
    else:
        if api.__doc__:
            api.__doc__ = notice + "\n\n" + api.__doc__
        else:
            api.__doc__ = notice
    return api


def _get_min_indent_of_docstring(docstring_str: str) -> str:
    """
    Get the minimum indentation string of a docstring, based on the assumption
    that the closing triple quote for multiline comments must be on a new line.
    Note that based on ruff rule D209, the closing triple quote for multiline
    comments must be on a new line.

    Args:
        docstring_str: string with docstring

    Returns:
        Whitespace corresponding to the indent of a docstring.
    """

    if not docstring_str or "\n" not in docstring_str:
        return ""

    match = re.match(r"^\s*", docstring_str.rsplit("\n", 1)[-1])
    return match.group() if match else ""
