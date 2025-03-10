import functools
import sys
from importlib import import_module
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
)
from warnings import warn

from asyncer._compat import run_sync

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

import anyio
import anyio.from_thread
import anyio.to_thread
import sniffio
from anyio._core._eventloop import threadlocals
from anyio.abc import TaskGroup as _TaskGroup


# This was obtained with: from anyio._core._eventloop import get_asynclib
# Removed in https://github.com/agronholm/anyio/pull/429
# Released in AnyIO 4.x.x
# The new function is anyio._core._eventloop.get_async_backend but that returns a
# class, not a module to extract the TaskGroup class from.
def get_asynclib(asynclib_name: Union[str, None] = None) -> Any:
    if asynclib_name is None:
        asynclib_name = sniffio.current_async_library()

    modulename = "anyio._backends._" + asynclib_name
    try:
        return sys.modules[modulename]
    except KeyError:  # pragma: no cover
        return import_module(modulename)


T_Retval = TypeVar("T_Retval")
T_ParamSpec = ParamSpec("T_ParamSpec")
T = TypeVar("T")


class PendingType:
    def __repr__(self) -> str:
        return "AsyncerPending"


Pending = PendingType()


class PendingValueException(Exception):
    pass


class SoonValue(Generic[T]):
    def __init__(self) -> None:
        self._stored_value: Union[T, PendingType] = Pending

    @property
    def value(self) -> T:
        if isinstance(self._stored_value, PendingType):
            raise PendingValueException(
                "The return value of this task is still pending. Maybe you forgot to "
                "access it after the async with asyncer.create_task_group() block. "
                "If you need to access values of async tasks inside the same task "
                "group, you probably need a different approach, for example with "
                "AnyIO Streams."
            )
        return self._stored_value

    @property
    def ready(self) -> bool:
        return not isinstance(self._stored_value, PendingType)


class TaskGroup(_TaskGroup):
    def soonify(
        self, async_function: Callable[T_ParamSpec, Awaitable[T]], name: object = None
    ) -> Callable[T_ParamSpec, SoonValue[T]]:
        """
        Create and return a function that when called will start a new task in this
        task group.

        Internally it uses the same `task_group.start_soon()` method. But
        `task_group.soonify()` supports keyword arguments additional to positional
        arguments and it adds better support for autocompletion and inline errors
        for the arguments of the function called.

        Use it like this:

        ```Python
        async with asyncer.create_task_group() as task_group:
            async def do_work(arg1, arg2, kwarg1="", kwarg2="") -> str:
                # Do work
                return "Some result value"

            result = task_group.soonify(do_work)("spam", "ham", kwarg1="a", kwarg2="b")

        print(result.value)
        ```

        The return value from that function (`result` in the example) is an object
        `SoonValue`.

        This `SoonValue` object has an attribute `soon_value.value` that will hold the
        return value of the original `async_function` *after* the `async with` block.

        If you try to access the `soon_value.value` inside the `async with` block,
        before it has the actual return value, it will raise a an exception
        `asyncer.PendingValueException`.

        If you think you need to access the return values inside the `async with` block,
        there's a high chance that you really need a different approach, for example
        using an AnyIO Stream.

        But either way, if you have checkpoints inside the `async with` block (you have
        some `await` there), one or more of the `SoonValue` objects you might have
        could end up having the result value ready before ending the `async with` block.
        You can check that with `soon_value.pending`. For example:

        ```Python
        async def do_work(name: str) -> str:
            return f"Hello {name}"

        async with asyncer.create_task_group() as task_group:
            result1 = task_group.soonify(do_work)(name="task 1")
            result2 = task_group.soonify(do_work)(name="task 2")
            await anyio.sleep(0)
            if not result1.pending:
                print(result1.value)
            if not result2.pending:
                print(result2.value)
        ```


        ## Arguments

        `async_function`: an async function to call soon
        `name`: name of the task, for the purposes of introspection and debugging

        ## Return

        A function that takes positional and keyword arguments and when called
        uses `task_group.start_soon()` to start the task in this task group.

        That function returns a `SoonValue` object holding the return value of the
        original function in `soon_value.value`.
        """

        @functools.wraps(async_function)
        def wrapper(
            *args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs
        ) -> SoonValue[T]:
            partial_f = functools.partial(async_function, *args, **kwargs)
            soon_value: SoonValue[T] = SoonValue()

            @functools.wraps(partial_f)
            async def value_wrapper(*args: Any) -> None:
                value = await partial_f()
                soon_value._stored_value = value

            self.start_soon(value_wrapper, name=name)
            return soon_value

        return wrapper

    # This is only for the return type annotation, but it won't really be called
    async def __aenter__(self) -> "TaskGroup":  # pragma: nocover
        """Enter the task group context and allow starting new tasks."""
        return await super().__aenter__()  # type: ignore


def create_task_group() -> "TaskGroup":
    """
    Create a task group used to start multiple concurrent tasks with async functions.

    `asyncer.create_task_group()` is different from `anyio.create_task_group()` in that
    it creates an extended `TaskGroup` object that includes the `task_group.soonify()`
    method.
    """

    LibTaskGroup = get_asynclib().TaskGroup

    class ExtendedTaskGroup(LibTaskGroup, TaskGroup):  # type: ignore
        pass

    return ExtendedTaskGroup()


def runnify(
    async_function: Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]],
    backend: str = "asyncio",
    backend_options: Optional[Dict[str, Any]] = None,
) -> Callable[T_ParamSpec, T_Retval]:
    """
    Take an async function and create a regular (blocking) function that receives the
    same keyword and positional arguments for the original async function, and that when
    called will create an event loop and use it to run the original `async_function`
    with those arguments.

    That function returns the return value from the original `async_function`.

    The current thread must not be already running an event loop.

    This calls `anyio.run()` underneath.

    Use it like this:

    ```Python
    async def program(name: str) -> str:
        return f"Hello {name}"


    result = asyncer.runnify(program)(name="World")
    print(result)
    ```

    ## Arguments

    `async_function`: an async function to call
    `backend`: name of the asynchronous event loop implementation - currently either
        `asyncio` or `trio`
    `backend_options` keyword arguments to call the backend `run()` implementation with

    ## Return

    The return value of the async function

    ## Raises

    `RuntimeError`: if an asynchronous event loop is already running in this thread
    `LookupError`: if the named backend is not found
    """

    @functools.wraps(async_function)
    def wrapper(*args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs) -> T_Retval:
        partial_f = functools.partial(async_function, *args, **kwargs)

        return anyio.run(partial_f, backend=backend, backend_options=backend_options)

    return wrapper


def syncify(
    async_function: Callable[T_ParamSpec, Coroutine[Any, Any, T_Retval]],
    raise_sync_error: bool = True,
) -> Callable[T_ParamSpec, T_Retval]:
    """
    Take an async function and create a regular one that receives the same keyword and
    positional arguments, and that when called, calls the original async function in
    the main async loop from the worker thread using `anyio.to_thread.run()`.

    By default this is expected to be used from a worker thread. For example inside
    some function passed to `asyncify()`.

    But if you set `raise_sync_error` to `False`, you can also use this function
    in a non-async context: without an async event loop. For example, from a
    blocking/regular function called at the top level of a Python file. In that case,
    if it is not being called from inside a worker thread started from an async context
    (e.g. this is not called from a function that was called with `asyncify()`) it will
    run `async_function` in a new async event loop with `anyio.run()`.

    This functionality with `raise_sync_error` is there only to allow using
    `syncify()` in codebases that are used by async code in some cases and by blocking
    code in others. For example, during migrations from blocking code to async code.

    Internally, `asyncer.syncify()` uses the same `anyio.from_thread.run()`, but it
    supports keyword arguments additional to positional arguments and it adds better
    support for tooling (e.g. editor autocompletion and inline errors) for the
    arguments and return value of the function.

    Use it like this:

    ```Python
    async def do_work(arg1, arg2, kwarg1="", kwarg2=""):
        # Do work

    result = from_thread.syncify(do_work)("spam", "ham", kwarg1="a", kwarg2="b")
    ```

    ## Arguments

    `async_function`: an async function to be called in the main thread, in the async
        event loop
    `raise_sync_error`: If set to `False`, when used in a non-async context (without
        an async event loop), it will run `async_function` in a new async event loop,
        instead of raising an exception.

    ## Return

    A regular blocking function that takes the same positional and keyword arguments
        as the original async one, that when called runs the same original function in
        the main async loop when called from a worker thread and returns the result.
    """

    @functools.wraps(async_function)
    def wrapper(*args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs) -> T_Retval:
        current_async_module = (
            getattr(threadlocals, "current_async_backend", None)
            or
            # TODO: remove when deprecating AnyIO 3.x
            getattr(threadlocals, "current_async_module", None)
        )
        partial_f = functools.partial(async_function, *args, **kwargs)
        if current_async_module is None and raise_sync_error is False:
            return anyio.run(partial_f)
        return anyio.from_thread.run(partial_f)

    return wrapper


def asyncify(
    function: Callable[T_ParamSpec, T_Retval],
    *,
    abandon_on_cancel: bool = False,
    cancellable: Union[bool, None] = None,
    limiter: Optional[anyio.CapacityLimiter] = None,
) -> Callable[T_ParamSpec, Awaitable[T_Retval]]:
    """
    Take a blocking function and create an async one that receives the same
    positional and keyword arguments, and that when called, calls the original function
    in a worker thread using `anyio.to_thread.run_sync()`. Internally,
    `asyncer.asyncify()` uses the same `anyio.to_thread.run_sync()`, but it supports
    keyword arguments additional to positional arguments and it adds better support for
    autocompletion and inline errors for the arguments of the function called and the
    return value.

    If the `cancellable` option is enabled and the task waiting for its completion is
    cancelled, the thread will still run its course but its return value (or any raised
    exception) will be ignored.

    Use it like this:

    ```Python
    def do_work(arg1, arg2, kwarg1="", kwarg2="") -> str:
            # Do work
            return "Some result"

    result = await to_thread.asyncify(do_work)("spam", "ham", kwarg1="a", kwarg2="b")
    print(result)
    ```

    ## Arguments

    `function`: a blocking regular callable (e.g. a function)
    `cancellable`: `True` to allow cancellation of the operation
    `limiter`: capacity limiter to use to limit the total amount of threads running
        (if omitted, the default limiter is used)

    ## Return

    An async function that takes the same positional and keyword arguments as the
    original one, that when called runs the same original function in a thread worker
    and returns the result.
    """
    if cancellable is not None:
        abandon_on_cancel = cancellable
        warn(
            "The `cancellable=` keyword argument to `asyncer.asyncify()` is "
            "deprecated since Asyncer 0.0.8, following AnyIO 4.1.0. "
            "Use `abandon_on_cancel=` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @functools.wraps(function)
    async def wrapper(
        *args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs
    ) -> T_Retval:
        partial_f = functools.partial(function, *args, **kwargs)

        return await run_sync(
            partial_f, abandon_on_cancel=abandon_on_cancel, limiter=limiter
        )

    return wrapper
