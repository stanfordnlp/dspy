import asyncio
import contextvars
import threading
from concurrent.futures import Future
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome = Future()

    def run():
        try:
            outcome.set_result(asyncio.run(coro))
        except BaseException as exc:
            outcome.set_exception(exc)

    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(run,), daemon=True).start()
    return outcome.result()


def syncify(program: "Module", in_place: bool = True) -> "Module":
    """Convert an async DSPy module to a sync program.

    There are two modes of this function:

    - `in_place=True` (recommended): Modify the module in place. But this may not work if you already have a `forward`
        method which does different things from `aforward`.
    - `in_place=False`: Return a wrapper module. This changes the module's architecture, but it's more robust.

    Args:
        program: The async program to convert, must have an `aforward` method implemented.
        in_place: If True, modify the module in place. Otherwise, return a wrapper module.

    Returns:
        The sync program, which has a `forward` method that can be called from a synchronous context.
    """
    if in_place:

        def forward(self, *args, **kwargs):
            return run_async(self.aforward(*args, **kwargs))

        # Create the `forward` method in place.
        program.forward = MethodType(forward, program)
        return program
    else:
        from dspy.primitives.module import Module

        class SyncWrapper(Module):
            def __init__(self, program: "Module"):
                self.program = program

            def forward(self, *args, **kwargs):
                return run_async(self.program.aforward(*args, **kwargs))

        return SyncWrapper(program)
