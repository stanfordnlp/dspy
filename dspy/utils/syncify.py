import asyncio
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If we're in a running event loop (e.g., Jupyter), use asyncio.create_task and run until done
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)


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
