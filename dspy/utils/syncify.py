import asyncio
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def run_async(coro):
    """Run an async coroutine from a synchronous context.

    Inside a running event loop (e.g. Jupyter, or a server handler) a sync
    call cannot be served without nested-loop hacks: `nest_asyncio` patches
    the event loop process-wide as an import side effect and does not support
    alternative loops such as uvloop. Follow the same fail-fast contract as
    `dspy.Tool` and direct the caller to the native async path instead.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Run the coroutine outside of "except" block to avoid propagation
        loop = None

    if loop is None:
        return asyncio.run(coro)

    coro.close()
    raise ValueError(
        "You are calling a syncified program from within a running event loop, which cannot be "
        "converted to a sync call. Please use the module's native async path instead, e.g. "
        "`await program.aforward(...)` (or `await program.acall(...)`)."
    )


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
