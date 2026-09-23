"""Release stream sources without replacing an active failure or cancellation."""

import inspect
from contextlib import asynccontextmanager, contextmanager

from dspy.utils.lazy_import import require

anyio = require("anyio")


def _secondary(primary, cleanup):
    # Keep diagnostics on the original error across Python 3.10-3.14. Do not
    # interpolate exception messages: transport errors may contain credentials.
    try:
        primary.cleanup_errors = (*getattr(primary, "cleanup_errors", ()), cleanup)
        if hasattr(primary, "add_note"):
            primary.add_note(f"Stream cleanup also failed ({type(cleanup).__name__}); see cleanup_errors.")
    except Exception:
        # An exception type may prohibit extra attributes. Preserving the
        # original failure takes precedence over attaching optional diagnostics.
        pass


@contextmanager
def closing_stream(source):
    primary = None
    try:
        yield source
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close = getattr(source, "close", None)
            if close is not None:
                close()
        except Exception as cleanup:
            if primary is None or isinstance(primary, GeneratorExit):
                # Let the outer closer attach this to its actual failure.
                raise
            _secondary(primary, cleanup)


@asynccontextmanager
async def aclosing_stream(source):
    primary = None
    try:
        yield source
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close = getattr(source, "aclose", None) or getattr(source, "close", None)
            if close is not None:
                with anyio.CancelScope(shield=True):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
        except Exception as cleanup:
            if primary is None or isinstance(primary, GeneratorExit):
                raise
            _secondary(primary, cleanup)
