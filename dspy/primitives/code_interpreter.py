"""
Abstract interpreter interface for code execution environments.

This module defines the CodeInterpreter protocol that allows RLM and other
code-executing modules to work with different interpreter implementations:
- PythonInterpreter: Local Deno/Pyodide WASM interpreter
- LocalInterpreter: Persistent local CPython worker
- MockInterpreter: Scriptable responses for testing

It also resolves which implementation a module gets: a module's own
``interpreter_factory`` argument, else ``dspy.settings.interpreter_factory``, else
``PythonInterpreter``. See :func:`resolve_interpreter_factory`.
"""

from typing import Any, Callable, Protocol, runtime_checkable

from dspy.dsp.utils.settings import settings
from dspy.utils.exceptions import DSPyError

# Types that can be used directly in Python function signatures for SUBMIT()
SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


class CodeInterpreterError(DSPyError, RuntimeError):
    """Base class for errors reported by a code interpreter.

    A bare instance indicates a failure that submitted code cannot repair, such
    as invalid host-side setup or a process/protocol failure. Recoverable
    submitted-code failures use :class:`CodeExecutionError`. Implementations
    should make process/protocol failures terminal for that interpreter session.
    """


class CodeExecutionError(CodeInterpreterError):
    """Recoverable error raised by code running in a healthy interpreter."""


class FinalOutput:
    """Returned by interpreter.execute() when SUBMIT() is called.

    This signals that the code execution loop should terminate and return
    the contained output to the caller.
    """

    def __init__(self, output: Any):
        self.output = output

    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


@runtime_checkable
class CodeInterpreter(Protocol):
    """Protocol for code execution environments (interpreters).

    Implementations must provide:
    - start(): Initialize the interpreter (optional, can be lazy)
    - execute(): Run code and return results
    - shutdown(): Clean up resources

    The interpreter maintains state across execute() calls within a session,
    allowing variables defined in one call to be used in subsequent calls.

    Lifecycle:
        1. Create instance (config only, no resources allocated)
        2. start() - Initialize interpreter (explicit) or let execute() do it (lazy)
        3. execute() - Run code (can be called many times)
        4. shutdown() - Release resources

    Example implementations:
        - PythonInterpreter: Deno/Pyodide WASM interpreter (local sandbox)
        - LocalInterpreter: Persistent local CPython worker (not a sandbox)
        - MockInterpreter: Scriptable responses for testing

    Pooling:
        For interpreter pooling, call start() to pre-warm instances, then
        distribute execute() calls across the pool.
    """

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        """Tools available for interpreter code to call.

        Tools are host-side functions that can be invoked from within the
        interpreter. Each tool accepts keyword arguments. Return values must
        satisfy the boundary supported by the interpreter; Flex tools must
        return JSON-compatible values.

        Implementations should accept tools via constructor and expose them
        through this property.
        """
        ...

    def start(self) -> None:
        """Initialize the interpreter and allocate resources.

        This method prepares the interpreter for code execution. It can be called
        explicitly to pre-warm the interpreter, or implementations may call it
        lazily on first execute().

        For pooling scenarios, call start() on multiple instances to have
        them ready before distributing work.

        Calling start() multiple times before shutdown should be safe (idempotent).
        If the underlying interpreter process exits, the session state is lost and
        the implementation should raise CodeInterpreterError instead of silently
        starting a new session.
        """
        ...

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code and return the result.

        Args:
            code: Python code to execute
            variables: Variables to inject into the namespace before execution.
                      These are available as top-level variables in the code.

        Returns:
            One of:
            - FinalOutput: If SUBMIT() was called in code
            - str: Captured stdout from print() statements
            - list: Multiple output lines
            - None: If no output was produced

        Raises:
            CodeExecutionError: On runtime errors in the submitted code or a called tool.
            CodeInterpreterError: If host-side setup or the interpreter process/protocol fails.
            SyntaxError: On invalid Python syntax

        Note:
            State persists across calls. Variables defined in one execute()
            call are available in subsequent calls until shutdown().

            If start() was not called, implementations should call it lazily.
        """
        ...

    def shutdown(self) -> None:
        """Release resources and terminate the interpreter session.

        After shutdown(), the interpreter should not be used again.
        A new instance should be created for a fresh session.
        """
        ...


def _validate_interpreter_factory(factory: Any, name: str = "interpreter_factory") -> None:
    """Validate the configured provider without invoking it."""
    if not isinstance(factory, type) and isinstance(factory, CodeInterpreter):
        raise TypeError(
            f"{name} received an object that already implements CodeInterpreter, so its ownership "
            "is ambiguous. Pass an existing interpreter as the first positional argument when calling the module. "
            "If this object also creates interpreters, pass a dedicated zero-argument creation callable instead."
        )
    if not callable(factory):
        raise TypeError(
            f"{name} must be a zero-argument callable that creates a CodeInterpreter, "
            f"not {type(factory).__name__}."
        )


def resolve_interpreter_factory(
    factory: Callable[[], CodeInterpreter] | None = None,
) -> Callable[[], CodeInterpreter]:
    """Return the factory that creates the next interpreter.

    The module's own factory wins, then ``dspy.settings.interpreter_factory``, then
    ``PythonInterpreter``. ``None`` and ``PythonInterpreter`` both mean "unset", since the
    code-executing modules carry ``PythonInterpreter`` as their literal default argument.

    Call this where you create an interpreter, not earlier, so that
    ``dspy.context(interpreter_factory=...)`` scopes the way callers expect.

    Raises:
        TypeError: If ``dspy.settings.interpreter_factory`` is not a zero-argument callable.
    """
    # Imported here because python_interpreter imports this module.
    from dspy.primitives.python_interpreter import PythonInterpreter

    if factory is not None and factory is not PythonInterpreter:
        return factory

    configured = settings.get("interpreter_factory")
    if configured is None:
        return PythonInterpreter

    _validate_interpreter_factory(configured, name="dspy.settings.interpreter_factory")
    return configured


def _create_interpreter(factory: Callable[[], CodeInterpreter] | None) -> CodeInterpreter:
    """Create an interpreter from ``factory``, or from the configured one, and validate it."""
    interpreter = resolve_interpreter_factory(factory)()
    if not isinstance(interpreter, CodeInterpreter):
        raise TypeError(
            "interpreter_factory must return a CodeInterpreter, "
            f"not {type(interpreter).__name__}."
        )
    return interpreter


def _validate_interpreter(interpreter: Any) -> None:
    """Validate a caller-owned interpreter."""
    if not isinstance(interpreter, CodeInterpreter):
        raise TypeError(f"interpreter must implement CodeInterpreter, not {type(interpreter).__name__}.")
