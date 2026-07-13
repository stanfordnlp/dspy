"""
Abstract interpreter interface for code execution environments.

This module defines the CodeInterpreter protocol that allows RLM and other
code-executing modules to work with different interpreter implementations:
- PythonInterpreter: Local Deno/Pyodide WASM interpreter
- MockInterpreter: Scriptable responses for testing
"""

from typing import Any, Callable, Protocol, runtime_checkable

# Types that can be used directly in Python function signatures for SUBMIT()
SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


class CodeInterpreterError(RuntimeError):
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
        - LocalInterpreter: Deno/Pyodide WASM interpreter (local)
        - MockInterpreter: Scriptable responses for testing

    Pooling:
        For interpreter pooling, call start() to pre-warm instances, then
        distribute execute() calls across the pool.
    """

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        """Tools available for interpreter code to call.

        Tools are host-side functions that can be invoked from within the
        interpreter. Each tool accepts keyword arguments and returns a string.

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


def _validate_interpreter_factory(factory: Any) -> None:
    """Validate the configured provider without invoking it."""
    if not isinstance(factory, type) and isinstance(factory, CodeInterpreter):
        raise TypeError(
            "interpreter_factory must create a new CodeInterpreter, not be an interpreter instance. "
            "Pass an existing instance as the first positional argument to forward(...) instead."
        )
    if not callable(factory):
        raise TypeError(
            "interpreter_factory must be a zero-argument callable that creates a CodeInterpreter, "
            f"not {type(factory).__name__}."
        )


def _create_interpreter(factory: Callable[[], CodeInterpreter]) -> CodeInterpreter:
    """Create an interpreter and validate the factory's return value."""
    interpreter = factory()
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
