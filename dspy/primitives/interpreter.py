"""
Abstract interpreter interface for code execution environments.

This module defines the Interpreter protocol that allows RLM and other
code-executing modules to work with different interpreter implementations:
- LocalInterpreter: Local Deno/Pyodide WASM interpreter
- MockInterpreter: Scriptable responses for testing
"""

from typing import Any, Callable, Protocol, runtime_checkable

# Types that can be used directly in Python function signatures for FINAL()
SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


class InterpreterError(RuntimeError):
    """Error raised during code execution in an interpreter.

    This covers runtime errors, undefined variables, tool call failures, etc.
    SyntaxError is raised separately for invalid Python syntax.
    """


class FinalAnswerResult:
    """Returned by interpreter.execute() when FINAL() or FINAL_VAR() is called.

    This signals that the code execution loop should terminate and return
    the contained answer to the caller.
    """

    def __init__(self, answer: Any):
        self.answer = answer

    def __repr__(self) -> str:
        return f"FinalAnswerResult({self.answer!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalAnswerResult):
            return NotImplemented
        return self.answer == other.answer


@runtime_checkable
class Interpreter(Protocol):
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

        Calling start() multiple times should be safe (idempotent).
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
            - FinalAnswerResult: If FINAL() or FINAL_VAR() was called in code
            - str: Captured stdout from print() statements
            - list: Multiple output lines
            - None: If no output was produced

        Raises:
            InterpreterError: On runtime errors (undefined vars, tool failures, etc.)
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
