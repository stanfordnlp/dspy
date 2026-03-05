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
    """Error raised during code interpretation.

    This exception covers two distinct failure modes:

    1. **Execution errors**: The sandbox ran user code that failed.
       - NameError, TypeError, ValueError, etc.
       - Tool call failures (unknown tool, tool raised exception)
       - These are normal user code errors.

    2. **Protocol errors**: Communication between host and sandbox failed.
       - Malformed JSON from sandbox
       - Sandbox process crashed or became unresponsive
       - Invalid JSON-RPC message structure
       - These may indicate a corrupted sandbox needing restart.

    The error message typically includes the original error type (e.g., "NameError: ...")
    which can help distinguish the failure mode.

    Note: SyntaxError is raised separately (not wrapped) for invalid Python syntax.
    """


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
            - FinalOutput: If SUBMIT() was called in code
            - str: Captured stdout from print() statements
            - list: Multiple output lines
            - None: If no output was produced

        Raises:
            CodeInterpreterError: On runtime errors (undefined vars, tool failures, etc.)
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
