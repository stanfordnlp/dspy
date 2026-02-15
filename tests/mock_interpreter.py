"""
Mock interpreter for testing RLM and other code-executing modules.

This interpreter doesn't actually execute code - it returns scripted responses
or uses a custom function to generate responses. Useful for:
- Unit testing without Deno/Pyodide dependencies
- Testing specific execution paths (errors, SUBMIT, etc.)
- Recording what code was submitted for execution
"""

from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

__all__ = ["MockInterpreter"]


class MockInterpreter:
    """Mock interpreter that returns scripted responses.

    Implements the Interpreter protocol for testing purposes.

    Example usage:
        ```python
        # Script specific responses
        mock = MockInterpreter(responses=[
            "data explored",
            FinalOutput("42"),
        ])
        result1 = mock.execute("print(len(context))")  # Returns "data explored"
        result2 = mock.execute("SUBMIT('42')")  # Returns FinalOutput("42")

        # Use custom execution function
        def custom_exec(code, variables):
            if "SUBMIT" in code:
                return FinalOutput("done")
            return f"executed: {code[:20]}..."

        mock = MockInterpreter(execute_fn=custom_exec)
        ```
    """

    def __init__(
        self,
        responses: list[str | FinalOutput | Exception] | None = None,
        execute_fn: Callable[[str, dict[str, Any]], Any] | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
    ):
        """Initialize the mock interpreter.

        Args:
            responses: List of responses to return in sequence. Each call to
                      execute() pops the next response. If an Exception is
                      in the list, it will be raised.
            execute_fn: Custom function that receives (code, variables) and
                       returns the result. Takes precedence over responses.
            tools: Dictionary mapping tool names to callable functions.
                   MockInterpreter doesn't use tools, but stores them for protocol compliance.
        """
        self.responses = list(responses) if responses else []
        self.execute_fn = execute_fn
        self.tools = tools or {}
        self.call_count = 0
        self.call_history: list[tuple[str, dict[str, Any]]] = []
        self._shutdown = False

    def start(self) -> None:
        pass

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute code and return the next scripted response.

        Args:
            code: The code that would be executed (recorded in call_history)
            variables: Variables that would be injected (recorded in call_history)

        Returns:
            The next response from the responses list, or result from execute_fn

        Raises:
            CodeInterpreterError: If the interpreter was shutdown, or if an Exception
                             is in the responses list
        """
        if self._shutdown:
            raise CodeInterpreterError("MockInterpreter has been shutdown")

        variables = variables or {}
        self.call_history.append((code, variables))
        self.call_count += 1

        # Custom function takes precedence
        if self.execute_fn is not None:
            return self.execute_fn(code, variables)

        # Return scripted responses
        if not self.responses:
            return ""

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response

    def shutdown(self) -> None:
        self._shutdown = True

    def reset(self) -> None:
        """Reset the interpreter state for reuse in tests."""
        self.call_count = 0
        self.call_history = []
        self._shutdown = False

    # Context manager support
    def __enter__(self) -> "MockInterpreter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __call__(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Shorthand for execute()."""
        return self.execute(code, variables)
