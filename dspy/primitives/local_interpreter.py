"""
Unsandboxed local Python interpreter for RLM.

Implements the CodeInterpreter protocol but executes code directly in the host
Python process via exec(). This gives the RLM agent full access to any installed
Python package (PIL, pydub, numpy, scipy, etc.).

Use this when the sandboxed PythonInterpreter (Deno/Pyodide) is too restrictive —
e.g., when the RLM agent needs to manipulate images with PIL or process audio
with pydub directly in its generated code.

Security: This is intentionally UNSANDBOXED. The LLM-generated code runs with
full host process privileges. Only use for local experiments or trusted workloads.

Usage:
    from dspy.primitives.local_interpreter import LocalInterpreter

    rlm = dspy.RLM("context -> answer", interpreter=LocalInterpreter())
"""

import io
import sys
import traceback
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput


class _SubmitCalledError(Exception):
    """Internal signal raised when SUBMIT() is called in user code."""
    def __init__(self, output: Any):
        self.output = output


class LocalInterpreter:
    """Unsandboxed Python interpreter implementing the CodeInterpreter protocol.

    Executes code directly in the host process via exec(). State persists
    across execute() calls within a session. Tools are injected as callable
    functions in the execution namespace.

    This gives the RLM agent full access to the host Python environment:
    - PIL/Pillow for image manipulation
    - pydub/ffmpeg for audio manipulation
    - numpy, scipy, scikit-image, etc.
    - Any installed Python package

    Note: Not thread-safe. Create separate instances for concurrent use.
    """

    def __init__(
        self,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
    ):
        """
        Args:
            tools: Dictionary mapping tool names to callable functions.
                   Tools are available as top-level functions in the namespace.
            output_fields: Output field definitions for typed SUBMIT signature.
        """
        self._tools: dict[str, Callable[..., str]] = dict(tools) if tools else {}
        self.output_fields = output_fields
        self._namespace: dict[str, Any] = {}
        self._started = False

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        """Tools available for interpreter code to call."""
        return self._tools

    @tools.setter
    def tools(self, value: dict[str, Callable[..., str]]) -> None:
        self._tools = value

    def start(self) -> None:
        """Initialize the interpreter namespace."""
        if self._started:
            return
        self._namespace = {"__builtins__": __builtins__}
        self._started = True

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute Python code in the host process.

        Args:
            code: Python code to execute.
            variables: Variables to inject into the namespace before execution.
                      Media objects (Audio, Image) are injected AS-IS, giving
                      code direct access to their data for manipulation.

        Returns:
            - FinalOutput: If SUBMIT() was called
            - str: Captured stdout (from print() calls)
            - None: If no output was produced

        Raises:
            CodeInterpreterError: On runtime errors
            SyntaxError: On invalid Python syntax
        """
        if not self._started:
            self.start()

        # Inject variables directly into namespace (no serialization — objects stay live)
        if variables:
            self._namespace.update(variables)

        # Inject tools as callable functions
        for name, func in self._tools.items():
            self._namespace[name] = func

        # Inject SUBMIT function — maps args to output field names (matching PythonInterpreter)
        output_fields = self.output_fields or []
        field_names = [f["name"] for f in output_fields]

        def SUBMIT(*args, **kwargs):  # noqa: N802
            if not args and not kwargs:
                raise ValueError("SUBMIT requires at least one argument")
            if args and kwargs:
                raise ValueError("SUBMIT accepts either positional args or keyword args, not both")
            if kwargs:
                output = kwargs
            elif field_names:
                if len(args) != len(field_names):
                    expected = ", ".join(field_names)
                    raise TypeError(
                        f"SUBMIT() takes {len(field_names)} positional argument(s) "
                        f"({expected}), but {len(args)} were given"
                    )
                output = dict(zip(field_names, args, strict=False))
            elif len(args) == 1:
                output = {"output": args[0]}
            else:
                output = {"output": args}
            raise _SubmitCalledError(output)

        self._namespace["SUBMIT"] = SUBMIT

        # Capture stdout
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        try:
            exec(code, self._namespace)
        except _SubmitCalledError as e:
            return FinalOutput(e.output)
        except SyntaxError:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            raise CodeInterpreterError(f"{type(e).__name__}: {e}\n{tb}") from e
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        if output:
            return output.rstrip("\n")
        return None

    def shutdown(self) -> None:
        """Release resources and clear the namespace."""
        self._namespace.clear()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.shutdown()
