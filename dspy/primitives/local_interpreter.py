from __future__ import annotations

import ast
import contextlib
import inspect
import io
import keyword
import sys
import threading
from collections.abc import Callable
from typing import Any

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput


class _Submission(BaseException):
    def __init__(self, value: Any) -> None:
        self.value = value


class LocalInterpreter:
    """Execute trusted Python code in the DSPy host process.

    This interpreter is fast and supports the host's installed Python packages,
    but it is not a security sandbox. Executed code has the same filesystem,
    network, environment, process, and credential access as the DSPy process.
    Use :class:`PythonInterpreter` for sandboxed model-generated code.

    Args:
        tools: Host functions exposed to executed code by name.
        output_fields: Optional field definitions for typed ``SUBMIT`` calls.
    """

    execution_instructions = (
        "Code runs as trusted Python in the host process. State, imports, functions, and variables persist "
        "for this session. Host tools and SUBMIT are available as global functions."
    )
    _execution_lock = threading.RLock()

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        self.tools = dict(tools or {})
        self.output_fields = None if output_fields is None else [dict(field) for field in output_fields]
        self._namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self._ended = False

    def start(self) -> None:
        """Validate that the in-process session is still active."""
        if self._ended:
            raise CodeInterpreterError("LocalInterpreter session has been shut down.")

    def _submit(self, *args: Any, **kwargs: Any) -> None:
        if self.output_fields is None:
            signature = inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
            value = signature.bind(*args, **kwargs).arguments["value"]
            raise _Submission({"output": value})

        names = [field["name"] for field in self.output_fields]
        signature = inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in names]
        )
        values = signature.bind(*args, **kwargs).arguments
        raise _Submission(dict(values))

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute Python code in a persistent in-process namespace."""
        with self._execution_lock:
            self.start()
            if variables:
                if any(
                    not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
                    for name in variables
                ):
                    raise CodeInterpreterError("Variable names must be valid Python identifiers.")
                self._namespace.update(variables)

            old_capabilities = self._namespace.pop("__dspy_capabilities__", set())
            for name in old_capabilities:
                self._namespace.pop(name, None)
            capabilities = set(self.tools) | {"SUBMIT"}
            self._namespace.update(self.tools)
            self._namespace["SUBMIT"] = self._submit
            self._namespace["__dspy_capabilities__"] = capabilities

            tree = ast.parse(code, mode="exec")
            stdout = io.StringIO()
            host_dspy_module = sys.modules.get("dspy")
            try:
                with contextlib.redirect_stdout(stdout):
                    value = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                        exec(compile(prefix, "<local-interpreter>", "exec"), self._namespace)
                        value = eval(
                            compile(ast.Expression(tree.body[-1].value), "<local-interpreter>", "eval"),
                            self._namespace,
                        )
                    else:
                        exec(compile(tree, "<local-interpreter>", "exec"), self._namespace)
            except _Submission as submission:
                return FinalOutput(submission.value)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                raise CodeExecutionError(f"{type(exc).__name__}: {exc}") from exc
            finally:
                if host_dspy_module is None:
                    sys.modules.pop("dspy", None)
                else:
                    sys.modules["dspy"] = host_dspy_module

        output = stdout.getvalue().rstrip("\n")
        return value if value is not None else (output or None)

    def __enter__(self) -> LocalInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        """End the session and discard its namespace."""
        with self._execution_lock:
            self._ended = True
            self._namespace.clear()
