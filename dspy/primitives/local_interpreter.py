from __future__ import annotations

import contextlib
import inspect
import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.syncify import _run_in_thread, run_async


class LocalInterpreter:
    """Execute Python in a persistent local subprocess.

    The process isolates DSPy's memory, stdout, and lifecycle, but is not a security sandbox: it retains the host
    user's files, environment, credentials, subprocess, and network authority.

    Args:
        tools: Host functions exposed by name with JSON-compatible arguments and results.
        output_fields: Optional field definitions for typed ``SUBMIT`` calls.
        execution_timeout: Maximum seconds for execution, including host tools. A timeout terminates the session, but
            its host callable may finish in a detached daemon thread; the result is discarded.
        callbacks: Optional instance-level callback handlers.
    """

    execution_instructions = (
        "Code runs as Python in a persistent local subprocess. State, imports, functions, and JSON-compatible "
        "variables persist for this session. Host tools and SUBMIT are available as global functions."
    )

    def __init__(
        self,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict[str, Any]] | None = None,
        *,
        execution_timeout: float | None = None,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        if execution_timeout is not None and execution_timeout <= 0:
            raise ValueError("execution_timeout must be positive or None")
        self.tools = dict(tools or {})
        self.output_fields = None if output_fields is None else [dict(field) for field in output_fields]
        self.execution_timeout = execution_timeout
        self.callbacks = list(callbacks or [])
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._ended = False

    @with_callbacks
    def start(self) -> None:
        """Start the worker, or return immediately if it is already running."""
        if self._ended:
            raise CodeInterpreterError("LocalInterpreter session has been shut down.")
        if self._process is not None:
            return
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).with_name("local_interpreter_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            raise CodeInterpreterError(f"Unable to start Python worker: {exc}") from exc
        self._process = process
        threading.Thread(target=self._read_responses, args=(process,), daemon=True).start()
        message = self._receive(time.monotonic() + 10, "Python worker did not start within 10 seconds.")
        if message != ["ready"]:
            self._raise_terminal_error(f"Python worker returned an invalid startup message: {message!r}")

    def _read_responses(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _send(self, message: list[Any]) -> None:
        process = self._process
        assert process is not None and process.stdin is not None
        try:
            print(json.dumps(message, separators=(",", ":"), allow_nan=False), file=process.stdin, flush=True)
        except (OSError, TypeError, ValueError) as exc:
            self._raise_terminal_error(f"Unable to send request to Python worker: {exc}", exc)

    def _receive(self, deadline: float | None, timeout_message: str) -> list[Any]:
        timeout = None if deadline is None else max(0, deadline - time.monotonic())
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty:
            self._raise_terminal_error(timeout_message)
        if line is None:
            self._raise_terminal_error("Python worker exited unexpectedly.")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self._raise_terminal_error(f"Python worker returned invalid JSON: {line!r}", exc)
        if not isinstance(message, list) or not message or not isinstance(message[0], str):
            self._raise_terminal_error(f"Python worker returned an invalid message: {message!r}")
        return message

    def _raise_terminal_error(self, message: str, cause: BaseException | None = None) -> NoReturn:
        self._ended = True
        self._kill()
        raise CodeInterpreterError(message) from cause

    def _kill(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    @with_callbacks
    def invoke_tool(self, tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """Invoke one currently registered host tool."""
        if tool_name not in self.tools:
            raise CodeInterpreterError(f"Unknown tool: {tool_name}")
        result = self.tools[tool_name](*args, **kwargs)
        return run_async(result) if inspect.isawaitable(result) else result

    def _handle_tool(self, request: list[Any], deadline: float | None, timeout_message: str) -> None:
        _, name, args, kwargs = request

        def invoke() -> list[Any]:
            try:
                value = self.invoke_tool(name, args, kwargs)
                json.dumps(value, allow_nan=False)
                return ["tool_result", value]
            except Exception as exc:
                return ["tool_error", f"{type(exc).__name__}: {exc}"]

        if deadline is None:
            response = invoke()
        else:
            timeout = max(0, deadline - time.monotonic())
            try:
                response = _run_in_thread(invoke).result(timeout)
            except TimeoutError:
                self._raise_terminal_error(timeout_message)
        self._send(response)

    @with_callbacks
    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute code in the worker's persistent namespace."""
        if not self._lock.acquire(blocking=False):
            raise CodeInterpreterError("LocalInterpreter already has an active execution.")
        try:
            variables = {} if variables is None else variables
            if not isinstance(code, str):
                raise CodeInterpreterError("code must be a string")
            if not isinstance(variables, dict) or any(
                not isinstance(name, str) or not name.isidentifier() for name in variables
            ):
                raise CodeInterpreterError("variables must map Python identifiers to JSON-compatible values")
            try:
                json.dumps(variables, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CodeInterpreterError(f"variables must be JSON-compatible: {exc}") from exc

            if self._process is None:
                self.start()
            deadline = None if self.execution_timeout is None else time.monotonic() + self.execution_timeout
            self._send(["execute", code, variables, list(self.tools), self.output_fields])
            timeout_message = (
                "Python worker did not respond."
                if self.execution_timeout is None
                else f"Python worker exceeded execution timeout of {self.execution_timeout:g} seconds."
            )
            message = self._receive(deadline, timeout_message)
            while message[0] == "tool":
                self._handle_tool(message, deadline, timeout_message)
                message = self._receive(deadline, timeout_message)
            kind, *payload = message
            if kind == "terminal_error":
                self._raise_terminal_error(f"Python worker failed: {payload[0]}")
            if kind == "syntax":
                raise SyntaxError(payload[0])
            if kind == "execution_error":
                raise CodeExecutionError(payload[0])
            if kind == "final":
                return FinalOutput(payload[0])
            if kind == "result":
                value, stdout = payload
                return value if value is not None else (stdout or None)
            self._raise_terminal_error(f"Python worker returned an unknown message: {message!r}")
        finally:
            self._lock.release()

    def __enter__(self) -> LocalInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    __call__ = execute

    @with_callbacks
    def shutdown(self) -> None:
        """Terminate the worker and discard its session state."""
        if self._ended:
            return
        self._ended = True
        if self._process is not None:
            with contextlib.suppress(CodeInterpreterError, subprocess.TimeoutExpired):
                self._send(["shutdown"])
                self._process.wait(timeout=1)
        self._kill()
