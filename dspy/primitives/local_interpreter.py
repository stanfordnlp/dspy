from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import json
import keyword
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, NoReturn

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreterError, FinalOutput
from dspy.utils.callback import BaseCallback, with_callbacks

_MAX_MESSAGE_CHARS = 16 * 1024 * 1024
_MAX_PENDING_TOOLS = 32


def _run_in_thread(function: Callable[[], Any]) -> Future[Any]:
    outcome = Future()

    def run() -> None:
        try:
            outcome.set_result(function())
        except BaseException as exc:
            outcome.set_exception(exc)

    threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True).start()
    return outcome


def _await_in_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    return _run_in_thread(lambda: asyncio.run(awaitable)).result()


class LocalInterpreter:
    """Execute Python in a persistent, MXC-confined local CPython process.

    Installed Python libraries are readable; only a private working directory
    and temporary directory are writable. Host environment credentials are not
    inherited and direct IP networking is denied. Host tools retain their host
    authority. MXC also grants OS-specific baseline system access: in particular,
    macOS permits broad system reads and detached children can survive shutdown.
    This is not a VM or an aggregate memory, process, or disk resource limit.

    The separately installed ``dspy-mxc-runtime`` wheel supplies the native
    executor (and bubblewrap on Linux). Unavailable or prohibited OS sandbox
    facilities fail closed; no runtime downloads or host-policy changes occur.

    Guest protocol messages are limited to 16 Mi characters, with at most 32
    outstanding host tool calls per execution. These transport bounds are not
    aggregate guest memory, process, disk, or output quotas.

    Args:
        tools: Host functions exposed to executed code by name. Arguments and
            return values must be JSON-compatible.
        output_fields: Optional field definitions for typed ``SUBMIT`` calls.
        execution_timeout: Maximum seconds for one execution, including host
            tool calls. A timeout terminates the worker and its session state.
            Python cannot forcibly stop a running host callable, so a timed-out
            callable may finish in a detached daemon thread; its result is discarded.
        callbacks: Optional instance-level callback handlers.
    """

    execution_instructions = (
        "Code runs in a persistent MXC-confined CPython worker. Installed Python libraries are available. "
        "State, imports, functions, and JSON-compatible variables persist for this session. "
        "Read and write files in the working directory; other host files are restricted. Direct network access "
        "is disabled. Host tools and SUBMIT are available as global functions. Finish all threads and child "
        "processes before returning; use context-managed executors for parallel work."
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
        self._validate_tool_names(tools or {})
        self.tools = dict(tools or {})
        fields = None if output_fields is None else [dict(field) for field in output_fields]
        self._validate_output_fields(fields)
        self.output_fields = fields
        self.execution_timeout = execution_timeout
        self.callbacks = list(callbacks or [])
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | BaseException | None] = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._ended = False
        self._session: Path | None = None
        self._diagnostics = ""

    @staticmethod
    def _validate_tool_names(tools: dict[str, Callable[..., Any]]) -> None:
        invalid = [
            name
            for name in tools
            if not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
            or name in {"SUBMIT", "__builtins__"}
        ]
        if invalid:
            raise ValueError(f"tool names must be non-reserved Python identifiers; invalid names: {invalid!r}")

    @staticmethod
    def _validate_output_fields(output_fields: list[dict[str, Any]] | None) -> None:
        if output_fields is None:
            return
        names = [field.get("name") if isinstance(field, dict) else None for field in output_fields]
        if any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) for name in names):
            raise ValueError("output field names must be non-keyword Python identifiers")
        if len(names) != len(set(names)):
            raise ValueError("output field names must be unique")
        try:
            json.dumps(output_fields, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"output fields must be JSON-compatible: {exc}") from exc

    @with_callbacks
    def start(self) -> None:
        """Start the worker, or return immediately if it is already running."""
        if self._ended:
            raise CodeInterpreterError("LocalInterpreter session has been shut down.")
        if self._process is not None:
            return
        try:
            process = self._spawn_worker()
            self._process = process
            threading.Thread(target=self._read_responses, args=(process,), daemon=True).start()
            message = self._receive(time.monotonic() + 10, "Python worker did not start within 10 seconds.")
            if message.get("type") != "ready":
                self._raise_terminal_error(f"Python worker returned an invalid startup message: {message!r}")
        except BaseException as exc:
            self._ended = True
            self._kill()
            if not isinstance(exc, Exception):
                raise
            raise CodeInterpreterError(f"Unable to start Python sandbox: {exc}\n{self._diagnostics}") from exc

    def _spawn_worker(self) -> subprocess.Popen[str]:
        """Keep policy/launch ownership here; MXC owns OS enforcement."""
        try:
            from dspy_mxc_runtime import executables
        except ImportError as exc:
            raise CodeInterpreterError("Install the dspy-mxc-runtime wheel to use LocalInterpreter.") from exc
        runtime, bwrap = executables()
        # Resolve macOS /var -> /private/var before generating Seatbelt paths.
        self._session = Path(tempfile.mkdtemp(prefix="dspy-mxc-")).resolve()
        workspace = self._session / "work"
        temporary = self._session / "tmp"
        workspace.mkdir()
        temporary.mkdir()
        worker = self._session / "worker.py"
        shutil.copyfile(Path(__file__).with_name("local_interpreter_worker.py"), worker)
        env = {
            "HOME": str(temporary),
            "TMPDIR": str(temporary),
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        if os.name == "nt":
            env.update(
                SYSTEMROOT=os.environ["SYSTEMROOT"],
                LOCALAPPDATA=str(temporary),
                TEMP=str(temporary),
                TMP=str(temporary),
            )
        else:
            env["PATH"] = os.pathsep.join(([str(bwrap.parent)] if bwrap else []) + ["/usr/local/bin", os.defpath])
        python = [sys.executable, "-I", "-B", "-u", "-X", "utf8", str(worker)]
        readonly = {sys.prefix, sys.base_prefix, str(Path(sys.executable).resolve().parent), str(worker)}
        # uv venvs may point through an alias of their base installation.
        base_alias = Path(sys._base_executable).parent.parent
        if base_alias.resolve() == Path(sys.base_prefix).resolve():
            readonly.add(str(base_alias))
        readonly.update(str(Path(path).resolve()) for path in tuple(readonly))
        policy = {
            "version": "1.0.0",
            "containment": {"linux": "bubblewrap", "darwin": "seatbelt", "win32": "processcontainer"}[sys.platform],
            "containerId": self._session.name,
            "process": {
                "commandLine": subprocess.list2cmdline(python) if os.name == "nt" else "exec " + shlex.join(python),
                "cwd": str(workspace),
                "env": [f"{key}={value}" for key, value in env.items()],
                "inheritDefaultEnv": False,
                # DSPy owns execution deadlines, not a competing session timer.
                "timeout": 0,
            },
            "filesystem": {
                "readonlyPaths": sorted(readonly),
                "readwritePaths": [str(workspace), str(temporary)],
                **({"deniedPaths": ["/etc"]} if sys.platform == "linux" else {}),
            },
            "network": {"egress": {"default": "deny"}, "ingress": {"default": "deny", "hostLoopback": "deny"}},
            "fallback": {"allowDaclMutation": False},
            "telemetry": {"enabled": False},
        }
        policy_path = self._session / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        process = subprocess.Popen(
            [str(runtime), "--config", str(policy_path)],
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name == "posix",
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._process = process
        threading.Thread(target=self._read_diagnostics, args=(process,), daemon=True).start()
        return process

    def _read_diagnostics(self, process: subprocess.Popen[str]) -> None:
        # Drain continuously without an unbounded on-disk log or growing buffer.
        assert process.stderr is not None
        with process.stderr:
            while chunk := process.stderr.read(4096):
                self._diagnostics = (self._diagnostics + chunk)[-8192:]

    def _read_responses(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        # Only the reader closes stdout: closing a buffered pipe from another
        # thread waits for readline, potentially held open by a Mac descendant.
        with process.stdout:
            while self._process is process:
                message: str | BaseException | None
                try:
                    message = process.stdout.readline(_MAX_MESSAGE_CHARS + 1) or None
                    if message is not None and len(message) > _MAX_MESSAGE_CHARS:
                        message = CodeInterpreterError("Python worker message exceeds the 16 Mi-character limit.")
                except (OSError, ValueError) as exc:
                    message = CodeInterpreterError(f"Unable to read Python worker response: {exc}")
                # Apply backpressure without stranding this reader after shutdown.
                while self._process is process:
                    try:
                        self._responses.put(message, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if not isinstance(message, str):
                    return

    def _send(self, message: dict[str, Any], process: subprocess.Popen[str] | None = None) -> None:
        with self._send_lock:
            current = self._process
            if process is not None and current is not process:
                return
            assert current is not None and current.stdin is not None
            try:
                print(json.dumps(message, separators=(",", ":"), allow_nan=False), file=current.stdin, flush=True)
            except (OSError, TypeError, ValueError) as exc:
                self._raise_terminal_error(f"Unable to send request to Python worker: {exc}", exc)
            finally:
                if self._process is not current:
                    with contextlib.suppress(OSError):
                        current.stdin.close()

    def _receive(self, deadline: float | None, timeout_message: str) -> dict[str, Any]:
        timeout = None if deadline is None else max(0, deadline - time.monotonic())
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty:
            self._raise_terminal_error(timeout_message)
        if line is None:
            self._raise_terminal_error("Python worker exited unexpectedly.")
        if isinstance(line, BaseException):
            self.shutdown()
            raise line
        try:
            message = json.loads(line)
        except (ValueError, RecursionError) as exc:
            self._raise_terminal_error(f"Python worker returned invalid JSON: {line[:200]!r}", exc)
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            self._raise_terminal_error("Python worker returned an invalid message.")
        return message

    def _raise_terminal_error(self, message: str, cause: BaseException | None = None) -> NoReturn:
        self._ended = True
        self._kill()
        raise CodeInterpreterError(message) from cause

    def _kill(self) -> None:
        # A deadline and a broken command writer can terminate concurrently.
        with self._cleanup_lock:
            process = self._process
            self._process = None
            if process is not None:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(OSError):
                    # Windows MXC owns a kill-on-close Job Object. Do not depend
                    # on a taskkill subprocess or mutate host ACLs for cleanup.
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)
                # A blocked writer owns stdin until it unwinds. Closing it here
                # would wait for its buffered write and defeat the deadline.
                if self._send_lock.acquire(blocking=False):
                    try:
                        if process.stdin is not None:
                            with contextlib.suppress(OSError):
                                process.stdin.close()
                    finally:
                        self._send_lock.release()
            if self._session is not None:
                shutil.rmtree(self._session)
                self._session = None

    @with_callbacks
    def invoke_tool(self, tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """Invoke one currently registered host tool."""
        if tool_name not in self.tools:
            raise CodeInterpreterError(f"Unknown tool: {tool_name}")
        result = self.tools[tool_name](*args, **kwargs)
        return _await_in_sync(result) if inspect.isawaitable(result) else result

    def _handle_tool(self, request: dict[str, Any], process: subprocess.Popen[str]) -> None:
        try:
            value = self.invoke_tool(request["name"], request.get("args", []), request.get("kwargs", {}))
            json.dumps(value, allow_nan=False)
            response = {"type": "tool_result", "id": request["id"], "value": value}
        except Exception as exc:
            response = {"type": "tool_error", "id": request["id"], "error": f"{type(exc).__name__}: {exc}"}
        except BaseException as exc:
            # Like PythonInterpreter, propagate cancellation/interrupts to the caller.
            # Wake execute(), but never strand a tool thread on a full guest queue.
            # The Future also carries the exception if a response is already queued.
            with contextlib.suppress(queue.Full):
                self._responses.put_nowait(exc)
            raise
        self._send(response, process)

    @with_callbacks
    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        """Execute code in the worker's persistent namespace."""
        if not self._lock.acquire(blocking=False):
            raise CodeInterpreterError("LocalInterpreter already has an active execution.")
        try:
            variables = {} if variables is None else variables
            self._validate_tool_names(self.tools)
            self._validate_output_fields(self.output_fields)
            if not isinstance(code, str):
                raise CodeInterpreterError("code must be a string")
            if not isinstance(variables, dict):
                raise CodeInterpreterError("variables must map Python identifiers to JSON-compatible values")
            invalid_variables = [
                name
                for name in variables
                if not isinstance(name, str)
                or not name.isidentifier()
                or keyword.iskeyword(name)
                or name in {"SUBMIT", "__builtins__"}
                or name in self.tools
            ]
            if invalid_variables:
                raise CodeInterpreterError(
                    "variable names must be non-reserved Python identifiers distinct from tool names; "
                    f"invalid names: {invalid_variables!r}"
                )
            try:
                json.dumps(variables, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise CodeInterpreterError(f"variables must be JSON-compatible: {exc}") from exc

            if self._process is None:
                self.start()
            deadline = None if self.execution_timeout is None else time.monotonic() + self.execution_timeout
            timeout_message = (
                "Python worker did not respond."
                if self.execution_timeout is None
                else f"Python worker exceeded execution timeout of {self.execution_timeout:g} seconds."
            )
            process = self._process
            assert process is not None
            request = {
                "type": "execute",
                "code": code,
                "variables": variables,
                "tools": list(self.tools),
                "output_fields": self.output_fields,
            }
            # A stopped worker can fill stdin before there is a response to wait
            # for. The same execution deadline must also cover that write.
            send = _run_in_thread(lambda: self._send(request, process))
            try:
                send.result(None if deadline is None else max(0, deadline - time.monotonic()))
            except FutureTimeoutError:
                self._raise_terminal_error(timeout_message)
            tool_calls: list[Future[Any]] = []
            message = self._receive(deadline, timeout_message)
            while message.get("type") == "tool_request":
                if (
                    type(message.get("id")) is not int
                    or message["id"] < 0
                    or not isinstance(message.get("name"), str)
                    or message["name"] not in self.tools
                    or not isinstance(message.get("args"), list)
                    or not isinstance(message.get("kwargs"), dict)
                ):
                    self._raise_terminal_error("Python worker returned an invalid tool request.")
                for tool_call in tuple(tool_calls):
                    if tool_call.done():
                        try:
                            tool_call.result()
                        except BaseException:
                            self.shutdown()
                            raise
                        tool_calls.remove(tool_call)
                if len(tool_calls) >= _MAX_PENDING_TOOLS:
                    self._raise_terminal_error("Python worker exceeded the limit of 32 outstanding host tool calls.")
                tool_calls.append(_run_in_thread(lambda request=message: self._handle_tool(request, process)))
                message = self._receive(deadline, timeout_message)
            for tool_call in tool_calls:
                timeout = None if deadline is None else max(0, deadline - time.monotonic())
                try:
                    tool_call.result(timeout)
                except FutureTimeoutError:
                    self._raise_terminal_error(timeout_message)
                except BaseException:
                    self.shutdown()
                    raise
            kind = message.get("type")
            if kind == "terminal_error":
                self._raise_terminal_error(f"Python worker failed: {message.get('error')}")
            if kind == "syntax":
                raise SyntaxError(message.get("error"))
            if kind == "execution_error":
                raise CodeExecutionError(message.get("error"))
            if kind == "final":
                return FinalOutput(message.get("value"))
            if kind == "result":
                value, stdout = message.get("value"), message.get("stdout")
                return value if value is not None else (stdout or None)
            self._raise_terminal_error(f"Python worker returned an unknown message: {message!r}")
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()
            raise
        finally:
            self._lock.release()

    def __enter__(self) -> LocalInterpreter:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    @with_callbacks
    def shutdown(self) -> None:
        """Terminate the worker and discard its session state."""
        if self._ended:
            return
        self._ended = True
        self._kill()
