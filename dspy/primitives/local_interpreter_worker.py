from __future__ import annotations

import ast
import builtins
import contextlib
import itertools
import json
import keyword
import os
import queue
import sys
import tempfile
import threading
from typing import Any

_protocol_output = os.fdopen(os.dup(sys.__stdout__.fileno()), "w", encoding="utf-8")
os.set_inheritable(_protocol_output.fileno(), False)
_worker_stdout, _worker_stderr = sys.stdout, sys.stderr
_sink = os.open(os.devnull, os.O_WRONLY)
os.dup2(_sink, 1)
os.dup2(_sink, 2)
os.close(_sink)
_send_lock = threading.Lock()


class Submission(BaseException):
    def __init__(self, value: Any) -> None:
        self.value = value


class Protocol:
    def __init__(self) -> None:
        self.commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.pending: dict[int, queue.Queue[dict[str, Any] | None]] = {}
        self.lock = threading.Lock()
        self.ids = itertools.count()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            while True:
                message = receive()
                request_id = message.get("id")
                if message.get("type") in {"tool_result", "tool_error"} and isinstance(request_id, int):
                    with self.lock:
                        response = self.pending.get(request_id)
                    if response is None:
                        raise ValueError("response does not match a pending tool request")
                    response.put(message)
                else:
                    self.commands.put(message)
        except EOFError:
            self.commands.put(None)
            with self.lock:
                pending = list(self.pending.values())
            for response in pending:
                response.put(None)
        except BaseException as exc:
            self.commands.put({"type": "protocol_error", "error": describe(exc)})

    def call_tool(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        request_id = next(self.ids)
        response: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
        with self.lock:
            self.pending[request_id] = response
        try:
            send({"type": "tool_request", "id": request_id, "name": name, "args": args, "kwargs": kwargs})
            message = response.get()
        finally:
            with self.lock:
                self.pending.pop(request_id, None)
        if message is None:
            raise EOFError
        if message.get("type") == "tool_error":
            raise RuntimeError(message.get("error"))
        return message.get("value")


class CapturedOutput:
    def __enter__(self) -> CapturedOutput:
        self.file = tempfile.TemporaryFile()
        sys.stdout, sys.stderr = _worker_stdout, _worker_stderr
        for stream in (_worker_stdout, _worker_stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        self.saved = os.dup(1), os.dup(2)
        os.dup2(self.file.fileno(), 1)
        os.dup2(self.file.fileno(), 2)
        return self

    def __exit__(self, *_: Any) -> None:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        sys.stdout, sys.stderr = _worker_stdout, _worker_stderr
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        os.close(self.saved[0])
        os.close(self.saved[1])
        self.file.seek(0)
        self.value = self.file.read().decode(errors="replace").rstrip("\n")
        self.file.close()


class Session:
    def __init__(self, protocol: Protocol) -> None:
        self.protocol = protocol
        self.worker_threads = set(threading.enumerate())
        self.namespace: dict[str, Any] = {"__builtins__": vars(builtins).copy()}
        self.tool_names: set[str] = set()
        self.output_fields: list[dict[str, Any]] | None = None

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.protocol.call_tool(name, args, kwargs)

    def configure(self, tool_names: list[str], output_fields: list[dict[str, Any]] | None) -> None:
        if any(
            not name.isidentifier() or keyword.iskeyword(name) or name in {"SUBMIT", "__builtins__"}
            for name in tool_names
        ):
            raise ValueError("tool names must be non-reserved Python identifiers")
        if output_fields is not None:
            names = [field.get("name") for field in output_fields]
            if any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) for name in names):
                raise ValueError("output field names must be Python identifiers")
            if len(names) != len(set(names)):
                raise ValueError("output field names must be unique")
        for name in self.tool_names:
            self.namespace.pop(name, None)
        self.tool_names = set(tool_names)
        self.output_fields = output_fields
        for name in tool_names:
            self.namespace[name] = lambda *args, __name=name, **kwargs: self.call_tool(__name, *args, **kwargs)
        self.namespace["__builtins__"] = vars(builtins).copy()
        self.namespace["SUBMIT"] = self.submit

    def submit(self, *args: Any, **kwargs: Any) -> None:
        if self.output_fields is None:
            if len(args) != 1 or kwargs:
                raise TypeError("SUBMIT requires one output value")
            raise Submission({"output": args[0]})
        names = [field["name"] for field in self.output_fields]
        if args and kwargs:
            raise TypeError("SUBMIT accepts positional or keyword values, not both")
        values = dict(zip(names, args, strict=False)) if args else dict(kwargs)
        if set(values) != set(names) or len(args) > len(names):
            raise TypeError("SUBMIT fields do not match the configured output fields")
        raise Submission(values)

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        self.configure(request["tools"], request.get("output_fields"))
        variables = request["variables"]
        if {"SUBMIT", "__builtins__", *self.tool_names} & variables.keys():
            raise ValueError("variables cannot replace interpreter-owned globals")
        code = request["code"]
        self.namespace.update(variables)
        outcome: dict[str, Any]
        with CapturedOutput() as captured:
            try:
                tree = ast.parse(code, mode="exec")
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<interpreter>", "exec"), self.namespace)
                value = (
                    eval(compile(ast.Expression(last.value), "<interpreter>", "eval"), self.namespace) if last else None
                )
                outcome = {"type": "result", "value": jsonable(value)}
            except Submission as submission:
                outcome = {"type": "final", "value": jsonable(submission.value)}
            except SyntaxError as exc:
                outcome = {"type": "syntax", "error": str(exc)}
            except BaseException as exc:
                outcome = {"type": "execution_error", "error": describe(exc)}
        outcome["stdout"] = captured.value
        leaked_threads = [thread.name for thread in threading.enumerate() if thread not in self.worker_threads]
        if leaked_threads:
            return {
                "type": "terminal_error",
                "error": f"executed code left background threads running: {leaked_threads!r}",
            }
        return outcome


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def send(message: dict[str, Any]) -> None:
    with _send_lock:
        print(json.dumps(message, separators=(",", ":"), allow_nan=False), file=_protocol_output, flush=True)


def receive() -> dict[str, Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise EOFError
    message = json.loads(line)
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ValueError("protocol messages must be tagged objects")
    return message


def main() -> None:
    protocol = Protocol()
    session = Session(protocol)
    send({"type": "ready"})
    try:
        while True:
            request = protocol.commands.get()
            if request is None:
                return
            if request.get("type") == "shutdown":
                return
            if request.get("type") == "protocol_error":
                raise ValueError(request.get("error"))
            if request.get("type") != "execute":
                raise ValueError("unknown request")
            response = session.execute(request)
            send(response)
            if response.get("type") == "terminal_error":
                return
    except EOFError:
        return
    except BaseException as exc:
        send({"type": "terminal_error", "error": describe(exc)})


if __name__ == "__main__":
    main()
