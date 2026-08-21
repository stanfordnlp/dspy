from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import keyword
import sys
from typing import Any


class Submission(BaseException):
    def __init__(self, value: Any) -> None:
        self.value = value


class Session:
    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": vars(builtins).copy()}
        self.tool_names: set[str] = set()
        self.output_fields: list[dict[str, Any]] | None = None

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        send({"type": "tool_request", "name": name, "args": args, "kwargs": kwargs})
        response = receive()
        if response.get("type") == "tool_error":
            raise RuntimeError(response.get("error"))
        if response.get("type") != "tool_result":
            raise RuntimeError("invalid host-tool response")
        return response.get("value")

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
        captured = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                exec(compile(tree, "<interpreter>", "exec"), self.namespace)
                value = (
                    eval(compile(ast.Expression(last.value), "<interpreter>", "eval"), self.namespace) if last else None
                )
        except Submission as submission:
            return {"type": "final", "value": jsonable(submission.value)}
        except SyntaxError as exc:
            return {"type": "syntax", "error": str(exc)}
        except BaseException as exc:
            return {"type": "execution_error", "error": describe(exc)}
        return {"type": "result", "value": jsonable(value), "stdout": captured.getvalue().rstrip("\n")}


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def send(message: dict[str, Any]) -> None:
    print(json.dumps(message, separators=(",", ":"), allow_nan=False), file=sys.__stdout__, flush=True)


def receive() -> dict[str, Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise EOFError
    message = json.loads(line)
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ValueError("protocol messages must be tagged objects")
    return message


def main() -> None:
    session = Session()
    send({"type": "ready"})
    try:
        while True:
            request = receive()
            if request.get("type") == "shutdown":
                return
            if request.get("type") != "execute":
                raise ValueError("unknown request")
            send(session.execute(request))
    except EOFError:
        return
    except BaseException as exc:
        send({"type": "terminal_error", "error": describe(exc)})


if __name__ == "__main__":
    main()
