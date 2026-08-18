from __future__ import annotations

import ast
import contextlib
import io
import json
import keyword
import sys
import uuid
from typing import Any


class Submission(BaseException):
    def __init__(self, value: Any) -> None:
        self.value = value


class Session:
    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.capabilities: set[str] = set()
        self.output_fields: list[dict[str, Any]] | None = None

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        request_id = uuid.uuid4().hex
        send({"type": "tool_request", "id": request_id, "name": name, "args": args, "kwargs": kwargs})
        response = receive()
        if response.get("type") != "tool_response" or response.get("id") != request_id:
            raise RuntimeError("mismatched host-tool response")
        if not response.get("ok"):
            raise RuntimeError(response.get("error"))
        return response.get("value")

    def configure(self, tool_names: list[str], output_fields: list[dict[str, Any]] | None) -> None:
        if any(not name.isidentifier() or keyword.iskeyword(name) or name == "SUBMIT" for name in tool_names):
            raise ValueError("tool names must be Python identifiers other than SUBMIT")
        if output_fields is not None:
            names = [field.get("name") for field in output_fields]
            if any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) for name in names):
                raise ValueError("output field names must be Python identifiers")
            if len(names) != len(set(names)):
                raise ValueError("output field names must be unique")
        for name in self.capabilities:
            self.namespace.pop(name, None)
        self.capabilities = set(tool_names) | {"SUBMIT"}
        self.output_fields = output_fields
        for name in tool_names:
            self.namespace[name] = lambda *args, __name=name, **kwargs: self.call_tool(__name, *args, **kwargs)
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
        self.configure(request.get("tools") or [], request.get("output_fields"))
        variables = request.get("variables") or {}
        if not isinstance(variables, dict) or any(not name.isidentifier() for name in variables):
            raise ValueError("variables must map Python identifiers to JSON values")
        self.namespace.update(variables)
        code = request.get("code")
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        captured = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                value = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1], type_ignores=[]), "<interpreter>", "exec"), self.namespace)
                    value = eval(compile(ast.Expression(tree.body[-1].value), "<interpreter>", "eval"), self.namespace)
                else:
                    exec(compile(tree, "<interpreter>", "exec"), self.namespace)
        except Submission as submission:
            return {"type": "execution_result", "kind": "final", "value": jsonable(submission.value)}
        except SyntaxError as exc:
            return {"type": "execution_result", "kind": "syntax", "error": str(exc)}
        except BaseException as exc:
            return {"type": "execution_result", "kind": "execution_error", "error": describe(exc)}
        return {
            "type": "execution_result",
            "kind": "result",
            "value": jsonable(value),
            "stdout": captured.getvalue().rstrip("\n"),
        }


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def send(message: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    sys.__stdout__.flush()


def receive() -> dict[str, Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise EOFError
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("protocol messages must be objects")
    return message


def main() -> None:
    session = Session()
    send({"type": "ready"})
    while True:
        try:
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
            return


if __name__ == "__main__":
    main()
