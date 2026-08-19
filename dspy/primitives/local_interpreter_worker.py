import ast
import contextlib
import io
import json
import keyword
import sys
from typing import Any


class Submission(BaseException):
    pass


class Session:
    def __init__(self) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.tool_names: set[str] = set()
        self.output_fields: list[dict[str, Any]] | None = None

    def call_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        send(["tool", name, args, kwargs])
        response = receive()
        kind, *payload = response
        if kind == "tool_error":
            raise RuntimeError(payload[0])
        if kind != "tool_result":
            raise RuntimeError("invalid host-tool response")
        return payload[0]

    def configure(self, tool_names: list[str], output_fields: list[dict[str, Any]] | None) -> None:
        if any(not name.isidentifier() or keyword.iskeyword(name) or name == "SUBMIT" for name in tool_names):
            raise ValueError("tool names must be Python identifiers other than SUBMIT")
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

    def execute(
        self,
        code: str,
        variables: dict[str, Any],
        tool_names: list[str],
        output_fields: list[dict[str, Any]] | None,
    ) -> list[Any]:
        self.configure(tool_names, output_fields)
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
            return ["final", jsonable(submission.args[0])]
        except SyntaxError as exc:
            return ["syntax", str(exc)]
        except BaseException as exc:
            return ["execution_error", f"{type(exc).__name__}: {exc}"]
        return ["result", jsonable(value), captured.getvalue().rstrip("\n")]


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def send(message: list[Any]) -> None:
    print(json.dumps(message, separators=(",", ":"), allow_nan=False), file=sys.__stdout__, flush=True)


def receive() -> list[Any]:
    line = sys.__stdin__.readline()
    if not line:
        raise EOFError
    message = json.loads(line)
    if not isinstance(message, list) or not message or not isinstance(message[0], str):
        raise ValueError("protocol messages must be tagged lists")
    return message


def main() -> None:
    session = Session()
    send(["ready"])
    try:
        while True:
            request = receive()
            if request[0] == "shutdown":
                return
            if request[0] != "execute":
                raise ValueError("unknown request")
            send(session.execute(*request[1:]))
    except EOFError:
        return
    except BaseException as exc:
        send(["terminal_error", f"{type(exc).__name__}: {exc}"])


if __name__ == "__main__":
    main()
