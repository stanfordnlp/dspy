"""Canonical tool-call data type and inbound wire-shape boundary.

Lives at the ``clients`` layer (the LiteLLM wrapper) so the dependency
direction is ``adapters → clients``, never the reverse. Outbound (DSPy
``Tool`` → wire shape) is handled by ``Tool.format_as_litellm_function_call``
parameterized by ``model_type``.
"""

import inspect
from typing import Any

import json_repair
import pydantic


class ToolCall(pydantic.BaseModel):
    name: str
    args: dict[str, Any]
    id: str | None = None

    def format(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "function",
            "function": {"name": self.name, "arguments": self.args},
        }
        if self.id is not None:
            payload["id"] = self.id
        return payload

    def execute(self, functions: Any = None) -> Any:
        """Execute this tool call.

        ``functions`` may be a ``{name: callable}`` dict, a list of objects
        with ``.name`` and ``.func`` attributes (e.g. ``dspy.Tool``), or
        ``None`` to look the name up in the caller's locals/globals.
        """
        func = None

        if functions is None:
            frame = inspect.currentframe().f_back
            try:
                func = frame.f_locals.get(self.name) or frame.f_globals.get(self.name)
            finally:
                del frame
        elif isinstance(functions, dict):
            func = functions.get(self.name)
        elif isinstance(functions, list):
            for tool in functions:
                if tool.name == self.name:
                    func = tool.func
                    break

        if func is None:
            raise ValueError(
                f"Tool function '{self.name}' not found. "
                "Please pass the tool functions to the `execute` method."
            )

        try:
            return func(**(self.args or {}))
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{self.name}': {e}") from e


def to_tool_call(item: Any) -> ToolCall:
    """Normalize a LiteLLM tool-call into a canonical ``ToolCall``.

    Single inbound boundary for wire-shape coercion. Falls back to attribute
    access when ``model_dump()`` raises ``TypeError`` because of the
    MockValSer/SchemaSerializer bug (pydantic#7713, litellm#9345).
    """
    if not isinstance(item, dict) and hasattr(item, "model_dump"):
        try:
            item = item.model_dump()
        except TypeError:
            fn = getattr(item, "function", None)
            if fn is not None:
                return ToolCall(name=fn.name, args=_parse_args(fn.arguments), id=getattr(item, "id", None))
            if getattr(item, "name", None) is None:
                raise
            return ToolCall(
                name=item.name,
                args=_parse_args(getattr(item, "arguments", None)),
                id=getattr(item, "call_id", None) or getattr(item, "id", None),
            )

    if not isinstance(item, dict):
        raise TypeError(f"Cannot normalize tool call from {type(item).__name__}: {item!r}")

    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        fn = item["function"]
        return ToolCall(name=fn["name"], args=_parse_args(fn.get("arguments")), id=item.get("id"))

    if item.get("type") == "function_call" and item.get("name"):
        return ToolCall(
            name=item["name"],
            args=_parse_args(item.get("arguments")),
            id=item.get("call_id") or item.get("id"),
        )

    raise ValueError(f"Unknown tool-call shape: {item!r}")


def _parse_args(args: Any) -> dict[str, Any]:
    if args is None or args == "":
        return {}
    return json_repair.loads(args) if isinstance(args, str) else args
