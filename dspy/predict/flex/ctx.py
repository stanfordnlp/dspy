from __future__ import annotations

import ast
import types
import typing
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin


@dataclass
class FlexContext:
    """Signature + tools that the baseline and code-optimization prompts render from."""

    signature_cls: type
    tools: list[Any] = field(default_factory=list)

    def context_names(self) -> dict[str, Any]:
        """Names to inject into the exec globals for the generated code."""
        out: dict[str, Any] = {}
        for tool in self.tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if not name or not name.isidentifier():
                raise ValueError(
                    f"Tool {tool!r} needs a name that is a valid Python identifier — it is "
                    f"referenced by name in the generated code — but got {name!r}. Use a `def` "
                    f"function or `dspy.Tool(func, name='my_tool')`."
                )
            if name == "dspy" or name.startswith(("_dspy", "__dspy")):
                raise ValueError(
                    f"Tool name {name!r} is reserved: 'dspy' and names starting with '_dspy' belong "
                    f"to the dspy.Flex sandbox. Rename it, e.g. `dspy.Tool(func, name='my_tool')`."
                )
            out[name] = tool
        return out

    def render_signature_spec(self) -> str:
        cls = self.signature_cls
        name = getattr(cls, "__name__", "AnonymousSignature")
        objective = getattr(cls, "instructions", "") or ""

        def _fmt(fields_dict: dict[str, Any]) -> str:
            lines: list[str] = []
            for fname, finfo in fields_dict.items():
                type_str = _type_name(finfo.annotation)
                extra = finfo.json_schema_extra or {}
                desc = extra.get("desc", "")
                line = f"  - {fname}: {type_str}"
                if desc and not desc.startswith("${"):
                    line += f"  -- {desc}"
                lines.append(line)
            return "\n".join(lines) if lines else "  (none)"

        return (
            f"Signature: {name}\n"
            f"Objective (docstring): {objective}\n"
            f"Input fields:\n{_fmt(cls.input_fields)}\n"
            f"Output fields:\n{_fmt(cls.output_fields)}\n"
        )

    def render_signature_string(self) -> str:
        """Render a parseable ``"in: T, in2 -> out: T2"`` string for the baseline."""
        cls = self.signature_cls
        custom = self.custom_types()

        def _render(fields_dict: dict[str, Any]) -> str:
            parts: list[str] = []
            for fname, finfo in fields_dict.items():
                try:
                    parts.append(f"{fname}: {render_annotation(finfo.annotation, custom)}")
                except ValueError:
                    parts.append(fname)
            return ", ".join(parts)

        return f"{_render(cls.input_fields)} -> {_render(cls.output_fields)}"

    def custom_types(self) -> dict[str, type]:
        cls = self.signature_cls
        out: dict[str, type] = {}
        for finfo in {**cls.input_fields, **cls.output_fields}.values():
            _collect_custom_types(finfo.annotation, out)
        return out

    def render_context_blurb(self, sandboxed: bool = False) -> str:
        parts: list[str] = []
        if self.tools:
            tool_lines: list[str] = []
            for tool in self.tools:
                tname = getattr(tool, "name", None) or getattr(tool, "__name__", "?")
                tdesc = getattr(tool, "desc", None) or getattr(tool, "__doc__", "") or ""
                tdesc = tdesc.strip().splitlines()[0] if tdesc else ""
                tool_lines.append(f"  - {tname}: {tdesc}")
            parts.append("Available tools (in scope by name):\n" + "\n".join(tool_lines))
        if sandboxed:
            parts.append(
                "This module runs in a sandbox: only the tools listed above may be passed to "
                "dspy.ReAct/dspy.RLM (their functions live on the host). A function you define inside "
                "the module can be called directly in forward, but cannot be handed to those predictors."
            )
        return "\n\n".join(parts) if parts else "(no extra context)"


def _type_name(t: Any) -> str:
    if t is None:
        return "None"
    name = getattr(t, "__name__", None)
    if name:
        return name
    return str(t).replace("typing.", "")


def _collect_custom_types(annotation: Any, out: dict[str, type]) -> None:
    """Add every non-builtin type named anywhere in ``annotation`` to ``out``, keyed by name."""
    for arg in get_args(annotation):
        _collect_custom_types(arg, out)
    if get_origin(annotation) is not None or annotation is type(None):
        return
    name = getattr(annotation, "__name__", None)
    if isinstance(annotation, type) and name and name.isidentifier() and annotation.__module__ != "builtins":
        out[name] = annotation


_BUILTIN_TYPES = (int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray)


def render_annotation(annotation: Any, custom_types: dict[str, type] | None = None) -> str:
    """Render ``annotation`` as a signature-string type token, the inverse of dspy's
    ``_parse_type_node``."""
    return ast.unparse(_render_type_node(annotation, custom_types or {}))


def _render_type_node(annotation: Any, custom_types: dict[str, type]) -> ast.expr:
    """Build the annotation AST for ``annotation``, emitting only nodes ``_parse_type_node`` handles."""
    if annotation is type(None):
        return ast.Constant(value=None)

    origin = get_origin(annotation)
    if origin is None:
        name = getattr(annotation, "__name__", None)
        if annotation in _BUILTIN_TYPES:
            return ast.Name(id=annotation.__name__)
        if annotation is Any or (name and typing.__dict__.get(name) is annotation):
            return ast.Name(id=name or "Any")
        if isinstance(annotation, type) and name and custom_types.get(name) is annotation:
            return ast.Name(id=name)
        raise ValueError(
            f"Annotation {annotation!r} has no name the signature-string grammar can resolve; "
            "pass it via custom_types to make it renderable."
        )

    args = get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        node = _render_type_node(args[0], custom_types)
        for arg in args[1:]:
            node = ast.BinOp(left=node, op=ast.BitOr(), right=_render_type_node(arg, custom_types))
        return node
    if origin is typing.Literal:
        if not all(v is None or isinstance(v, (str, int, bool, bytes)) for v in args):
            raise ValueError(f"Literal values in {annotation!r} are not signature-string constants.")
        elts: list[ast.expr] = [ast.Constant(value=v) for v in args]
        return ast.Subscript(value=ast.Name(id="Literal"), slice=ast.Tuple(elts=elts) if len(elts) > 1 else elts[0])

    base = _render_type_node(origin, custom_types)
    elts = [ast.Constant(value=...) if a is Ellipsis else _render_type_node(a, custom_types) for a in args]
    return ast.Subscript(value=base, slice=ast.Tuple(elts=elts) if len(elts) > 1 else elts[0])


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    # Normalize tabs so exec doesn't choke on mixed tabs/spaces from the LM's output.
    return s.strip().expandtabs(4)
