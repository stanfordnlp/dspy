from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_origin


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
        """Render a parseable ``"in: T, in2 -> out: T2"`` string for the baseline ``dspy.RLM``.

        Emits types (unlike the names-only ``Signature.signature``); unresolvable annotations
        are emitted untyped so baseline construction never fails on them.
        """
        cls = self.signature_cls

        def _render(fields_dict: dict[str, Any]) -> str:
            parts: list[str] = []
            for fname, finfo in fields_dict.items():
                type_str = _parseable_type_str(finfo.annotation)
                parts.append(f"{fname}: {type_str}" if type_str else fname)
            return ", ".join(parts)

        return f"{_render(cls.input_fields)} -> {_render(cls.output_fields)}"

    def render_context_blurb(self) -> str:
        if not self.tools:
            return "(no extra context)"
        tool_lines: list[str] = []
        for tool in self.tools:
            tname = getattr(tool, "name", None) or getattr(tool, "__name__", "?")
            tdesc = getattr(tool, "desc", None) or getattr(tool, "__doc__", "") or ""
            tdesc = tdesc.strip().splitlines()[0] if tdesc else ""
            tool_lines.append(f"  - {tname}: {tdesc}")
        return "Available tools (in scope by name):\n" + "\n".join(tool_lines)


def _type_name(t: Any) -> str:
    if t is None:
        return "None"
    name = getattr(t, "__name__", None)
    if name:
        return name
    return str(t).replace("typing.", "")


_SIMPLE_TYPES = (str, int, float, bool, list, dict, tuple, set)


def _parseable_type_str(annotation: Any) -> str | None:
    """A signature-string type token DSPy's parser can resolve (builtin or typing generic), else None."""
    if annotation in _SIMPLE_TYPES:
        return annotation.__name__
    if get_origin(annotation) is not None:
        return str(annotation).replace("typing.", "")
    return None


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
