from __future__ import annotations

import ast
import types
import typing
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from dspy.adapters.types.decision import Choice, Noul, Score, _Decision, decision_type


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
        if self.decision_outputs():
            parts.append(DECISION_NOTE)
        if sandboxed:
            parts.append(
                "This module runs in a sandbox: only the tools listed above may be passed to "
                "dspy.ReActV2/dspy.RLM (their functions live on the host). A function you define inside "
                "the module can be called directly in forward, but cannot be handed to those predictors."
            )
        return "\n\n".join(parts) if parts else "(no extra context)"


    def decision_outputs(self) -> list[str]:
        """The signature's outputs declared with a decision type (``Noul``, ``Score``, ``Choice``)."""
        return [
            name
            for name, finfo in self.signature_cls.output_fields.items()
            if decision_type(finfo) is not None
            and any(isinstance(a, type) and issubclass(a, _Decision) for a in (finfo.annotation, *finfo.metadata))
        ]


DECISION_NOTE = """\
Decision outputs: this signature declares outputs with decision types, whose values are decided from
probabilities rather than generated. Sub-signature strings can declare them too:
`"ticket: str -> duplicate: Noul[(True, 'Repeats an open ticket'), (False, 'New issue')]"`,
`"ticket -> severity: Score['minor', 'major', 'critical']"` (ordered lowest to highest), or
`"ticket -> team: Choice[('billing', 'Payment issue'), ('tech', 'Product bug')]"`. A bare `bool` output
works as `Noul`. Each decision output needs a question: set it on the predictor in `__init__`, as in
`self.check.fields["duplicate"] = {"instructions": "Does this ticket repeat an open one?"}`. The result is a
decision object: `bool(out.duplicate)` and `out.duplicate.value` for a Noul, `float(out.severity)` and
`out.severity.level` for a Score,
`out.team.value` for a Choice; `.confidence` and `.probability`/`.probabilities` carry the evidence.
The thresholds, cuts, and weights that turn probabilities into values are calibrated outside this code."""


def _render_decision(annotation: type) -> ast.expr:
    """``Noul[...]``, ``Score[...]``, or ``Choice[...]`` for a decision type, from its declared criteria."""
    criteria = annotation.criteria()
    if issubclass(annotation, Noul):
        base = "Noul"
        pairs = [(value, criteria[str(value).lower()]) for value in (True, False) if criteria and str(value).lower() in criteria]
    elif issubclass(annotation, Score):
        base, pairs = "Score", criteria
    elif issubclass(annotation, Choice):
        base = "Choice"
        values = get_args(annotation.model_fields["value"].annotation) if criteria is not None else ()
        pairs = [(value, criteria[str(value)]) for value in values]
    else:
        raise ValueError(f"Unsupported decision type {annotation!r}.")
    if not pairs:
        return ast.Name(id=base)

    def constant(value: Any) -> ast.expr:
        if value is not None and not isinstance(value, (str, int, bool)):
            raise ValueError(f"Decision criteria in {annotation!r} are not signature-string constants.")
        return ast.Constant(value=value)

    elts = [
        constant(pair) if base == "Score" else ast.Tuple(elts=[constant(pair[0]), constant(pair[1])]) for pair in pairs
    ]
    return ast.Subscript(value=ast.Name(id=base), slice=ast.Tuple(elts=elts) if len(elts) > 1 else elts[0])


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

    if isinstance(annotation, type) and issubclass(annotation, _Decision):
        return _render_decision(annotation)

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
