"""A signature as Python source, written as declared so a user can paste it into their code."""

from typing import Literal, get_origin

from dspy.adapters.types.decision import Choice, Noul, Score, decision_type
from dspy.adapters.utils import get_annotation_name
from dspy.predict.flex.ctx import render_annotation

LINE = 96  # a rubric annotation longer than this is written one option per line


def _desc(field) -> str:
    """A field's description, empty when DSPy back-filled the `${name}` placeholder."""
    d = (field.json_schema_extra or {}).get("desc") or ""
    return "" if d.startswith("${") else d


def _head(kind: type) -> str:
    return "Noul" if issubclass(kind, Noul) else "Score" if issubclass(kind, Score) else "Choice"


def _is_rich(field) -> bool:
    return isinstance(field.annotation, type) and issubclass(field.annotation, (Noul, Score, Choice))


def _native_source(field) -> str:
    """`bool`, or the `Literal[...]` a native decision output is annotated with."""
    if get_origin(field.annotation) is Literal:
        return render_annotation(field.annotation)
    return field.annotation.__name__


def _bare_metadata(field) -> str | None:
    """The bare `Noul` or `Choice` a native output carries as metadata, which marks it for probabilities."""
    for m in field.metadata:
        if isinstance(m, type) and issubclass(m, (Noul, Choice)) and not m.options:
            return _head(m)
    return None


def _rubric_source(field, kind: type, indent: int) -> str | None:
    """The `Noul[...]`, `Score[...]`, or `Choice[...]` the type declares, or None when it declares no descriptions."""
    head = _head(kind)
    if head == "Noul":
        if not kind.options:
            return None
        items = [repr((v, d)) for want in (True, False) for v, d in kind.options if v is want]
    elif head == "Score":
        items = [repr(c) for c in kind.options]
    else:
        if not _is_rich(field) and not any(d for _, d in kind.options):
            return None
        items = [repr((v, d or "")) for v, d in kind.options]
    flat = f"{head}[{', '.join(items)}]"
    if len(flat) <= LINE:
        return flat
    pad = " " * (indent + 4)
    return f"{head}[\n" + "".join(f"{pad}{item},\n" for item in items) + " " * indent + "]"


def annotation_source(field, indent: int = 4) -> str:
    """The field's annotation as source, with its type's criteria written out."""
    kind = decision_type(field)
    if kind is None:
        try:
            return render_annotation(field.annotation)
        except ValueError:
            return get_annotation_name(field.annotation)
    rubric = _rubric_source(field, kind, indent)
    if _is_rich(field):
        return rubric or _head(kind)
    native = _native_source(field)
    marker = rubric or _bare_metadata(field)
    return f"Annotated[{native}, {marker}]" if marker else native


def _render_docstring(text: str) -> str:
    """The docstring as a triple-quoted line, with every backslash and quote escaped so it stays valid Python."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'    """{escaped}"""'


def _field_call(kind: str, desc: str) -> str:
    if not desc:
        return f"dspy.{kind}()"
    if len(desc) < 70:
        return f"dspy.{kind}(desc={desc!r})"
    return f"dspy.{kind}(\n        desc={desc!r},\n    )"


def render_signature(sig, name: str | None = None) -> str:
    """The signature as declared, as a class a user could paste into their code."""
    lines = [f"class {name or sig.__name__}(dspy.Signature):"]
    if sig.instructions and not sig.instructions.startswith("Given the fields"):
        lines.append(_render_docstring(sig.instructions))
    lines.append("")
    for kind, fields in (("InputField", sig.input_fields), ("OutputField", sig.output_fields)):
        for fname, f in fields.items():
            lines.append(f"    {fname}: {annotation_source(f)} = {_field_call(kind, _desc(f))}")
    return "\n".join(lines) + "\n"
