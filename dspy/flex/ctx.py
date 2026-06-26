from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, get_origin

import dspy
from dspy.flex.primitives_doc import PRIMITIVES_CATALOG

logger = logging.getLogger(__name__)


@dataclass
class FlexContext:
    """Bundle of inputs the codegen LM gets to author a Flex implementation."""

    signature_cls: type
    tools: list[Any] = field(default_factory=list)
    style_notes: list[str] = field(default_factory=list)

    def context_names(self) -> dict[str, Any]:
        """Names to inject into the exec globals for the generated code."""
        out: dict[str, Any] = {}
        for tool in self.tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if not name:
                raise ValueError(
                    f"Cannot determine a name for context tool {tool!r}. Use a "
                    f"`dspy.Tool` with `.name`, or a plain function with `__name__`."
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
        """Render a parseable ``"in: T, in2 -> out: T2"`` signature string.

        Used to construct the baseline ``dspy.RLM(<this string>)`` deterministically
        (no LM). ``Signature.signature`` is lossy (names only), so we emit types too —
        builtins and typing generics are included; an unrecognized/custom annotation is
        emitted untyped (defaults to ``str`` when parsed), which keeps the baseline
        constructible rather than failing on a type name the parser can't resolve.
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
        sections: list[str] = []
        if self.tools:
            tool_lines: list[str] = []
            for tool in self.tools:
                tname = getattr(tool, "name", None) or getattr(tool, "__name__", "?")
                tdesc = getattr(tool, "desc", None) or getattr(tool, "__doc__", "") or ""
                tdesc = (tdesc or "").strip().splitlines()[0] if tdesc else ""
                tool_lines.append(f"  - {tname}: {tdesc}")
            sections.append("Available tools (in scope by name):\n" + "\n".join(tool_lines))
        if self.style_notes:
            sections.append("Style notes:\n" + "\n".join(f"  - {n}" for n in self.style_notes))
        return "\n\n".join(sections) if sections else "(no extra context)"


def _type_name(t: Any) -> str:
    if t is None:
        return "None"
    name = getattr(t, "__name__", None)
    if name:
        return name
    return str(t).replace("typing.", "")


_SIMPLE_TYPES = (str, int, float, bool, list, dict, tuple, set)


def _parseable_type_str(annotation: Any) -> str | None:
    """Return a signature-string type token that DSPy's parser can resolve, or None.

    Builtins map to their name (``int``); typing generics (``list[str]``) map to their
    string form with the ``typing.`` prefix stripped. Anything else (custom classes,
    Pydantic models not in the parser's namespace) returns None → render the field
    untyped so baseline construction never fails on an unresolvable type name.
    """
    if annotation in _SIMPLE_TYPES:
        return annotation.__name__
    if get_origin(annotation) is not None:
        return str(annotation).replace("typing.", "")
    return None


class RepairSignature(dspy.Signature):
    """Fix a broken dspy.Flex implementation.

    The user (or a previous codegen run) produced a ``module_src`` (a single
    ``dspy.Module`` subclass) that either fails to bind or raises at runtime.
    Your job is to produce a corrected module class that satisfies the parent
    Signature.

    Preserve as much of the broken implementation's structure and intent as
    possible — touch only what's needed to make the error go away. Honor the
    rules in ``primitives_catalog``.

    Common runtime bugs to check for when diagnosing ``error_text``:

    - ``AttributeError: '...' object has no attribute 'X'`` on a predictor's
      sub-signature field: the predictor's signature string declares
      different output names than the code reads. Either fix the signature
      string or fix the attribute access.
    - The final return is ``dspy.Prediction(field=<Prediction>)`` instead of
      ``dspy.Prediction(field=<Prediction>.field)``. Unwrap the inner
      ``dspy.Prediction`` by reading its declared output attribute.
    - ``TypeError``/``ValueError`` when constructing the final
      ``dspy.Prediction``: the value's type doesn't match the declared
      output type. Add an explicit cast (``int(...)``, ``float(...)``, etc.).
    """

    signature_spec: str = dspy.InputField(
        desc="Rendered description of the user's Signature: name, objective docstring, input and output fields."
    )
    context_blurb: str = dspy.InputField(
        desc="Optional extra context: tools available by name, style notes. May be '(no extra context)'."
    )
    primitives_catalog: str = dspy.InputField(
        desc="Catalog of DSPy primitives and conventions the generated code should follow."
    )
    broken_module_src: str = dspy.InputField(desc="The current (broken) module class source.")
    failure_kind: str = dspy.InputField(
        desc="Either 'bind' (broken at import/exec) or 'runtime' (raised while forward() was running)."
    )
    error_text: str = dspy.InputField(
        desc="The exception class and message (and a short traceback when available)."
    )
    module_src: str = dspy.OutputField(
        desc="Corrected Python source defining ONE `dspy.Module` subclass (with `__init__` and "
        "`forward(self, **inputs)` returning dspy.Prediction(...))."
    )


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    # Normalize tabs to spaces: the persistence round-trip indents/dedents these
    # source regions with textwrap, which mishandles mixed tabs and spaces.
    return s.strip().expandtabs(4)


def repair(
    ctx: FlexContext,
    *,
    broken: str,
    failure_kind: str,
    error_text: str,
    lm: dspy.LM | None = None,
) -> str:
    """Ask the codegen LM to fix a broken ``module_src`` (a dspy.Module subclass).

    ``failure_kind`` is ``'bind'`` (the code didn't import/exec cleanly) or
    ``'runtime'`` (the code bound but ``forward()`` raised). ``error_text``
    should include the exception class and message.
    """
    predictor = dspy.Predict(RepairSignature)
    inputs = dict(
        signature_spec=ctx.render_signature_spec(),
        context_blurb=ctx.render_context_blurb(),
        primitives_catalog=PRIMITIVES_CATALOG,
        broken_module_src=(broken or "").strip(),
        failure_kind=failure_kind,
        error_text=error_text,
    )
    if lm is not None:
        with dspy.context(lm=lm):
            out = predictor(**inputs)
    else:
        out = predictor(**inputs)

    return _strip_code_fences(out.module_src)
