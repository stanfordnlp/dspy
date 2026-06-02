from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dspy
from dspy.flex.primitives_doc import PRIMITIVES_CATALOG


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


class CodegenSignature(dspy.Signature):
    """Author the implementation of a dspy.Flex module from a user-declared Signature.

    Output TWO Python source strings:

    - ``predictors_src``: module-scope code that defines a single dict named
      ``PREDICTORS`` mapping attribute names to ``dspy.Predict`` /
      ``dspy.ChainOfThought`` / ``dspy.ReAct`` instances.
    - ``forward_src``: a single ``def forward(self, **inputs):`` whose body
      orchestrates the predictors (via ``self.<name>(...)``) and returns
      ``dspy.Prediction(<output fields>=...)`` matching the parent signature.

    You must follow the rules in ``primitives_catalog`` exactly.

    When ``seed_implementation`` is not ``'(none)'``, treat it as the starting
    point — it is an existing implementation (possibly hand-edited by the user)
    that must be adapted to the *current* signature. Preserve its structure,
    helper logic, and intent; change only what the current signature requires.
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
    seed_implementation: str = dspy.InputField(
        desc="An existing implementation to adapt (the current `predictors_src` and "
        "`forward_src`), or '(none)' for a fresh generation. When not '(none)', "
        "preserve its structure and intent and change only what the current "
        "signature requires."
    )
    predictors_src: str = dspy.OutputField(
        desc="Python source defining `PREDICTORS = {...}` at module scope."
    )
    forward_src: str = dspy.OutputField(
        desc="Python source defining `def forward(self, **inputs):` that returns dspy.Prediction(...)."
    )


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def generate(
    ctx: FlexContext,
    *,
    lm: dspy.LM | None = None,
    seed: tuple[str, str] | None = None,
) -> tuple[str, str]:
    """Run the codegen LM against ``ctx`` and return ``(predictors_src, forward_src)``.

    When ``seed`` is given as ``(predictors_src, forward_src)``, the codegen LM is
    asked to *adapt* that existing implementation to the current signature rather
    than author one from scratch. Used to carry hand-edited code forward across a
    signature change.
    """
    predictor = dspy.Predict(CodegenSignature)
    if seed is not None:
        seed_predictors, seed_forward = seed
        seed_text = (
            "# Existing predictors_src:\n"
            + seed_predictors.strip()
            + "\n\n# Existing forward_src:\n"
            + seed_forward.strip()
        )
    else:
        seed_text = "(none)"
    inputs = dict(
        signature_spec=ctx.render_signature_spec(),
        context_blurb=ctx.render_context_blurb(),
        primitives_catalog=PRIMITIVES_CATALOG,
        seed_implementation=seed_text,
    )
    if lm is not None:
        with dspy.context(lm=lm):
            out = predictor(**inputs)
    else:
        out = predictor(**inputs)

    return _strip_code_fences(out.predictors_src), _strip_code_fences(out.forward_src)
