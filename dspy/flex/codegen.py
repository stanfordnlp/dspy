from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

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
      ``dspy.ChainOfThought`` / ``dspy.ReAct`` instances. Use a predictor ONLY
      for a step that genuinely needs a language model (extraction,
      classification, generation, open-ended reasoning). If the whole task is
      deterministic, emit ``PREDICTORS = {}`` and do all the work in Python.
    - ``forward_src``: a single ``def forward(self, **inputs):`` whose body may
      call the predictors (via ``self.<name>(...)``) AND/OR run arbitrary
      deterministic Python — parsing, arithmetic, regex, string and
      data-structure manipulation, and small nested ``def`` helpers defined
      inside ``forward`` — and returns ``dspy.Prediction(<output fields>=...)``
      matching the parent signature.

    You MUST follow the rules and patterns in ``primitives_catalog`` exactly.
    In particular: every predictor call returns a ``dspy.Prediction`` — read
    its declared output fields off as attributes (``result.foo``) before
    composing them or building the final return. Never pass a whole
    ``dspy.Prediction`` as the value of an output field. Coerce values to the
    declared output type (``int(...)``, ``float(...)``, etc.) before returning.

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
        desc="Python source defining `PREDICTORS = {...}` at module scope (use `PREDICTORS = {}` when no LM call is needed)."
    )
    forward_src: str = dspy.OutputField(
        desc="Python source defining `def forward(self, **inputs):` that returns dspy.Prediction(...). May contain deterministic Python and nested helpers."
    )


class RepairSignature(dspy.Signature):
    """Fix a broken dspy.Flex implementation.

    The user (or a previous codegen run) produced a ``predictors_src`` /
    ``forward_src`` pair that either fails to bind or raises at runtime. Your
    job is to produce a corrected pair that satisfies the parent Signature.

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
    broken_predictors_src: str = dspy.InputField(desc="The current (broken) predictors_src.")
    broken_forward_src: str = dspy.InputField(desc="The current (broken) forward_src.")
    failure_kind: str = dspy.InputField(
        desc="Either 'bind' (broken at import/exec) or 'runtime' (raised while forward() was running)."
    )
    error_text: str = dspy.InputField(
        desc="The exception class and message (and a short traceback when available)."
    )
    predictors_src: str = dspy.OutputField(
        desc="Corrected Python source defining `PREDICTORS = {...}` at module scope."
    )
    forward_src: str = dspy.OutputField(
        desc="Corrected Python source defining `def forward(self, ...)` returning dspy.Prediction(...)."
    )


class FlexIntentError(Exception):
    """Raised when a Flex Signature's intent is too vague to implement well.

    Surfaced before any code is generated. Carries the clarity judge's
    ``concern`` and a concrete ``clarifying_question`` to answer in the
    Signature docstring, and renders an actionable message.
    """

    def __init__(self, flex_id: str, concern: str, clarifying_question: str):
        self.flex_id = flex_id
        self.concern = concern
        self.clarifying_question = clarifying_question
        super().__init__(self._render())

    def _render(self) -> str:
        concern = self.concern or "the mapping from inputs to outputs is underdetermined."
        question = self.clarifying_question or (
            "What exactly should this module do — define the inputs→outputs behavior precisely?"
        )
        return (
            f"dspy.Flex {self.flex_id!r}: the Signature's intent is too ambiguous to "
            f"generate a reliable implementation.\n\n"
            f"Concern: {concern}\n\n"
            f"To fix, expand the Signature docstring to answer this question, then re-run:\n"
            f"  > {question}\n\n"
            f'(Pass intent_check="warn" to generate a best-effort module instead of raising, '
            f'or intent_check="off" to skip this check entirely.)'
        )


class IntentClaritySignature(dspy.Signature):
    """Judge whether a dspy.Flex module's Signature is specified clearly enough to
    implement well — BEFORE any code is written.

    You are given the user's Signature (name, objective docstring, input and
    output fields) and any extra context. Decide whether a competent engineer
    could implement the intended input→output behavior correctly from this alone.

    Judge INTENT clarity only — NOT whether the task needs a language model. A
    purely deterministic task (arithmetic, parsing, formatting, sorting) can be
    perfectly 'clear'. Genuine ambiguity means: the mapping from inputs to
    outputs is underdetermined, key terms/units/formats are undefined, important
    edge cases are unspecified, or the objective is missing or self-contradictory.
    Do NOT nitpick — only flag ambiguity that would actually change the
    implementation.

    Verdict values:
    - 'clear': unambiguous enough to implement well.
    - 'underspecified': implementable, but one or more decisions are guesses a
      short docstring note should pin down.
    - 'insufficient': too vague to implement responsibly without guessing the
      core behavior.

    When the verdict is not 'clear', give a SHORT concrete `concern` and exactly
    ONE specific `clarifying_question` the user should answer in the docstring to
    remove the ambiguity. When 'clear', leave both empty.
    """

    signature_spec: str = dspy.InputField(
        desc="Rendered description of the user's Signature: name, objective docstring, input and output fields."
    )
    context_blurb: str = dspy.InputField(
        desc="Optional extra context: tools available by name, style notes. May be '(no extra context)'."
    )
    verdict: str = dspy.OutputField(
        desc="Exactly one of: clear, underspecified, insufficient."
    )
    concern: str = dspy.OutputField(
        desc="One short sentence naming the ambiguity, or empty when verdict is 'clear'."
    )
    clarifying_question: str = dspy.OutputField(
        desc="One concrete question to answer in the Signature docstring, or empty when 'clear'."
    )


class IntentAssessment(NamedTuple):
    """Result of :func:`assess_intent`."""

    verdict: str  # "clear" | "underspecified" | "insufficient"
    concern: str
    clarifying_question: str


_VALID_VERDICTS = ("insufficient", "underspecified", "clear")


def assess_intent(ctx: FlexContext, *, lm: dspy.LM | None = None) -> IntentAssessment:
    """Ask the codegen LM whether ``ctx``'s Signature is clear enough to implement.

    Degrades gracefully: any LM/parse failure, or an unrecognized verdict, is
    treated as ``'clear'`` so a flaky judge never blocks code generation. The
    verdict is matched leniently (most-severe-first) so stray punctuation or
    surrounding words don't downgrade an 'insufficient' result.
    """
    predictor = dspy.Predict(IntentClaritySignature)
    inputs = {
        "signature_spec": ctx.render_signature_spec(),
        "context_blurb": ctx.render_context_blurb(),
    }
    try:
        if lm is not None:
            with dspy.context(lm=lm):
                out = predictor(**inputs)
        else:
            out = predictor(**inputs)
        raw = (out.verdict or "").strip().lower()
        verdict = next((v for v in _VALID_VERDICTS if v in raw), None)
        if verdict is None:
            logger.debug("assess_intent: unrecognized verdict %r; treating as 'clear'.", raw)
            return IntentAssessment("clear", "", "")
        return IntentAssessment(
            verdict, (out.concern or "").strip(), (out.clarifying_question or "").strip()
        )
    except Exception as err:  # a flaky judge must never block codegen
        logger.debug("assess_intent: judge failed (%s); treating as 'clear'.", err)
        return IntentAssessment("clear", "", "")


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


def repair(
    ctx: FlexContext,
    *,
    broken: tuple[str, str],
    failure_kind: str,
    error_text: str,
    lm: dspy.LM | None = None,
) -> tuple[str, str]:
    """Ask the codegen LM to fix a broken ``(predictors_src, forward_src)`` pair.

    ``failure_kind`` is ``'bind'`` (the code didn't import/exec cleanly) or
    ``'runtime'`` (the code bound but ``forward()`` raised). ``error_text``
    should include the exception class and message.
    """
    broken_predictors, broken_forward = broken
    predictor = dspy.Predict(RepairSignature)
    inputs = dict(
        signature_spec=ctx.render_signature_spec(),
        context_blurb=ctx.render_context_blurb(),
        primitives_catalog=PRIMITIVES_CATALOG,
        broken_predictors_src=broken_predictors.strip(),
        broken_forward_src=broken_forward.strip(),
        failure_kind=failure_kind,
        error_text=error_text,
    )
    if lm is not None:
        with dspy.context(lm=lm):
            out = predictor(**inputs)
    else:
        out = predictor(**inputs)

    return _strip_code_fences(out.predictors_src), _strip_code_fences(out.forward_src)
