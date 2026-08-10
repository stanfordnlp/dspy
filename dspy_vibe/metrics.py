"""Metrics for optimizing and grading the pipeline.

These are DSPy metrics: `metric(example, prediction, trace=None) -> float`.
They are deliberately mechanical. A metric an LM judges is a metric that can be
talked into a good score; every check here is something a script can settle.

The bias is toward penalizing *invention*. A thin brief wastes a little time; a
brief that adds requirements the user never stated sends work in the wrong
direction and looks authoritative while doing it.
"""

from __future__ import annotations

import re
from typing import Any

from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeSpec

STOPWORDS = {
    "a", "az", "egy", "és", "es", "hogy", "kell", "legyen", "lehet", "csak", "meg", "majd", "is",
    "the", "and", "for", "with", "that", "this", "should", "would", "make", "just", "some", "very",
}


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-zá-űA-ZÁ-Ű0-9_.\-]{3,}", text.lower())
        if word not in STOPWORDS
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def spec_faithfulness(instruction: str, spec: VibeSpec) -> float:
    """Fraction of scope claims that are anchored in the instruction.

    A claim counts as anchored when it shares any content word with the source.
    The bar is deliberately low: the question is "did the converter invent a
    piece of work nobody asked for", not "did it paraphrase". Requiring heavy
    overlap would score a copy-paste brief perfectly and punish the decomposed
    one — the opposite of what this metric is for.

    Constraints, risks, and open questions are exempt. Naming a risk the user
    never mentioned is the converter doing its job.
    """
    source = _tokens(instruction) | _tokens(spec.context)
    claims = [*spec.scope, *spec.non_goals, spec.goal]
    checkable = [claim for claim in claims if _tokens(claim)]
    if not checkable:
        return 0.0
    anchored = sum(1 for claim in checkable if _tokens(claim) & source)
    return _clamp(anchored / len(checkable))


def spec_completeness(spec: VibeSpec) -> float:
    """Whether the brief carries the parts that make it usable."""
    checks = [
        bool(spec.goal.strip()),
        bool(spec.scope),
        bool(spec.acceptance),
        all(criterion.verification.strip() for criterion in spec.acceptance),
        bool(spec.open_questions) or spec.confidence == "HIGH",
        bool(spec.risks) or spec.confidence == "HIGH",
    ]
    return sum(1 for check in checks if check) / len(checks)


def spec_honesty(spec: VibeSpec) -> float:
    """Whether unresolved decisions are surfaced instead of silently made.

    An open question with no assumption is fine — it is marked blocking. An
    assumption stated as if it were a requirement is the failure this catches.
    """
    if not spec.open_questions:
        # No questions at all is only credible for a fully specified request.
        return 1.0 if spec.confidence == "HIGH" else 0.4
    documented = sum(
        1
        for question in spec.open_questions
        if question.why_it_matters.strip() and (question.assumption_used.strip() or question.blocking)
    )
    return documented / len(spec.open_questions)


CONCRETE_VERIFICATION = re.compile(
    r"\b(run|runs|execute|command|test|tests|pytest|npm|yarn|curl|assert|lint|build|"
    r"measure|profile|screenshot|open|click|query|inspect|benchmark|fut|futtat|teszt|mér)\b",
    re.I,
)


def spec_specificity(spec: VibeSpec) -> float:
    """Whether the brief decomposed the request or merely echoed it.

    A converter that copies the instruction into `scope` and appends
    "is implemented" to each line has produced a structurally complete but
    useless brief. These checks are what an LM-backed pipeline should win on,
    so the deterministic baseline deliberately does not score full marks here.
    """
    checks: list[bool] = []

    # A goal is a single claim, not the whole request pasted back.
    checks.append(bool(spec.goal.strip()) and len(spec.goal.split()) <= 25)

    # Scope items are atomic units of work, not copied paragraphs.
    checks.append(bool(spec.scope) and all(len(item.split()) <= 15 for item in spec.scope))

    # No acceptance criterion may be a scope line with "is implemented" glued on.
    scope_tokens = [_tokens(item) for item in spec.scope]
    templated = sum(
        1
        for criterion in spec.acceptance
        if (statement := _tokens(criterion.statement))
        and any(statement <= source | {"implemented", "done", "works"} for source in scope_tokens)
    )
    checks.append(bool(spec.acceptance) and templated == 0)

    # Verification has to name something a person can actually do, and the
    # methods must differ from each other.
    checks.append(
        bool(spec.acceptance)
        and all(CONCRETE_VERIFICATION.search(criterion.verification) for criterion in spec.acceptance)
        and len({criterion.verification.strip().lower() for criterion in spec.acceptance}) == len(spec.acceptance)
    )

    # Open questions must be about *this* request, not generic boilerplate.
    source_tokens = _tokens(spec.source_instruction) | _tokens(spec.goal)
    if spec.open_questions:
        specific = sum(1 for question in spec.open_questions if _tokens(question.question) & source_tokens)
        checks.append(specific >= max(1, len(spec.open_questions) // 2))

    # Every stated constraint should be visible in the acceptance criteria,
    # otherwise it is decoration.
    if spec.constraints:
        acceptance_text = _tokens(" ".join(c.statement + " " + c.verification for c in spec.acceptance))
        covered = sum(1 for item in spec.constraints if _tokens(item) & acceptance_text)
        checks.append(covered >= max(1, len(spec.constraints) // 2))

    return sum(1 for check in checks if check) / len(checks)


def spec_quality(example: Any, prediction: Any, trace: Any = None) -> float:
    """Composite brief score in [0, 1]. Usable directly as a DSPy metric."""
    spec = getattr(prediction, "spec", None) or getattr(prediction, "bundle", None)
    if hasattr(spec, "spec"):
        spec = spec.spec
    if not isinstance(spec, VibeSpec):
        return 0.0
    instruction = getattr(example, "instruction", "") or spec.source_instruction
    # Faithfulness multiplies rather than adds. Copying the instruction verbatim
    # scores perfectly on it, so as a bonus it would reward the laziest possible
    # brief; as a gate it only punishes invention, which is what it is for.
    quality = 0.35 * spec_completeness(spec) + 0.45 * spec_specificity(spec) + 0.20 * spec_honesty(spec)
    return _clamp(quality * (0.4 + 0.6 * spec_faithfulness(instruction, spec)))


def agent_validity(agent: AgentArtifact, allowed_tools: list[str] | None = None) -> float:
    """Whether an agent definition is executable and bounded."""
    checks = [
        bool(agent.description.strip()),
        bool(agent.role.strip()),
        bool(agent.instructions),
        bool(agent.guardrails),
        bool(agent.success_criteria),
        not allowed_tools or set(agent.tools) <= set(allowed_tools),
    ]
    return sum(1 for check in checks if check) / len(checks)


def skill_validity(skill: SkillArtifact) -> float:
    """Whether a skill definition is reusable and honestly bounded."""
    checks = [
        bool(skill.description.strip()),
        bool(skill.triggers),
        len(skill.procedure) >= 2,
        bool(skill.outputs),
        bool(skill.checks),
        bool(skill.limits),
    ]
    return sum(1 for check in checks if check) / len(checks)


def bundle_quality(example: Any, prediction: Any, trace: Any = None) -> float:
    """Composite score over the whole bundle. Usable directly as a DSPy metric."""
    bundle = getattr(prediction, "bundle", None)
    if bundle is None:
        return 0.0
    allowed = getattr(example, "available_tools", None)
    return _clamp(
        0.5 * spec_quality(example, prediction, trace)
        + 0.25 * agent_validity(bundle.agent, allowed)
        + 0.25 * skill_validity(bundle.skill)
    )
