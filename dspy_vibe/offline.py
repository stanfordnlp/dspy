"""Deterministic, LM-free conversion.

Two reasons this exists. First, a brief you can produce without a configured LM
makes the pipeline testable and usable offline. Second, it is the baseline an
optimizer measures against: if the LM-backed module cannot beat plain string
handling, the LM is not earning its cost.

The heuristics are deliberately shallow and say so — every offline spec is
`LOW` confidence and carries the open questions the text left unanswered.
"""

from __future__ import annotations

import re

from dspy_vibe.types import (
    AcceptanceCriterion,
    AgentArtifact,
    OpenQuestion,
    SkillArtifact,
    VibeBundle,
    VibeSpec,
    slugify,
)

# Cue words that introduce an exclusion, in the two languages this repo's users write in.
NEGATION_CUES = (
    "ne ",
    "nem kell",
    "nélkül",
    "don't",
    "do not",
    "no ",
    "without",
    "avoid",
    "skip",
)

QUALITY_CUES = {
    "gyors": "Performance is a stated priority.",
    "fast": "Performance is a stated priority.",
    "biztonságos": "Security is a stated priority.",
    "secure": "Security is a stated priority.",
    "egyszerű": "Simplicity is a stated priority.",
    "simple": "Simplicity is a stated priority.",
    "szép": "Visual quality is a stated priority; the bar is undefined.",
    "pretty": "Visual quality is a stated priority; the bar is undefined.",
    "reszponzív": "Responsive layout is required.",
    "responsive": "Responsive layout is required.",
    "tesztelt": "Tests are expected as part of the work.",
    "tested": "Tests are expected as part of the work.",
    "akadálymentes": "Accessibility is a stated priority.",
    "accessible": "Accessibility is a stated priority.",
}

STACK_CUES = (
    "python", "typescript", "javascript", "react", "vue", "svelte", "next.js", "nextjs",
    "django", "flask", "fastapi", "node", "express", "rust", "go", "java", "kotlin",
    "swift", "postgres", "postgresql", "sqlite", "mysql", "redis", "docker", "kubernetes",
    "tailwind", "css", "html", "graphql", "rest", "dspy",
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [part.strip(" -•\t") for part in SENTENCE_SPLIT.split(text.strip()) if part.strip(" -•\t")]


def _detect_stack(text: str) -> list[str]:
    """Find technology names, tolerating inflection.

    Hungarian glues case endings onto foreign words — "Reactban", "Pythonnal" —
    so an exact word-boundary match misses them. Names of four characters or
    more allow a short suffix; shorter ones (`go`, `css`) stay exact, because
    `go[a-z]{0,4}` would happily match "good".
    """
    lowered = text.lower()
    found = set()
    for cue in STACK_CUES:
        pattern = rf"\b{re.escape(cue)}[a-zá-űö-ü]{{0,4}}\b" if len(cue) >= 4 else rf"\b{re.escape(cue)}\b"
        if re.search(pattern, lowered):
            found.add(cue)
    return sorted(found)


CLAUSE_SPLIT = re.compile(r",\s*|\s+(?:but|however|viszont|de)\s+")


def _clauses(sentence: str) -> list[str]:
    """Split a sentence into clauses.

    Exclusions usually arrive as one clause inside an otherwise positive
    sentence — "make a login form, no backend needed". Classifying whole
    sentences would drop the entire request into non-goals.
    """
    return [clause.strip() for clause in CLAUSE_SPLIT.split(sentence) if clause.strip()]


def _split_scope(sentences: list[str]) -> tuple[list[str], list[str]]:
    scope: list[str] = []
    non_goals: list[str] = []
    for sentence in sentences:
        for clause in _clauses(sentence):
            lowered = clause.lower()
            if any(cue in lowered for cue in NEGATION_CUES):
                non_goals.append(clause)
            else:
                scope.append(clause)
    return scope, non_goals


def _constraints(text: str) -> list[str]:
    lowered = text.lower()
    found = {note for cue, note in QUALITY_CUES.items() if re.search(rf"\b{re.escape(cue)}", lowered)}
    return sorted(found)


def _title(instruction: str) -> str:
    first = _sentences(instruction)[0] if _sentences(instruction) else instruction
    words = first.split()
    return " ".join(words[:8]).rstrip(",.:;") or "Untitled task"


def _acceptance(scope: list[str], constraints: list[str]) -> list[AcceptanceCriterion]:
    criteria = [
        AcceptanceCriterion(
            statement=item.rstrip(".") + " is implemented",
            verification="Reviewer runs the feature and confirms the described behaviour.",
        )
        for item in scope[:3]
    ]
    if any("Tests are expected" in note for note in constraints):
        criteria.append(
            AcceptanceCriterion(
                statement="Automated tests cover the new behaviour",
                verification="The project's test command passes and fails when the change is reverted.",
            )
        )
    if not criteria:
        criteria.append(
            AcceptanceCriterion(
                statement="The requested change exists and runs",
                verification="Reviewer executes the affected path and observes the intended result.",
            )
        )
    return criteria


def _open_questions(instruction: str, constraints: list[str], stack: list[str]) -> list[OpenQuestion]:
    questions: list[OpenQuestion] = []
    if not stack:
        questions.append(
            OpenQuestion(
                question="Which language, framework, and runtime should this target?",
                why_it_matters="The instruction names no stack, so any choice is the converter's, not the user's.",
                assumption_used="Follow the conventions already present in the target repository.",
            )
        )
    if any("Visual quality" in note for note in constraints):
        questions.append(
            OpenQuestion(
                question="What is the visual reference or design system for 'nice-looking'?",
                why_it_matters="Aesthetic requirements without a reference cannot be accepted or rejected objectively.",
                assumption_used="Match the existing UI of the surrounding application.",
            )
        )
    if not re.search(r"\b(test|teszt|spec)\b", instruction.lower()):
        questions.append(
            OpenQuestion(
                question="Is automated test coverage expected for this change?",
                why_it_matters="It changes the size of the work and the definition of done.",
                assumption_used="Add tests for new logic where the repository already has a test suite.",
            )
        )
    return questions


def spec_from_instruction(instruction: str, repo_context: str = "") -> VibeSpec:
    """Build a structured brief from a loose instruction, without an LM."""
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")

    sentences = _sentences(instruction)
    scope, non_goals = _split_scope(sentences)
    constraints = _constraints(instruction)
    stack = _detect_stack(f"{instruction}\n{repo_context}")
    context_parts = [part for part in (repo_context.strip(), "Detected stack: " + ", ".join(stack) if stack else "") if part]

    return VibeSpec(
        title=_title(instruction),
        slug=slugify(_title(instruction)),
        goal=sentences[0] if sentences else instruction,
        context="\n".join(context_parts),
        scope=scope,
        non_goals=non_goals,
        constraints=constraints,
        acceptance=_acceptance(scope, constraints),
        risks=[
            "The brief was derived by keyword heuristics, so intent may be misread.",
            "Requirements the instruction implied but did not state are absent by design.",
        ],
        open_questions=_open_questions(instruction, constraints, stack),
        source_instruction=instruction,
        confidence="LOW",
    )


def agent_from_spec(spec: VibeSpec, available_tools: list[str] | None = None) -> AgentArtifact:
    """Derive an agent definition from a brief, without an LM."""
    guardrails = [
        "Work only within the stated scope; anything else needs a new instruction.",
        "Do not resolve a blocking open question by guessing; ask the user.",
    ]
    guardrails += [f"Out of scope: {item}" for item in spec.non_goals]
    guardrails += [f"Risk to watch: {item}" for item in spec.risks]

    instructions = [
        "Read the brief and confirm the assumptions still hold.",
        *(f"Implement: {item}" for item in spec.scope),
        "Verify every acceptance criterion by its stated verification method.",
        "Report what was done, what was skipped, and why.",
    ]

    return AgentArtifact(
        name=f"{spec.slug}-agent"[:64],
        description=f"Implements: {spec.goal}",
        role=(
            f"You implement the brief '{spec.title}'. The brief is the authority; "
            "where it is silent, state your assumption instead of inventing a requirement."
        ),
        tools=sorted(available_tools or []),
        instructions=instructions,
        guardrails=guardrails,
        success_criteria=[criterion.statement for criterion in spec.acceptance],
        source_spec=spec.slug,
    )


def skill_from_spec(spec: VibeSpec) -> SkillArtifact:
    """Derive a reusable skill definition from a brief, without an LM."""
    keywords = re.findall(r"[a-zá-űA-ZÁ-Ű0-9]{4,}", spec.title.lower())[:5]
    return SkillArtifact(
        name=f"{spec.slug}-skill"[:64],
        description=f"{spec.goal} Use when a request matches this task shape.",
        triggers=sorted({spec.title.lower(), *keywords}),
        procedure=[
            "Restate the request as a brief: goal, scope, non-goals, acceptance.",
            *(f"Apply constraint: {item}" for item in spec.constraints),
            *(f"Deliver: {item}" for item in spec.scope),
            "Check each acceptance criterion and report the result.",
        ],
        inputs=["The user's request", "The target repository or file paths"],
        outputs=["The implemented change", "A short report of criteria met and skipped"],
        checks=[criterion.verification for criterion in spec.acceptance],
        limits=[
            "Does not decide questions the brief left open.",
            *spec.non_goals,
        ],
        source_spec=spec.slug,
    )


def bundle_from_instruction(
    instruction: str,
    repo_context: str = "",
    available_tools: list[str] | None = None,
) -> VibeBundle:
    """Full offline pipeline: instruction to spec, agent, and skill."""
    spec = spec_from_instruction(instruction, repo_context)
    return VibeBundle(
        spec=spec,
        agent=agent_from_spec(spec, available_tools),
        skill=skill_from_spec(spec),
    )
