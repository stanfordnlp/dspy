"""DSPy signatures for the vibe-coding conversion.

Signatures declare *what* each step maps to; the prompt is DSPy's business and
an optimizer's to improve. Each output is a typed artifact from `types`, so a
malformed generation fails validation instead of reaching a file.
"""

from __future__ import annotations

import dspy
from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeSpec


class VibeInstructionToSpec(dspy.Signature):
    """Convert a loose, informal coding request into a structured development brief.

    Keep the user's intent; do not invent requirements they never stated. When
    the instruction leaves something undecided, record it as an open question
    with the assumption you applied, rather than silently choosing for them.
    Acceptance criteria must be checkable by running or observing something.
    """

    instruction: str = dspy.InputField(desc="The raw, informal request, in the user's own words.")
    repo_context: str = dspy.InputField(
        desc="Known facts about the codebase: stack, conventions, constraints. May be empty."
    )
    spec: VibeSpec = dspy.OutputField(desc="The structured brief.")


class SpecToAgent(dspy.Signature):
    """Derive an executable agent definition from a development brief.

    The agent is the *worker*: it carries the role, the ordered steps, and the
    limits. Guardrails must include anything the brief marked as a non-goal or
    a risk, so the agent cannot quietly widen its own scope.
    """

    spec: VibeSpec = dspy.InputField()
    available_tools: str = dspy.InputField(desc="Comma-separated tool names the host offers. May be empty.")
    agent: AgentArtifact = dspy.OutputField()


class SpecToSkill(dspy.Signature):
    """Derive a reusable skill definition from a development brief.

    The skill is the *method*: the part worth reusing on the next task of this
    shape. Write triggers as phrases a user would actually type, and keep the
    procedure ordered and concrete. State explicitly what the skill does not
    cover.
    """

    spec: VibeSpec = dspy.InputField()
    skill: SkillArtifact = dspy.OutputField()


class CritiqueSpec(dspy.Signature):
    """Judge whether a brief is faithful to the original instruction and usable.

    Flag invented requirements first: a brief that adds work the user never
    asked for is worse than a thin one.
    """

    instruction: str = dspy.InputField()
    spec: VibeSpec = dspy.InputField()
    invented_requirements: list[str] = dspy.OutputField(desc="Requirements not supported by the instruction.")
    missing_aspects: list[str] = dspy.OutputField(desc="Parts of the instruction the brief dropped.")
    score: float = dspy.OutputField(desc="0.0-1.0 overall usability of the brief.")
