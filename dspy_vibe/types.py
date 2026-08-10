"""Typed artifacts produced by the vibe-coding pipeline.

Everything the pipeline emits is a pydantic model, so an LM-generated result and
a deterministically generated one are validated by the same rules. Nothing here
imports dspy: artifacts can be built, validated, and written without an LM.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

Confidence = Literal["HIGH", "MEDIUM", "LOW"]


def slugify(text: str, *, fallback: str = "task", max_words: int = 6) -> str:
    """Turn free text into a stable kebab-case slug.

    Accented characters are transliterated rather than dropped, so a Hungarian
    title yields `csinalj-egy-login-formot`, not `csin-lj-egy-login-formot`.
    """
    normalized = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", stripped)
    slug = "-".join(words[:max_words])
    return slug or fallback


class AcceptanceCriterion(BaseModel):
    """One check that decides whether the work is done.

    `verification` is what someone actually runs or looks at. A criterion no one
    can check is a wish, not a criterion.
    """

    statement: str = Field(description="Observable outcome, phrased as a testable claim.")
    verification: str = Field(description="Concrete command, test, or observation that settles it.")

    @field_validator("statement", "verification")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("acceptance criterion fields must be non-empty")
        return value.strip()


class OpenQuestion(BaseModel):
    """Something the vibe instruction did not settle.

    Vibe instructions are underspecified by nature. Recording the gap with the
    assumption used in its place keeps the brief honest instead of inventing a
    requirement the user never stated.
    """

    question: str
    why_it_matters: str
    assumption_used: str = Field(
        default="",
        description="The assumption applied so work can proceed. Empty means the question blocks work.",
    )

    @property
    def blocking(self) -> bool:
        return not self.assumption_used.strip()


class VibeSpec(BaseModel):
    """A loose instruction turned into a structured development brief."""

    title: str
    slug: str = ""
    goal: str = Field(description="One sentence: what must be true when this is done.")
    context: str = Field(default="", description="Stack, repository, or environment facts stated by the user.")
    scope: list[str] = Field(default_factory=list, description="Work that is in scope.")
    non_goals: list[str] = Field(default_factory=list, description="Explicitly out of scope.")
    constraints: list[str] = Field(default_factory=list, description="Technical, legal, or stylistic limits.")
    acceptance: list[AcceptanceCriterion] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    source_instruction: str = Field(default="", description="The original vibe instruction, verbatim.")
    confidence: Confidence = "MEDIUM"

    @model_validator(mode="after")
    def _fill_slug(self) -> VibeSpec:
        if not self.slug:
            object.__setattr__(self, "slug", slugify(self.title))
        if not SLUG_RE.fullmatch(self.slug):
            raise ValueError(f"slug must be kebab-case, got {self.slug!r}")
        return self

    @property
    def blocking_questions(self) -> list[OpenQuestion]:
        return [question for question in self.open_questions if question.blocking]


class AgentArtifact(BaseModel):
    """An `.agent` definition: who does the work and under what limits."""

    name: str
    description: str = Field(description="One line; used to decide when to invoke the agent.")
    role: str = Field(description="The agent's standing brief.")
    tools: list[str] = Field(default_factory=list)
    model: str = ""
    instructions: list[str] = Field(default_factory=list, description="Ordered working steps.")
    guardrails: list[str] = Field(default_factory=list, description="What the agent must not do.")
    success_criteria: list[str] = Field(default_factory=list)
    source_spec: str = Field(default="", description="Slug of the spec this was generated from.")

    @field_validator("name")
    @classmethod
    def _slug_name(cls, value: str) -> str:
        value = value.strip()
        if not SLUG_RE.fullmatch(value):
            raise ValueError(f"agent name must be kebab-case, got {value!r}")
        return value


class SkillArtifact(BaseModel):
    """A `.skill` definition: a reusable procedure plus the cues that trigger it."""

    name: str
    description: str = Field(description="One line stating what the skill does and when it triggers.")
    triggers: list[str] = Field(default_factory=list, description="Phrases or situations that should load it.")
    procedure: list[str] = Field(default_factory=list, description="Ordered steps.")
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list, description="Verification before the skill reports done.")
    limits: list[str] = Field(default_factory=list, description="What the skill does not cover.")
    source_spec: str = Field(default="", description="Slug of the spec this was generated from.")

    @field_validator("name")
    @classmethod
    def _slug_name(cls, value: str) -> str:
        value = value.strip()
        if not SLUG_RE.fullmatch(value):
            raise ValueError(f"skill name must be kebab-case, got {value!r}")
        return value


class VibeBundle(BaseModel):
    """Everything one vibe instruction produces."""

    spec: VibeSpec
    agent: AgentArtifact
    skill: SkillArtifact
