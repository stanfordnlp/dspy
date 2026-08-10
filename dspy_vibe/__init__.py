"""dspy_vibe — turn vibe-coding instructions into specs, agents, and skills.

Quick start (no LM needed):

    from dspy_vibe import bundle_from_instruction, write_bundle

    bundle = bundle_from_instruction("csinálj egy login formot, legyen szép és gyors")
    write_bundle(bundle, "./vibe-out")

With an LM:

    import dspy
    from dspy_vibe import VibeCoder

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    result = VibeCoder(available_tools=["Read", "Edit", "Bash"])(
        instruction="csinálj egy login formot, legyen szép és gyors"
    )
    result.bundle.spec, result.bundle.agent, result.bundle.skill

The DSPy modules are imported lazily so the deterministic path stays usable in
environments where dspy's own dependencies are unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy_vibe.emitters import (
    agent_to_markdown,
    skill_to_markdown,
    spec_to_markdown,
    write_bundle,
)
from dspy_vibe.metrics import agent_validity, bundle_quality, skill_validity, spec_quality
from dspy_vibe.offline import (
    agent_from_spec,
    bundle_from_instruction,
    skill_from_spec,
    spec_from_instruction,
)
from dspy_vibe.types import (
    AcceptanceCriterion,
    AgentArtifact,
    OpenQuestion,
    SkillArtifact,
    VibeBundle,
    VibeSpec,
)

if TYPE_CHECKING:  # pragma: no cover
    from dspy_vibe.modules import SpecToAgentModule, SpecToSkillModule, VibeCoder, VibeToSpec

_LAZY = {
    "VibeCoder": "dspy_vibe.modules",
    "VibeToSpec": "dspy_vibe.modules",
    "SpecToAgentModule": "dspy_vibe.modules",
    "SpecToSkillModule": "dspy_vibe.modules",
    "VibeInstructionToSpec": "dspy_vibe.signatures",
    "SpecToAgent": "dspy_vibe.signatures",
    "SpecToSkill": "dspy_vibe.signatures",
    "CritiqueSpec": "dspy_vibe.signatures",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "AcceptanceCriterion",
    "AgentArtifact",
    "CritiqueSpec",
    "OpenQuestion",
    "SkillArtifact",
    "SpecToAgent",
    "SpecToAgentModule",
    "SpecToSkill",
    "SpecToSkillModule",
    "VibeBundle",
    "VibeCoder",
    "VibeInstructionToSpec",
    "VibeSpec",
    "VibeToSpec",
    "agent_from_spec",
    "agent_to_markdown",
    "agent_validity",
    "bundle_from_instruction",
    "bundle_quality",
    "skill_from_spec",
    "skill_to_markdown",
    "skill_validity",
    "spec_from_instruction",
    "spec_quality",
    "spec_to_markdown",
    "write_bundle",
]
