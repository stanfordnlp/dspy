"""DSPy modules for the vibe-coding pipeline.

`VibeCoder` is the whole thing: instruction in, spec plus `.agent` plus
`.skill` out. It is a normal `dspy.Module`, so it can be compiled by any DSPy
optimizer against the metrics in `metrics.py`.

Every step falls back to the deterministic converter when no LM is configured,
which keeps the pipeline usable offline and gives an optimizer a real baseline
to beat.
"""

from __future__ import annotations

import logging

import dspy
from dspy_vibe import offline
from dspy_vibe.signatures import SpecToAgent, SpecToSkill, VibeInstructionToSpec
from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeBundle, VibeSpec

logger = logging.getLogger(__name__)


def lm_available() -> bool:
    """True when DSPy has an LM configured for the current context."""
    try:
        return dspy.settings.lm is not None
    except Exception:  # pragma: no cover - settings access should not break callers
        return False


class VibeToSpec(dspy.Module):
    """Turn a loose instruction into a structured brief."""

    def __init__(self, *, use_cot: bool = True):
        super().__init__()
        predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.convert = predictor(VibeInstructionToSpec)

    def forward(self, instruction: str, repo_context: str = "") -> dspy.Prediction:
        if not instruction or not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not lm_available():
            return dspy.Prediction(spec=offline.spec_from_instruction(instruction, repo_context))
        result = self.convert(instruction=instruction, repo_context=repo_context)
        spec: VibeSpec = result.spec
        # The source text is evidence, not a generation target: keep it verbatim
        # so a later reviewer can compare the brief against what was asked.
        if not spec.source_instruction:
            spec = spec.model_copy(update={"source_instruction": instruction.strip()})
        return dspy.Prediction(spec=spec)


class SpecToAgentModule(dspy.Module):
    """Derive an `.agent` definition from a brief."""

    def __init__(self, *, use_cot: bool = False):
        super().__init__()
        predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.convert = predictor(SpecToAgent)

    def forward(self, spec: VibeSpec, available_tools: list[str] | None = None) -> dspy.Prediction:
        tools = sorted(available_tools or [])
        if not lm_available():
            return dspy.Prediction(agent=offline.agent_from_spec(spec, tools))
        result = self.convert(spec=spec, available_tools=", ".join(tools))
        agent: AgentArtifact = result.agent
        updates: dict[str, object] = {}
        if not agent.source_spec:
            updates["source_spec"] = spec.slug
        # A tool the host does not offer is a promise the agent cannot keep.
        if tools:
            allowed = [tool for tool in agent.tools if tool in tools]
            if allowed != agent.tools:
                logger.warning("dropping tools not offered by the host: %s", sorted(set(agent.tools) - set(tools)))
                updates["tools"] = allowed
        return dspy.Prediction(agent=agent.model_copy(update=updates) if updates else agent)


class SpecToSkillModule(dspy.Module):
    """Derive a `.skill` definition from a brief."""

    def __init__(self, *, use_cot: bool = False):
        super().__init__()
        predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.convert = predictor(SpecToSkill)

    def forward(self, spec: VibeSpec) -> dspy.Prediction:
        if not lm_available():
            return dspy.Prediction(skill=offline.skill_from_spec(spec))
        result = self.convert(spec=spec)
        skill: SkillArtifact = result.skill
        if not skill.source_spec:
            skill = skill.model_copy(update={"source_spec": spec.slug})
        return dspy.Prediction(skill=skill)


class VibeCoder(dspy.Module):
    """The full pipeline: vibe instruction to spec, agent, and skill.

    Args:
        available_tools: Tool names the host offers. Generated agents are
            restricted to these.
        use_cot: Reason step-by-step before producing the brief. Worth it for
            the brief, rarely for the mechanical derivations after it.
    """

    def __init__(self, *, available_tools: list[str] | None = None, use_cot: bool = True):
        super().__init__()
        self.available_tools = sorted(available_tools or [])
        self.to_spec = VibeToSpec(use_cot=use_cot)
        self.to_agent = SpecToAgentModule()
        self.to_skill = SpecToSkillModule()

    def forward(self, instruction: str, repo_context: str = "") -> dspy.Prediction:
        spec = self.to_spec(instruction=instruction, repo_context=repo_context).spec
        agent = self.to_agent(spec=spec, available_tools=self.available_tools).agent
        skill = self.to_skill(spec=spec).skill
        bundle = VibeBundle(spec=spec, agent=agent, skill=skill)
        return dspy.Prediction(bundle=bundle, spec=spec, agent=agent, skill=skill)
