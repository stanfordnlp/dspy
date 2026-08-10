"""Tests for the DSPy-backed pipeline.

These use `DummyLM`, so they exercise the real signature/adapter path without a
network call: the LM returns fixed field values and DSPy must parse them back
into the typed artifacts.
"""

import json

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from dspy_vibe.modules import SpecToAgentModule, SpecToSkillModule, VibeCoder, VibeToSpec, lm_available
from dspy_vibe.types import AgentArtifact, SkillArtifact, VibeSpec

INSTRUCTION = "add a dark mode toggle to the settings page"


def _spec_payload(**overrides) -> str:
    payload = {
        "title": "Dark mode toggle",
        "slug": "dark-mode-toggle",
        "goal": "The settings page offers a dark mode toggle.",
        "context": "",
        "scope": ["Add a dark mode toggle to the settings page"],
        "non_goals": [],
        "constraints": [],
        "acceptance": [
            {
                "statement": "Toggling switches the theme",
                "verification": "Open the settings page and click the toggle.",
            }
        ],
        "risks": [],
        "open_questions": [],
        "source_instruction": "",
        "confidence": "MEDIUM",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _agent_payload(**overrides) -> str:
    payload = {
        "name": "dark-mode-agent",
        "description": "Adds a dark mode toggle.",
        "role": "You implement the dark mode toggle brief.",
        "tools": ["Read", "Edit"],
        "model": "",
        "instructions": ["Read the brief", "Implement the toggle"],
        "guardrails": ["Do not touch the login screen"],
        "success_criteria": ["Toggling switches the theme"],
        "source_spec": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _skill_payload(**overrides) -> str:
    payload = {
        "name": "theme-toggle-skill",
        "description": "Adds a theme toggle to a page.",
        "triggers": ["dark mode", "theme toggle"],
        "procedure": ["Locate the page", "Add the control", "Wire the theme"],
        "inputs": ["Target page"],
        "outputs": ["The toggle"],
        "checks": ["Click the toggle and observe the theme"],
        "limits": ["Does not add persistence"],
        "source_spec": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def no_lm():
    """Run with DSPy configured but no LM, to exercise the offline fallback."""
    with dspy.context(lm=None):
        yield


def test_pipeline_falls_back_to_the_deterministic_converter_without_an_lm(no_lm):
    assert lm_available() is False
    result = VibeCoder()(instruction=INSTRUCTION)
    assert result.spec.confidence == "LOW"
    assert result.bundle.agent.source_spec == result.spec.slug


def test_spec_module_parses_a_typed_response_and_keeps_the_source_text():
    lm = DummyLM([{"reasoning": "…", "spec": _spec_payload()}])
    with dspy.context(lm=lm):
        spec = VibeToSpec()(instruction=INSTRUCTION).spec
    assert isinstance(spec, VibeSpec)
    assert spec.slug == "dark-mode-toggle"
    # The module backfills the verbatim instruction so a reviewer can compare.
    assert spec.source_instruction == INSTRUCTION


def test_agent_module_drops_tools_the_host_does_not_offer(caplog):
    lm = DummyLM([{"agent": _agent_payload(tools=["Read", "Edit", "Deploy"])}])
    spec = VibeSpec(title="Dark mode toggle", goal="g", scope=["s"])
    with dspy.context(lm=lm):
        agent = SpecToAgentModule()(spec=spec, available_tools=["Read", "Edit"]).agent
    assert isinstance(agent, AgentArtifact)
    assert agent.tools == ["Read", "Edit"]
    assert agent.source_spec == spec.slug


def test_skill_module_backfills_the_source_spec():
    lm = DummyLM([{"skill": _skill_payload()}])
    spec = VibeSpec(title="Dark mode toggle", goal="g", scope=["s"])
    with dspy.context(lm=lm):
        skill = SpecToSkillModule()(spec=spec).skill
    assert isinstance(skill, SkillArtifact)
    assert skill.source_spec == spec.slug


def test_full_pipeline_with_an_lm_returns_all_three_artifacts():
    lm = DummyLM(
        [
            {"reasoning": "…", "spec": _spec_payload()},
            {"agent": _agent_payload()},
            {"skill": _skill_payload()},
        ]
    )
    with dspy.context(lm=lm):
        result = VibeCoder(available_tools=["Read", "Edit"])(instruction=INSTRUCTION)
    assert result.spec.slug == "dark-mode-toggle"
    assert result.agent.name == "dark-mode-agent"
    assert result.skill.name == "theme-toggle-skill"
    assert result.bundle.spec is result.spec


def test_empty_instruction_is_rejected_before_any_lm_call():
    with pytest.raises(ValueError):
        VibeToSpec()(instruction="  ")
