import pytest

from dspy_vibe import bundle_from_instruction, spec_from_instruction
from dspy_vibe.offline import agent_from_spec, skill_from_spec
from dspy_vibe.types import slugify

HU = "Csinálj egy login formot Reactban, legyen szép és gyors, ne kelljen hozzá backend. Tesztelt legyen."
EN = "build a small CLI that lists open PRs, no auth flows please"


def test_negation_moves_only_the_clause_not_the_sentence():
    spec = spec_from_instruction(HU)
    assert any("login formot" in item for item in spec.scope)
    assert spec.non_goals == ["ne kelljen hozzá backend."]


def test_english_negation_cue_is_recognized():
    spec = spec_from_instruction(EN)
    assert spec.non_goals
    assert "auth" in " ".join(spec.non_goals)


def test_slug_transliterates_accents():
    assert slugify("Csinálj egy login formot") == "csinalj-egy-login-formot"
    assert slugify("!!!") == "task"


def test_quality_cues_become_constraints():
    spec = spec_from_instruction(HU)
    joined = " ".join(spec.constraints)
    assert "Performance" in joined
    assert "Visual quality" in joined
    assert "Tests are expected" in joined


def test_stack_detection_lands_in_context():
    spec = spec_from_instruction(HU)
    assert "react" in spec.context.lower()


def test_offline_spec_is_low_confidence_and_asks_questions():
    spec = spec_from_instruction(EN)
    assert spec.confidence == "LOW"
    assert spec.open_questions
    assert all(question.why_it_matters for question in spec.open_questions)


def test_acceptance_criteria_are_always_verifiable():
    spec = spec_from_instruction(EN)
    assert spec.acceptance
    assert all(criterion.verification.strip() for criterion in spec.acceptance)


def test_empty_instruction_is_rejected():
    with pytest.raises(ValueError):
        spec_from_instruction("   ")


def test_agent_guardrails_carry_non_goals_and_risks():
    spec = spec_from_instruction(HU)
    agent = agent_from_spec(spec, ["Read", "Edit"])
    guardrails = " ".join(agent.guardrails)
    assert "backend" in guardrails
    assert agent.tools == ["Edit", "Read"]
    assert agent.source_spec == spec.slug


def test_skill_limits_include_non_goals():
    spec = spec_from_instruction(HU)
    skill = skill_from_spec(spec)
    assert any("backend" in limit for limit in skill.limits)
    assert skill.triggers
    assert len(skill.procedure) >= 2


def test_bundle_links_all_three_artifacts():
    bundle = bundle_from_instruction(HU, "repo: monorepo, pnpm", ["Read"])
    assert bundle.agent.source_spec == bundle.spec.slug
    assert bundle.skill.source_spec == bundle.spec.slug
    assert "monorepo" in bundle.spec.context
