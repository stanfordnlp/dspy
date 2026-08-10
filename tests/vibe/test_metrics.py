from types import SimpleNamespace

from dspy_vibe import bundle_from_instruction, spec_from_instruction
from dspy_vibe.metrics import (
    agent_validity,
    bundle_quality,
    skill_validity,
    spec_completeness,
    spec_faithfulness,
    spec_honesty,
    spec_quality,
    spec_specificity,
)
from dspy_vibe.types import AcceptanceCriterion, OpenQuestion, VibeSpec

INSTRUCTION = "add a dark mode toggle to the settings page"


def _pred(spec):
    return SimpleNamespace(spec=spec)


def test_invented_scope_lowers_faithfulness():
    grounded = spec_from_instruction(INSTRUCTION)
    invented = grounded.model_copy(
        update={"scope": [*grounded.scope, "migrate the billing system to Stripe"]}
    )
    assert spec_faithfulness(INSTRUCTION, invented) < spec_faithfulness(INSTRUCTION, grounded)


def test_templated_acceptance_criteria_lower_specificity():
    spec = spec_from_instruction(INSTRUCTION)
    templated = spec.model_copy(
        update={
            "acceptance": [
                AcceptanceCriterion(
                    statement=f"{item} is implemented",
                    verification="Reviewer runs the feature.",
                )
                for item in spec.scope
            ]
        }
    )
    assert spec_specificity(templated) < spec_specificity(spec) or spec_specificity(templated) < 1.0


def _well_specified() -> VibeSpec:
    """A brief of the shape an LM-backed run should produce.

    Note what is *not* here: persistence across sessions. The instruction never
    asked for it, so it belongs in an open question, not in scope. Putting it in
    scope is exactly the invention `spec_faithfulness` exists to punish.
    """
    return VibeSpec(
        title="Dark mode toggle",
        goal="The settings page offers a working dark mode toggle.",
        scope=[
            "Add a dark mode toggle control to the settings page",
            "Apply the dark theme when the toggle is on",
        ],
        constraints=["Must not change the login screen"],
        acceptance=[
            AcceptanceCriterion(
                statement="Toggling switches the page between light and dark",
                verification="Open the settings page, click the toggle, observe the theme change.",
            ),
            AcceptanceCriterion(
                statement="No other page changes appearance",
                verification="Run the visual regression suite and confirm only settings differs.",
            ),
        ],
        risks=["Theme tokens may be hard-coded in some components"],
        open_questions=[
            OpenQuestion(
                question="Should the dark mode choice persist across sessions?",
                why_it_matters="The instruction does not say, and it changes the storage design.",
                assumption_used="Keep it in component state for now; no persistence.",
            )
        ],
        source_instruction=INSTRUCTION,
    )


def test_unverifiable_acceptance_criteria_lower_specificity():
    good = _well_specified()
    vague = good.model_copy(
        update={
            "acceptance": [
                AcceptanceCriterion(statement="It feels good", verification="Looks fine to the author.")
            ]
        }
    )
    assert spec_specificity(vague) < spec_specificity(good)


def test_a_well_specified_brief_outscores_the_deterministic_baseline():
    example = SimpleNamespace(instruction=INSTRUCTION)
    baseline = spec_quality(example, _pred(spec_from_instruction(INSTRUCTION)))
    good = spec_quality(example, _pred(_well_specified()))
    assert good > baseline


def test_missing_open_questions_is_only_credible_at_high_confidence():
    spec = spec_from_instruction(INSTRUCTION).model_copy(update={"open_questions": []})
    assert spec_honesty(spec) < 1.0
    assert spec_honesty(spec.model_copy(update={"confidence": "HIGH"})) == 1.0


def test_blocking_question_still_counts_as_honest():
    spec = spec_from_instruction(INSTRUCTION).model_copy(
        update={
            "open_questions": [
                OpenQuestion(question="Which theme system?", why_it_matters="Changes the approach.")
            ]
        }
    )
    assert spec_honesty(spec) == 1.0
    assert spec.blocking_questions


def test_completeness_needs_verification_on_every_criterion():
    full = spec_from_instruction(INSTRUCTION)
    assert spec_completeness(full) == 1.0
    thin = VibeSpec(title="thin", goal="do something")
    assert spec_completeness(thin) < 0.5


def test_spec_quality_is_bounded_and_leaves_headroom_over_the_baseline():
    spec = spec_from_instruction(INSTRUCTION)
    score = spec_quality(SimpleNamespace(instruction=INSTRUCTION), _pred(spec))
    assert 0.0 <= score <= 1.0
    # The deterministic converter must not score full marks, or the metric
    # would give an optimizer nothing to improve.
    assert score < 0.95


def test_spec_quality_returns_zero_for_a_missing_spec():
    assert spec_quality(SimpleNamespace(instruction=INSTRUCTION), SimpleNamespace()) == 0.0


def test_agent_validity_penalizes_tools_the_host_does_not_offer():
    bundle = bundle_from_instruction(INSTRUCTION, available_tools=["Read"])
    assert agent_validity(bundle.agent, ["Read"]) == 1.0
    assert agent_validity(bundle.agent, ["Bash"]) < 1.0


def test_skill_validity_requires_limits_and_checks():
    bundle = bundle_from_instruction(INSTRUCTION)
    assert skill_validity(bundle.skill) == 1.0
    stripped = bundle.skill.model_copy(update={"limits": [], "checks": []})
    assert skill_validity(stripped) < skill_validity(bundle.skill)


def test_bundle_quality_combines_all_three_artifacts():
    bundle = bundle_from_instruction(INSTRUCTION)
    example = SimpleNamespace(instruction=INSTRUCTION, available_tools=None)
    score = bundle_quality(example, SimpleNamespace(bundle=bundle, spec=bundle.spec))
    assert 0.0 < score <= 1.0
    assert bundle_quality(example, SimpleNamespace()) == 0.0
