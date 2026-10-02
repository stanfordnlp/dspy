import copy
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from gepa.strategies.instruction_proposal import InstructionProposalSignature

import dspy
from dspy.teleprompt.gepa import instruction_proposal
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM


def count_messages_with_image_url_pattern(messages):
    """Helper to count image URLs in messages - borrowed from image adapter tests"""
    pattern = {"type": "image_url", "image_url": {"url": lambda x: isinstance(x, str)}}

    try:

        def check_pattern(obj, pattern):
            if isinstance(pattern, dict):
                if not isinstance(obj, dict):
                    return False
                return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
            if callable(pattern):
                return pattern(obj)
            return obj == pattern

        def count_patterns(obj, pattern):
            count = 0
            if check_pattern(obj, pattern):
                count += 1
            if isinstance(obj, dict):
                count += sum(count_patterns(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count

        return count_patterns(messages, pattern)
    except Exception:
        return 0


@dataclass
class ImagesInHistory:
    has_structured_images: bool
    has_text_serialized_images: bool


def check_images_in_history(history: list[Any]) -> ImagesInHistory:
    def check_text_serialized(item: Any) -> bool:
        if isinstance(item, list):
            return any(check_text_serialized(i) for i in item)
        if isinstance(item, dict):
            return any(check_text_serialized(i) for i in item.values())
        if isinstance(item, str):
            return "CUSTOM-TYPE-START-IDENTIFIER" in item

        return False

    has_structured_images = False

    for call in history:
        if call.get("messages"):
            image_count = count_messages_with_image_url_pattern(call["messages"])
            if image_count > 0:
                has_structured_images = True

                break

    return ImagesInHistory(
        has_structured_images=has_structured_images,
        has_text_serialized_images=any(check_text_serialized(i) for i in history),
    )


def test_reflection_lm_gets_structured_images():
    """
    Verify reflection LM receives structured image messages, not serialized text.
    """
    student = dspy.Predict("image: dspy.Image -> label: str")
    image = dspy.Image("https://example.com/test.jpg")
    example = dspy.Example(image=image, label="dog").with_inputs("image")

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Better instruction"},
            {"improved_instruction": "Enhanced visual analysis instruction"},
            {"improved_instruction": "Focus on key features"},
            {"improved_instruction": "Analyze visual patterns systematically"},
            {"improved_instruction": "Consider distinctive visual elements"},
            {"improved_instruction": "Enhance recognition accuracy"},
            {"improved_instruction": "Improve classification methodology"},
        ]
    )
    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "canine"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
            {"label": "quadruped"},
            {"label": "vertebrate"},
        ]
    )
    dspy.configure(lm=lm)

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=2,
        reflection_lm=reflection_lm,
        instruction_proposer=instruction_proposal.MultiModalInstructionProposer(),
    )

    gepa.compile(student, trainset=[example], valset=[example])

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_structured_images, "Reflection LM should have received structured images"
    assert not images_in_history.has_text_serialized_images, "Reflection LM received serialized images in prompts"


def test_custom_proposer_without_reflection_lm():
    """Test that custom instruction proposers can work without reflection_lm when using updated GEPA core."""

    # External reflection LM managed by the custom proposer
    external_reflection_lm = DummyLM(
        [
            {"improved_instruction": "External LM response"},
            {"improved_instruction": "Enhanced instruction"},
            {"improved_instruction": "Better guidance"},
            {"improved_instruction": "Optimized instruction"},
            {"improved_instruction": "Refined approach"},
        ]
    )

    class ProposerWithExternalLM:
        def __call__(self, candidate, reflective_dataset, components_to_update):
            # This proposer manages its own external reflection LM
            with dspy.context(lm=external_reflection_lm):
                # Use external LM for reflection (optional - could be any custom logic)
                external_reflection_lm([{"role": "user", "content": "Improve this instruction"}])
                return {name: f"Externally-improved: {candidate[name]}" for name in components_to_update}

    student = dspy.Predict("text -> label")
    example = dspy.Example(text="test input", label="test").with_inputs("text")

    # Use a robust dummy LM with enough responses for optimization steps
    lm = DummyLM(
        [
            {"label": "test"},
            {"label": "result"},
            {"label": "output"},
            {"label": "response"},
            {"label": "classification"},
            {"label": "prediction"},
            {"label": "category"},
            {"label": "type"},
            {"label": "class"},
            {"label": "group"},
            {"label": "kind"},
            {"label": "variant"},
            {"label": "form"},
            {"label": "style"},
            {"label": "mode"},
        ]
    )
    dspy.configure(lm=lm)

    # Test the full flexibility: no reflection_lm provided to GEPA at all!
    # The updated GEPA core library now allows this when using custom proposers
    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.7,  # Score to trigger optimization
        max_metric_calls=5,  # More calls to allow proper optimization
        reflection_lm=None,  # No reflection_lm provided - this now works!
        instruction_proposer=ProposerWithExternalLM(),
    )

    result = gepa.compile(student, trainset=[example], valset=[example])

    assert result is not None
    assert len(lm.history) > 0, "Main LM should have been called"
    assert len(external_reflection_lm.history) > 0, "External reflection LM should have been called by custom proposer"


def test_image_serialization_into_strings():
    """
    Test that demonstrates the image serialization problem when calling lm directly with serialized image data.
    """

    class InstructionProposerCallingLMDirectly:
        def __call__(
            self,
            candidate: dict[str, str],
            reflective_dataset: dict[str, list[dict[str, Any]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            updated_components = {}

            for component_name in components_to_update:
                if component_name not in candidate or component_name not in reflective_dataset:
                    continue

                current_instruction = candidate[component_name]
                component_data = reflective_dataset[component_name]

                feedback_analysis = "Feedback analysis:\n"
                for i, example in enumerate(component_data):
                    feedback_analysis += f"Example {i + 1}:\n"

                    # Non ideal approach: extract and serialize image objects directly
                    inputs = example.get("Inputs", {})
                    for key, value in inputs.items():
                        feedback_analysis += f"  {key}: {value}\n"

                    outputs = example.get("Generated Outputs", {})
                    feedback = example.get("Feedback", "")
                    feedback_analysis += f"  Outputs: {outputs}\n"
                    feedback_analysis += f"  Feedback: {feedback}\n\n"

                context_lm = dspy.settings.lm
                messages = [
                    {"role": "system", "content": "You are an instruction improvement assistant."},
                    {
                        "role": "user",
                        "content": f"Current instruction: {current_instruction}\n\nFeedback: {feedback_analysis}\n\nProvide an improved instruction:",
                    },
                ]

                result = context_lm(messages=messages)
                updated_components[component_name] = result[0]

            return updated_components

    direct_lm_call_proposer = InstructionProposerCallingLMDirectly()

    student = dspy.Predict("image -> label")

    image = dspy.Image("https://picsum.photos/id/237/200/300")

    examples = [
        dspy.Example(image=image, label="cat").with_inputs("image"),
        dspy.Example(image=image, label="animal").with_inputs("image"),
    ]

    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
        ]
    )
    dspy.configure(lm=lm)

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Be more specific about image analysis"},
            {"improved_instruction": "Focus on visual features when classifying"},
            {"improved_instruction": "Consider contextual clues in the image"},
            {"improved_instruction": "Analyze shape, color, and texture patterns"},
            {"improved_instruction": "Look for distinguishing characteristics"},
        ]
    )

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=5,
        reflection_lm=reflection_lm,
        instruction_proposer=direct_lm_call_proposer,
    )

    gepa.compile(student, trainset=examples, valset=examples)

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_text_serialized_images, (
        "Expected to find serialized images (CUSTOM-TYPE-START-IDENTIFIER)"
    )


@pytest.mark.parametrize("reasoning", [True, False])
def test_default_proposer(reasoning: bool, caplog):
    student = dspy.Predict("image -> label")

    image = dspy.Image("https://picsum.photos/id/237/200/300")

    examples = [
        dspy.Example(image=image, label="cat").with_inputs("image"),
        dspy.Example(image=image, label="animal").with_inputs("image"),
    ]

    lm = DummyLM(
        [
            {"label": "cat"},
            {"label": "dog"},
            {"label": "animal"},
            {"label": "pet"},
            {"label": "feline"},
            {"label": "mammal"},
            {"label": "creature"},
            {"label": "species"},
            {"label": "domestic"},
            {"label": "wild"},
            {"label": "carnivore"},
            {"label": "herbivore"},
        ]
    )
    dspy.configure(lm=lm)

    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Be more specific about image analysis"},
            {"improved_instruction": "Focus on visual features when classifying"},
            {"improved_instruction": "Consider contextual clues in the image"},
            {"improved_instruction": "Analyze shape, color, and texture patterns"},
            {"improved_instruction": "Look for distinguishing characteristics"},
        ],
        reasoning=reasoning,
    )

    gepa = dspy.GEPA(
        metric=lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 0.3,
        max_metric_calls=5,
        reflection_lm=reflection_lm,
    )

    with caplog.at_level(logging.INFO, logger="dspy.teleprompt.gepa.gepa"):
        # Let logs propagate up to root because gepa uses try-catch and logs the error
        # https://github.com/gepa-ai/gepa/blob/1b5eff5133be1015210e0512953c25a4b85ad454/src/gepa/proposer/reflective_mutation/reflective_mutation.py#L128
        dspy_logger = logging.getLogger("dspy")
        original_propagate = dspy_logger.propagate
        dspy_logger.propagate = True

        gepa.compile(student, trainset=examples, valset=examples)

        dspy_logger.propagate = original_propagate

        # Check that no internal GEPA reflection errors occurred
        assert "Exception during reflection/proposal" not in caplog.text

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_text_serialized_images, (
        "Expected to find serialized images (CUSTOM-TYPE-START-IDENTIFIER)"
    )


# --- reflection_instruction: additive guidance for the default proposer ------


DATASET = [{"Inputs": {"input": "x"}, "Generated Outputs": {"output": "y"}, "Feedback": "bad"}]
HEADING = "Additional instructions for proposing the new instruction:"


class CapturingLM:
    """Callable reflection LM that records the prompt it receives and returns scripted outputs."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts: list[Any] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.prompts.append(prompt)
        if self.responses:
            return [self.responses.pop(0)]
        return ["```\nNew instruction\n```"]


def _make_adapter(reflection_lm, reflection_instruction=None, custom_instruction_proposer=None):
    return DspyAdapter(
        student_module=dspy.Predict("input -> output"),
        metric_fn=lambda *args, **kwargs: 0.0,
        feedback_map={},
        reflection_lm=reflection_lm,
        custom_instruction_proposer=custom_instruction_proposer,
        reflection_instruction=reflection_instruction,
    )


def _baseline(current_instruction: str, dataset: list[dict[str, Any]]) -> str:
    return InstructionProposalSignature.prompt_renderer(
        {"current_instruction_doc": current_instruction, "dataset_with_feedback": dataset}
    )


@pytest.mark.parametrize("guidance", [None, "", "   ", "\n\t "])
def test_default_proposer_prompt_unchanged_without_guidance(guidance):
    lm = CapturingLM()
    adapter = _make_adapter(lm, reflection_instruction=guidance)

    adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    assert lm.prompts == [_baseline("current", DATASET)]


def test_reflection_instruction_appended_once_after_baseline():
    guidance = "Keep proposed instructions concise."
    lm = CapturingLM()
    adapter = _make_adapter(lm, reflection_instruction=guidance)

    adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    baseline = _baseline("current", DATASET)
    expected = f"{baseline}\n\n{HEADING}\n{guidance}"
    assert lm.prompts[0] == expected
    assert lm.prompts[0].startswith(baseline)
    assert lm.prompts[0].count(HEADING) == 1


def test_reflection_instruction_is_appended_literally():
    guidance = (
        "Use <curr_param> and <side_info> literally. {braces} and \\n and newlines\n"
        "second line \u2014 unicode \u2713 and `backticks`."
    )
    lm = CapturingLM()
    adapter = _make_adapter(lm, reflection_instruction=guidance)

    adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    baseline = _baseline("current", DATASET)
    assert lm.prompts[0] == f"{baseline}\n\n{HEADING}\n{guidance}"


def test_reflection_instruction_reaches_each_selected_component():
    guidance = "Be concise."
    lm = CapturingLM()
    adapter = _make_adapter(lm, reflection_instruction=guidance)
    candidate = {"a": "inst a", "b": "inst b", "c": "inst c"}
    reflective = {"a": DATASET, "b": DATASET, "c": DATASET}
    candidate_before = copy.deepcopy(candidate)
    reflective_before = copy.deepcopy(reflective)

    out = adapter.propose_new_texts(candidate, reflective, ["a", "b"])

    assert set(out) == {"a", "b"}  # the unselected component "c" is left alone
    assert len(lm.prompts) == 2
    for prompt, current in zip(lm.prompts, ["inst a", "inst b"], strict=True):
        assert prompt == f"{_baseline(current, DATASET)}\n\n{HEADING}\n{guidance}"
    # The candidate and the reflective records are never mutated by injection.
    assert candidate == candidate_before
    assert reflective == reflective_before
    assert out["a"] == "New instruction" and out["b"] == "New instruction"


def test_reflection_instruction_does_not_accumulate_across_calls():
    guidance = "Be concise."
    lm = CapturingLM()
    adapter = _make_adapter(lm, reflection_instruction=guidance)

    for _ in range(3):
        adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    expected = f"{_baseline('current', DATASET)}\n\n{HEADING}\n{guidance}"
    assert lm.prompts == [expected] * 3


def test_reflection_instruction_does_not_leak_between_instances():
    lm_with = CapturingLM()
    lm_without = CapturingLM()

    _make_adapter(lm_with, reflection_instruction="only here").propose_new_texts(
        {"predict": "current"}, {"predict": DATASET}, ["predict"]
    )
    _make_adapter(lm_without).propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    baseline = _baseline("current", DATASET)
    assert lm_with.prompts[0] == f"{baseline}\n\n{HEADING}\nonly here"
    assert lm_without.prompts[0] == baseline


def test_custom_proposer_ignores_reflection_instruction():
    captured = {}

    def custom_instruction_proposer(*, candidate, reflective_dataset, components_to_update):
        captured["candidate"] = candidate
        captured["reflective_dataset"] = reflective_dataset
        captured["components_to_update"] = components_to_update
        return dict.fromkeys(components_to_update, "from custom")

    lm = CapturingLM()
    adapter = _make_adapter(
        lm,
        reflection_instruction="should be ignored",
        custom_instruction_proposer=custom_instruction_proposer,
    )
    candidate = {"predict": "current"}
    reflective = {"predict": DATASET}

    out = adapter.propose_new_texts(candidate, reflective, ["predict"])

    assert out == {"predict": "from custom"}
    assert captured == {
        "candidate": candidate,
        "reflective_dataset": reflective,
        "components_to_update": ["predict"],
    }
    # The default proposer (and therefore the guidance-bearing callback) never runs.
    assert lm.prompts == []


def test_falsey_custom_proposer_still_wins_over_reflection_instruction():
    """A callable custom proposer wins even if its truthiness is False, matching the
    constructor's `is not None` warning check and the reflective-dataset handling."""
    captured = {}

    class FalseyProposer:
        def __bool__(self):
            return False

        def __call__(self, *, candidate, reflective_dataset, components_to_update):
            captured["called"] = True
            return dict.fromkeys(components_to_update, "from falsey")

    lm = CapturingLM()
    adapter = _make_adapter(
        lm,
        reflection_instruction="should be ignored",
        custom_instruction_proposer=FalseyProposer(),
    )

    out = adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    assert out == {"predict": "from falsey"}
    assert captured.get("called") is True
    # The guidance-bearing default callback never ran.
    assert lm.prompts == []


@pytest.mark.parametrize("response", ["```\nBetter instruction\n```", {"text": "```\nBetter instruction\n```"}])
def test_reflection_instruction_output_parsing(response):
    lm = CapturingLM([response])
    adapter = _make_adapter(lm, reflection_instruction="Be concise.")

    out = adapter.propose_new_texts({"predict": "current"}, {"predict": DATASET}, ["predict"])

    assert out == {"predict": "Better instruction"}
