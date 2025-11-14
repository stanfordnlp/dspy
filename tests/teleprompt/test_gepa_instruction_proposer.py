from dataclasses import dataclass
from typing import Any

import dspy
from dspy.teleprompt.gepa import instruction_proposal
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


def test_default_proposer():
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
    )

    gepa.compile(student, trainset=examples, valset=examples)

    assert len(lm.history) > 0, "LM should have been called"
    assert len(reflection_lm.history) > 0, "Reflection LM should have been called"

    images_in_history = check_images_in_history(reflection_lm.history)

    assert images_in_history.has_text_serialized_images, (
        "Expected to find serialized images (CUSTOM-TYPE-START-IDENTIFIER)"
    )
