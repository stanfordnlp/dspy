import json
import threading
from typing import Any
from unittest import mock

import pytest

import dspy
import dspy.clients
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt.gepa import instruction_proposal
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


class DictDummyLM(dspy.clients.lm.LM):
    def __init__(self, history):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.history = {}
        for m in history:
            self.history[hash(repr(m["messages"]))] = m

    def __call__(self, prompt=None, messages=None, **kwargs):
        assert hash(repr(messages)) in self.history, f"Message {messages} not found in history"
        m = self.history[hash(repr(messages))]
        return m["outputs"]


def simple_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return dspy.Prediction(score=example.output == prediction.output, feedback="Wrong answer.")


def bad_metric(example, prediction):
    return 0.0


@pytest.fixture
def mock_mlflow():
    mock_mlflow = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        yield mock_mlflow


@pytest.mark.parametrize("use_mlflow", [True, False])
def test_basic_workflow(use_mlflow, mock_mlflow):
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")

    with open("tests/teleprompt/gepa_dummy_lm.json") as f:
        data = json.load(f)
    lm_history = data["lm"]
    reflection_lm_history = data["reflection_lm"]

    lm_main = DictDummyLM(lm_history)
    dspy.settings.configure(lm=lm_main)
    reflection_lm = DictDummyLM(reflection_lm_history)

    optimizer = dspy.GEPA(
        metric=simple_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        use_mlflow=use_mlflow
    )


    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]

    optimized_program = optimizer.compile(student, trainset=trainset, valset=trainset)
    assert optimized_program.predictor.signature.instructions == 'Given the field `input` containing a question or phrase, produce the field `output` containing the exact, direct, and contextually appropriate answer or response that the user expects, without additional explanations, commentary, or general knowledge unless explicitly requested.\n\nKey details and guidelines:\n\n1. The `input` field contains a question or phrase that may be literal, factual, or culturally specific (e.g., references to popular culture or memes).\n\n2. The `output` must be the precise answer or response that directly addresses the `input` as intended by the user, not a general or encyclopedic explanation.\n\n3. If the `input` is a well-known phrase or question from popular culture (e.g., "What does the fox say?"), the `output` should reflect the expected or canonical answer associated with that phrase, rather than a factual or scientific explanation.\n\n4. Avoid providing additional background information, scientific explanations, or alternative interpretations unless explicitly requested.\n\n5. The goal is to produce the answer that the user expects or the "correct" answer in the context of the question, including culturally recognized or meme-based answers.\n\n6. If the `input` is a straightforward factual question (e.g., "What is the color of the sky?"), provide the commonly accepted direct answer (e.g., "Blue") rather than a detailed scientific explanation.\n\n7. The output should be concise, clear, and focused solely on answering the question or phrase in the `input`.\n\nExample:\n\n- Input: "What is the color of the sky?"\n- Output: "Blue."\n\n- Input: "What does the fox say?"\n- Output: "Ring-ding-ding-ding-dingeringeding!"\n\nThis approach ensures that the assistant provides the expected, contextually appropriate answers rather than general or overly detailed responses that may be considered incorrect by the user.'
    if use_mlflow:
        assert mock_mlflow.start_run.call_count == 1
    else:
        assert mock_mlflow.start_run.call_count == 0

def test_workflow_with_custom_instruction_proposer_and_component_selector():
    """Test to ensure the basic compile flow runs without errors when using a custom instruction proposer and component selector."""

    class TimeReader(dspy.Module):
        def __init__(self):
            super().__init__()
            self.hour_predictor = dspy.ChainOfThought("clock_photo: dspy.Image -> hour: int")
            self.minute_predictor = dspy.ChainOfThought("clock_photo: dspy.Image -> minute: int")

            self.parallel = dspy.Parallel(num_threads=2)

        def forward(self, clock_photo: dspy.Image):
            hour_prediction, minute_prediction = self.parallel(
                [
                    (self.hour_predictor, dict(clock_photo=clock_photo)),
                    (self.minute_predictor, dict(clock_photo=clock_photo)),
                ]
            )
            return dspy.Prediction(hour=hour_prediction.hour, minute=minute_prediction.minute)

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        target_hour, target_minute = example.hour, example.minute
        predicted_hour, predicted_minute = prediction.hour, prediction.minute

        score = -abs(target_hour * 60 + target_minute - (predicted_hour * 60 + predicted_minute))

        return dspy.Prediction(
            score=score,
            feedback=f"Target: {target_hour}:{target_minute}, Predicted: {predicted_hour}:{predicted_minute}",
        )

    def all_component_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
        """Select all components."""
        return list(candidate.keys())

    student = TimeReader()

    with open("tests/teleprompt/gepa_dummy_lm_custom_component_selector_custom_instruction_proposer.json") as f:
        data = json.load(f)

    lm_history = data["lm"]
    reflection_lm_history = data["reflection_lm"]

    lm_main = DictDummyLM(lm_history)
    reflection_lm = DictDummyLM(reflection_lm_history)

    dspy.settings.configure(lm=lm_main)
    optimizer = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        instruction_proposer=instruction_proposal.MultiModalInstructionProposer(),
        component_selector=all_component_selector,
        num_threads=16,
    )
    trainset = [
        Example(
            clock_photo=dspy.Image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Pendulum_clock_by_Jacob_Kock%2C_antique_furniture_photography%2C_IMG_0931_edit.jpg/500px-Pendulum_clock_by_Jacob_Kock%2C_antique_furniture_photography%2C_IMG_0931_edit.jpg",
                download=False,
            ),
            hour=8,
            minute=18,
        ).with_inputs("clock_photo"),
        Example(
            clock_photo=dspy.Image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Telechron_clock_2H07-Br_Administrator.JPG/960px-Telechron_clock_2H07-Br_Administrator.JPG",
                download=False,
            ),
            hour=4,
            minute=16,
        ).with_inputs("clock_photo"),
    ]
    o = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert o.hour_predictor.predict.signature.instructions == "Task\n- Input: clock_photo (an image of an analog clock)\n- Output: hour (an integer 1\u201312). Output only the hour number with no extra text.\n\nGoal\n- Determine the correct hour by accurately identifying the hour hand and its position relative to the hour marks, taking into account the minute hand\u2019s position (since the hour hand moves continuously between numbers).\n\nStep-by-step procedure\n1) Find the dial and pivot\n- Locate the clock face and the central pivot where all hands originate.\n- Ignore decorative elements that do not originate at the central pivot (e.g., ornaments, shadows, reflections).\n\n2) Determine the 12 o\u2019clock direction\n- Prefer the numeral \u201c12\u201d if visible. Otherwise use the upright orientation of numerals or the topmost marker.\n- If the photo is rotated, mentally rotate so numerals read upright: 12 at top, 3 right, 6 bottom, 9 left.\n\n3) Identify the hands correctly (do not assume a default \u201c10:10\u201d)\n- Second hand: thinnest, often with a counterweight, may span very long; ignore for the hour.\n- Minute hand: longest, usually reaches or nearly reaches the outer minute tick marks.\n- Hour hand: shortest, usually thicker, typically ends well inside the numerals.\n- If ambiguous, classify by tip distance from center: minute \u2265 hour. Use the piece actually anchored at the pivot, not its shadow.\n\n4) Measure positions (angles)\n- Measure each hand\u2019s angle clockwise from 12 o\u2019clock.\n- Minute angle \u03b8m \u2248 position of the minute hand; hour angle \u03b8h \u2248 position of the hour hand.\n\n5) Use minute-hand position to validate the hour-hand location\n- The hour hand advances 0.5\u00b0 per minute (i.e., 1/12 of the distance between hour marks every 5 minutes).\n- Sanity check examples:\n  - ~15 minutes past: hour hand \u2248 1/4 of the way from the current hour toward the next.\n  - ~30 minutes: \u2248 halfway.\n  - ~45 minutes: \u2248 3/4 of the way.\n- If this relationship doesn\u2019t hold, you likely swapped hour and minute hands\u2014re-identify them.\n\n6) Determine the hour\n- Compute the \u201clast passed\u201d hour: H = floor((\u03b8h mod 360) / 30). Map 0 to 12 (i.e., if floor(...) = 0, H = 12).\n- Do not round up to the next hour. The correct hour is the number the hour hand has most recently passed, not the one it is approaching.\n- If the hour hand appears exactly on an hour mark but the minute hand is not at 12, treat it as still between hours and choose the lower (last passed) hour.\n\n7) Edge cases and robustness\n- Stylized or missing numerals: rely on the 12/3/6/9 axes and tick marks rather than numeral shapes.\n- Roman numerals: \u201c4\u201d may be IIII; positions are unchanged.\n- Ignore mirrored effects, reflections, and shadows; only consider hands anchored at the pivot.\n- Overlap times: if hands nearly overlap, use \u03b8m to ensure the hour hand offset matches 0.5\u00b0 per minute.\n- Return 12, not 0, when appropriate (e.g., just after 12:00).\n\nOutput format\n- Provide only: hour as an integer in [1,12], with no additional text.\n\nCommon error prevention (from prior mistakes)\n- Do not confuse the minute hand for the hour hand; verify by length and reach to the outer tick marks.\n- Do not infer times like \u201c10:10\u201d by default; always read from the actual hand angles.\n- Ensure the hour chosen matches the \u201clast passed\u201d number given the minute hand\u2019s position (e.g., at ~:16, the hour hand must be just past the hour, not near 1 when the minute hand is at 3)."
    assert o.minute_predictor.predict.signature.instructions == "Task: From the image field clock_photo (an analog clock), output the minute value as an integer from 0\u201359 in the field minute. Output only the minute number\u2014no text or other fields.\n\nWhat to analyze\n- Clock face orientation: Identify where \u201c12\u201d is on the dial. Use the numerals (Arabic or Roman, stylized fonts) or the positions of 3, 6, 9, 12 to set the reference. If the photo is tilted, measure angles relative to the clock face, not the image frame.\n- Hands identification (do not confuse them):\n  - Minute hand: typically the longest solid hand reaching near the minute ticks/outer ring; thicker than the second hand; often has a pronounced pointer tip.\n  - Hour hand: shorter and thicker, typically ends near the numerals.\n  - Second hand (if present): the thinnest, often the longest, usually with a counterweight; ignore it for minute reading.\n  - If two non-second hands look similar, the one whose tip reaches closer to the minute tick ring is the minute hand.\n- Ticks and numerals: Each numeral-to-numeral segment equals 5 minutes. If minute tick marks exist, use them. If not, divide each numeral interval evenly into five.\n\nHow to compute the minute\n1. Locate the clock center and the minute hand\u2019s tip.\n2. Determine the angle of the minute hand from the 12 o\u2019clock direction, increasing clockwise.\n3. Convert angle to minutes: minute_estimate = (angle_from_12 / 6). Round to the nearest whole minute.\n   - Mapping: 12 \u2192 0, 1 \u2192 5, 2 \u2192 10, 3 \u2192 15, 4 \u2192 20, 5 \u2192 25, 6 \u2192 30, 7 \u2192 35, 8 \u2192 40, 9 \u2192 45, 10 \u2192 50, 11 \u2192 55.\n   - If the tip is slightly past a numeral (e.g., just past 3), do not snap to the numeral; round to the nearest minute (e.g., 16 instead of 15).\n4. Consistency check with the hour hand (useful to avoid off-by-one and hand mix-ups):\n   - The hour hand moves continuously: it advances 0.5 degrees per minute (i.e., 1/12 of the way to the next numeral every 5 minutes).\n   - If your minute_estimate is an exact multiple of 5 but the hour hand is clearly between hour markers (not aligned with an hour), re-examine: the minute hand is likely slightly past the numeral; adjust to the nearest minute accordingly.\n   - If the minute hand choice is ambiguous, infer the minute from the hour hand\u2019s fraction toward the next hour: minute \u2248 fraction_between_hour_markers \u00d7 60, then choose the hand assignment that matches this.\n5. Edge cases:\n   - Overlapping hands: Look at which tip extends farther toward the tick ring to identify the minute hand.\n   - Strong perspective or glare: Use the line from center to the visible tip; ignore reflections.\n   - No minute ticks: Evenly interpolate between numerals.\n   - Subdials or decorative elements (e.g., pendulum windows) are not the minute indicator; use the main dial only.\n\nOutput format\n- Return only the integer minute value (0\u201359) in the minute field.\n- If the angle computes to 60, output 0.\n\nError prevention reminders\n- Do not treat the hour hand as the minute hand.\n- Do not use the second hand to compute minutes.\n- Do not assume the minute hand is exactly on a numeral\u2014check for slight offsets and round to the nearest minute.\n- Ensure the final minute agrees with the hour hand\u2019s position trend (hour hand slightly past an hour implies minutes > 0)."


def test_metric_requires_feedback_signature():
    reflection_lm = DictDummyLM([])
    with pytest.raises(TypeError):
        dspy.GEPA(metric=bad_metric, reflection_lm=reflection_lm, max_metric_calls=1)


def any_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """
    For this test, we only care that the program runs, not the score.
    """
    return 0.0  # ← Just returns 0.0, doesn't access any attributes!


def test_gepa_compile_with_track_usage_no_tuple_error(caplog):
    """
    GEPA.compile should not log tuple-usage error when track_usage=True and complete without hanging.
    Before, compile would hang and/or log "'tuple' object has no attribute 'set_lm_usage'" repeatedly.
    """
    student = dspy.Predict("question -> answer")
    trainset = [dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")]

    task_lm = DummyLM([{"answer": "mock answer 1"}])
    reflection_lm = DummyLM([{"new_instruction": "Something new."}])

    compiled_container: dict[str, Any] = {}
    exc_container: dict[str, BaseException] = {}

    def run_compile():
        try:
            with dspy.context(lm=task_lm, track_usage=True):
                optimizer = dspy.GEPA(metric=any_metric, reflection_lm=reflection_lm, max_metric_calls=3)
                compiled_container["prog"] = optimizer.compile(student, trainset=trainset, valset=trainset)
        except BaseException as e:
            exc_container["e"] = e

    t = threading.Thread(target=run_compile, daemon=True)
    t.start()
    t.join(timeout=1.0)

    # Assert compile did not hang (pre-fix behavior would time out here)
    assert not t.is_alive(), "GEPA.compile did not complete within timeout (likely pre-fix behavior)."

    # Assert no tuple-usage error is logged anymore
    assert "'tuple' object has no attribute 'set_lm_usage'" not in caplog.text

    # If any exception occurred, fail explicitly
    if "e" in exc_container:
        pytest.fail(f"GEPA.compile raised unexpectedly: {exc_container['e']}")

    # No timeout, no exception -> so the program must exist
    if "prog" not in compiled_container:
        pytest.fail("GEPA.compile did return a program (likely pre-fix behavior).")


class MultiComponentModule(dspy.Module):
    """Test module with multiple predictors."""

    def __init__(self):
        super().__init__()
        self.classifier = Predict("input -> category")
        self.generator = Predict("category, input -> output")

    def forward(self, input):
        category = self.classifier(input=input).category
        output = self.generator(category=category, input=input).output
        return dspy.Prediction(category=category, output=output)


def component_selection_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Simple metric for component selection testing."""
    return dspy.Prediction(score=0.3, feedback="Test feedback")


def test_component_selector_functionality():
    """Test custom component selector function can select single/multiple components."""

    # Track calls for verification
    selector_calls = []

    def test_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
        selector_calls.append({"components": list(candidate.keys()), "candidate_idx": candidate_idx})
        # Test both single and multiple selection
        return ["classifier"] if candidate_idx == 0 else ["classifier", "generator"]

    student = MultiComponentModule()

    # Provide enough responses for all possible LM calls during optimization
    task_lm = DummyLM([{"category": "test_category", "output": "test_output"}] * 20)
    reflection_lm = DummyLM(
        [
            {"improved_instruction": "Improved classifier instruction"},
            {"improved_instruction": "Improved generator instruction"},
        ]
        * 10
    )
    trainset = [dspy.Example(input="test", output="expected").with_inputs("input")]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=component_selection_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=6,  # Reduced to minimize output
            component_selector=test_selector,
        )
        result = optimizer.compile(student, trainset=trainset, valset=trainset)

    # Verify selector was called with correct parameters
    assert len(selector_calls) > 0, "Custom selector should be invoked"
    assert "classifier" in selector_calls[0]["components"], "Should receive all available components"
    assert "generator" in selector_calls[0]["components"], "Should receive all available components"
    assert result is not None, "Should return optimized program"


def test_component_selector_default_behavior():
    """Test default behavior when no custom selector provided."""
    student = MultiComponentModule()

    # Provide enough responses for all possible LM calls
    task_lm = DummyLM([{"category": "test_category", "output": "test_output"}] * 15)
    reflection_lm = DummyLM([{"improved_instruction": "Better instruction"}] * 8)
    trainset = [dspy.Example(input="test", output="expected").with_inputs("input")]

    with dspy.context(lm=task_lm):
        # No component_selector - should use round-robin default
        optimizer = dspy.GEPA(
            metric=component_selection_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=4,  # Minimal calls to reduce noise
        )
        result = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert result is not None, "Should work with default selector"


def test_component_selector_string_round_robin():
    """Test string-based round_robin selector."""
    student = MultiComponentModule()

    # Provide enough responses for all possible LM calls
    task_lm = DummyLM([{"category": "test_category", "output": "test_output"}] * 15)
    reflection_lm = DummyLM([{"improved_instruction": "Better instruction"}] * 8)
    trainset = [dspy.Example(input="test", output="expected").with_inputs("input")]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=component_selection_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=4,
            component_selector="round_robin",  # String-based selector
        )
        result = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert result is not None, "Should work with 'round_robin' string selector"


def test_component_selector_string_all():
    """Test string-based 'all' selector and verify it actually updates all components."""
    student = MultiComponentModule()

    # Store original instructions to verify they get updated
    original_classifier_instruction = student.classifier.signature.instructions
    original_generator_instruction = student.generator.signature.instructions

    def optimize(component_selector):
        # Metric that progressively improves to encourage GEPA to accept new candidates
        call_count = 0

        def improving_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
            nonlocal call_count
            call_count += 1
            # Score improves with each call to encourage acceptance of new candidates
            score = min(0.3 + (call_count * 0.1), 1.0)
            return dspy.Prediction(score=score, feedback="Improving feedback")

        # Provide enough responses for all possible LM calls
        task_lm = DummyLM([{"category": "test_category", "output": "test_output"}] * 20)
        reflection_lm = DummyLM(
            [
                {"improved_instruction": "Updated classifier instruction"},
                {"improved_instruction": "Updated generator instruction"},
            ]
            * 10
        )
        trainset = [dspy.Example(input="test", output="expected").with_inputs("input")]

        with dspy.context(lm=task_lm):
            optimizer = dspy.GEPA(
                metric=improving_metric,
                reflection_lm=reflection_lm,
                max_metric_calls=8,
                component_selector=component_selector,
                track_stats=True,  # Track intermediate results to verify updates
            )
            return optimizer.compile(student, trainset=trainset, valset=trainset)

    result_round_robin = optimize(component_selector="round_robin")

    candidates_round_robin = result_round_robin.detailed_results.candidates

    assert (
        candidates_round_robin[1].classifier.signature.instructions == original_classifier_instruction
        and candidates_round_robin[1].generator.signature.instructions != original_generator_instruction
    ) or (
        candidates_round_robin[1].classifier.signature.instructions != original_classifier_instruction
        and candidates_round_robin[1].generator.signature.instructions == original_generator_instruction
    ), "First candidate should have only one component updated, when using round_robin selector"

    result_all = optimize(component_selector="all")

    candidates_all = result_all.detailed_results.candidates

    assert (
        candidates_all[1].classifier.signature.instructions != original_classifier_instruction
        and candidates_all[1].generator.signature.instructions != original_generator_instruction
    ), "First candidate should have both components updated, when using all selector"


def test_component_selector_custom_random():
    """Test custom component selector function that randomly samples components."""
    import random

    # Simple function-based selector
    def random_component_selector(state, trajectories, subsample_scores, candidate_idx, candidate):
        """Randomly select half of the available components."""
        component_names = list(candidate.keys())
        num_to_select = max(1, len(component_names) // 2)  # At least 1, half of total
        return random.sample(component_names, num_to_select)

    student = MultiComponentModule()

    # Provide enough responses for all possible LM calls
    task_lm = DummyLM([{"category": "test_category", "output": "test_output"}] * 15)
    reflection_lm = DummyLM([{"improved_instruction": "Better instruction"}] * 8)
    trainset = [dspy.Example(input="test", output="expected").with_inputs("input")]

    with dspy.context(lm=task_lm):
        optimizer = dspy.GEPA(
            metric=component_selection_metric,
            reflection_lm=reflection_lm,
            max_metric_calls=4,
            component_selector=random_component_selector,  # Function-based selector
        )
        result = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert result is not None, "Should work with custom random function selector"
