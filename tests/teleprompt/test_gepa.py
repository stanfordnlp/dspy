import json
import threading
from typing import Any

import pytest

import dspy
import dspy.clients
from dspy import Example
from dspy.predict import Predict
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


def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")

    with open("tests/teleprompt/gepa_dummy_lm.json") as f:
        data = json.load(f)
    lm_history = data["lm"]
    reflection_lm_history = data["reflection_lm"]

    lm_main = DictDummyLM(lm_history)
    dspy.settings.configure(lm=lm_main)
    reflection_lm = DictDummyLM(reflection_lm_history)
    optimizer = dspy.GEPA(metric=simple_metric, reflection_lm=reflection_lm, max_metric_calls=5)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    o = optimizer.compile(student, trainset=trainset, valset=trainset)
    assert (
        o.predictor.signature.instructions
        == 'Given the field `input` containing a question or phrase, produce the field `output` containing the exact, direct, and contextually appropriate answer or response that the user expects, without additional explanations, commentary, or general knowledge unless explicitly requested.\n\nKey details and guidelines:\n\n1. The `input` field contains a question or phrase that may be literal, factual, or culturally specific (e.g., references to popular culture or memes).\n\n2. The `output` must be the precise answer or response that directly addresses the `input` as intended by the user, not a general or encyclopedic explanation.\n\n3. If the `input` is a well-known phrase or question from popular culture (e.g., "What does the fox say?"), the `output` should reflect the expected or canonical answer associated with that phrase, rather than a factual or scientific explanation.\n\n4. Avoid providing additional background information, scientific explanations, or alternative interpretations unless explicitly requested.\n\n5. The goal is to produce the answer that the user expects or the "correct" answer in the context of the question, including culturally recognized or meme-based answers.\n\n6. If the `input` is a straightforward factual question (e.g., "What is the color of the sky?"), provide the commonly accepted direct answer (e.g., "Blue") rather than a detailed scientific explanation.\n\n7. The output should be concise, clear, and focused solely on answering the question or phrase in the `input`.\n\nExample:\n\n- Input: "What is the color of the sky?"\n- Output: "Blue."\n\n- Input: "What does the fox say?"\n- Output: "Ring-ding-ding-ding-dingeringeding!"\n\nThis approach ensures that the assistant provides the expected, contextually appropriate answers rather than general or overly detailed responses that may be considered incorrect by the user.'
    )


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
