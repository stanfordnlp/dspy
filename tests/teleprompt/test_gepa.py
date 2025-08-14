import json

import dspy
import dspy.clients
from dspy import Example
from dspy.predict import Predict


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

def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

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
        max_metric_calls=5
    )
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    o = optimizer.compile(student, trainset=trainset, valset=trainset)
    assert o.predictor.signature.instructions == 'Given the field `input` containing a question or phrase, produce the field `output` containing the exact, direct, and contextually appropriate answer or response that the user expects, without additional explanations, commentary, or general knowledge unless explicitly requested.\n\nKey details and guidelines:\n\n1. The `input` field contains a question or phrase that may be literal, factual, or culturally specific (e.g., references to popular culture or memes).\n\n2. The `output` must be the precise answer or response that directly addresses the `input` as intended by the user, not a general or encyclopedic explanation.\n\n3. If the `input` is a well-known phrase or question from popular culture (e.g., "What does the fox say?"), the `output` should reflect the expected or canonical answer associated with that phrase, rather than a factual or scientific explanation.\n\n4. Avoid providing additional background information, scientific explanations, or alternative interpretations unless explicitly requested.\n\n5. The goal is to produce the answer that the user expects or the "correct" answer in the context of the question, including culturally recognized or meme-based answers.\n\n6. If the `input` is a straightforward factual question (e.g., "What is the color of the sky?"), provide the commonly accepted direct answer (e.g., "Blue") rather than a detailed scientific explanation.\n\n7. The output should be concise, clear, and focused solely on answering the question or phrase in the `input`.\n\nExample:\n\n- Input: "What is the color of the sky?"\n- Output: "Blue."\n\n- Input: "What does the fox say?"\n- Output: "Ring-ding-ding-ding-dingeringeding!"\n\nThis approach ensures that the assistant provides the expected, contextually appropriate answers rather than general or overly detailed responses that may be considered incorrect by the user.'
