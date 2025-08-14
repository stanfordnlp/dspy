import types

import dspy
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.raw = []

    def on_lm_raw_response(self, call_id, instance, response):
        self.raw.append((call_id, instance.model, response))


class DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.model = "dummy-model"


class DummyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        return DummyResponse()


def test_on_lm_raw_response_hook(monkeypatch):
    cb = RecordingCallback()
    lm = DummyLM(model="dummy-model")
    dspy.settings.configure(callbacks=[cb], lm=lm)
    pred = dspy.Predict("q->a")
    pred(q="hi")

    assert len(cb.raw) == 1
    call_id, model, response = cb.raw[0]
    assert model == "dummy-model"
    assert hasattr(response, "choices")
