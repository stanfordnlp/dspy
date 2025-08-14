import types

import dspy
from dspy.utils.callback import BaseCallback


class RecordingCallback(BaseCallback):
    def __init__(self):
        self.raw = []


class DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1}
        self.model = "dummy-model"


class DummyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        return DummyResponse()
