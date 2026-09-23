"""Stand-in decision clients and generative LMs for the ReAnchor tests."""

import copy
import re
import threading

from dspy.clients.engines.dummy_engine import AsyncDummyEngine, DummyEngine
from dspy.utils.dummies import DummyLM


class FakeClient:
    """A System One client. Answers each question with `answer(state, name, question)`, and records every request."""

    supports_decision_requests = True

    def __init__(self, answer, cache=True):
        self.answer = answer
        self.cache = cache
        self.calls = []
        self.history = []
        self.callbacks = []

    def __call__(self, state, questions):
        self.calls.append(copy.deepcopy((state, questions)))
        return {name: self.answer(state, name, q) for name, q in questions.items()}

    async def acall(self, state, questions):
        return self(state, questions)


class _ComputedEngine(DummyEngine):
    _lock = threading.Lock()

    def _complete_messages(self, messages):
        with self._lock:
            self.owner.answers = {"": self.owner.answer(messages)}
            return super()._complete_messages(messages)


class ComputedLM(DummyLM):
    """A generative LM whose output field values are `answer(inputs, evidence)`.

    `inputs` maps each input field to its text in the final message, and `evidence` is True when
    the request asks for probabilities instead of native values.
    """

    def __init__(self, answer, adapter=None):
        super().__init__({}, adapter=adapter)
        self.answer_fn = answer
        self._engine_spec = _ComputedEngine(self)
        self._async_engine_spec = AsyncDummyEngine(self._engine_spec)

    def answer(self, messages):
        inputs = dict(re.findall(r"\[\[ ## (\w+) ## \]\]\n(.*)", messages[-1]["content"]))
        evidence = any("noul" in m["content"] or "probabilities" in m["content"] for m in messages)
        return self.answer_fn(inputs, evidence)


def noul(p: float) -> dict:
    return {"noul": p}


def score(probabilities: dict[int, float], confidence: float = 0.8) -> dict:
    return {"score": 0.0, "confidence": confidence, "probabilities": probabilities}


def choice(probabilities: dict[str, float], confidence: float = 0.7) -> dict:
    return {"confidence": confidence, "probabilities": probabilities}
