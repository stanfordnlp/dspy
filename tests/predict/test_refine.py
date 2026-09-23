import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        # The answer should always be one word.
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)

    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")


def test_refine_feedback_reaches_retry_with_instance_adapter():
    """Refine's hint wrapper must run even when the module has an instance adapter via `set_adapter()`."""
    adapter_calls = []

    class TrackingChatAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            adapter_calls.append(dict(inputs))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    def forward_fn(self, **kwargs):
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", forward_fn)
    tracking_adapter = TrackingChatAdapter()
    module.set_adapter(tracking_adapter)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # The first attempt goes through the configured adapter without a hint.
    assert any("hint_" not in inputs for inputs in adapter_calls)
    # The retry must go through the configured adapter *and* carry the feedback hint.
    hinted = [inputs for inputs in adapter_calls if "hint_" in inputs]
    assert hinted, "Feedback hint never reached the retry through the configured adapter"
    assert hinted[0]["hint_"] == "Answer with 'right'."
    # The original module must not be mutated.
    assert module.predictor.adapter is tracking_adapter


def test_refine_feedback_retry_with_stateful_adapter_requiring_constructor_args():
    """The hint wrapper must not re-instantiate adapters whose constructors require arguments."""
    adapter_calls = []

    class StatefulAdapter(dspy.ChatAdapter):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag
            self.calls = []

        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            call = dict(inputs)
            self.calls.append(call)
            adapter_calls.append((self.tag, call))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    stateful_adapter = StatefulAdapter(tag="configured")
    module.set_adapter(stateful_adapter)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # Refine runs a deep copy of the module. The external log proves that the copied
    # stateful adapter retained its required constructor state and received the hint.
    hinted = [(tag, inputs) for tag, inputs in adapter_calls if "hint_" in inputs]
    assert hinted == [("configured", {"question": "What is the right answer?", "hint_": "Answer with 'right'."})]
    # The original module and adapter remain unmodified.
    assert module.predictor.adapter is stateful_adapter
    assert type(stateful_adapter) is StatefulAdapter
    assert stateful_adapter.tag == "configured"
    assert stateful_adapter.calls == []


def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )
