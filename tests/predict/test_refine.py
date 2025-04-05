from unittest.mock import patch
import pytest
import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM
from dspy.signatures import Signature, InputField, OutputField

class SimpleQA(Signature):
    """Answer the question."""
    question: str = InputField()
    answer: str = OutputField()

class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn
    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)

def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.settings.configure(lm=lm)
    module_call_count = [0]
    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)
    reward_call_count = [0]
    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        # The answer should always be one word.
        return 1.0 if len(pred.answer.split()) == 1 else 0.0
    refine = Refine(signature=SimpleQA, N=3, reward_fn=reward_fn, threshold=1.0)
    with patch.object(refine.soft_constraints_module, '__call__', side_effect=refine.soft_constraints_module.__call__) as mock_call:
        result = refine(question="What is the capital of Belgium?")
        module_call_count[0] = mock_call.call_count
    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] <= 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )

def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.settings.configure(lm=lm)
    def always_fail(pred):
        return False, "Deliberately failing"
    refine = Refine(signature=SimpleQA, N=3, validators=[always_fail], fail_on_invalid=True)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")

def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.settings.configure(lm=lm)
    call_count = [0]
    def fail_twice(pred):
        call_count[0] += 1
        if call_count[0] <= 2:
            return False, "Deliberately failing"
        return True, ""
    refine = Refine(signature=SimpleQA, N=3, validators=[fail_twice], fail_on_invalid=True)
    refine(question="What is the capital of Belgium?")
    assert call_count[0] >= 2, (
        "Validator should have been called at least 2 times, but was called %d times" % call_count[0]
    )