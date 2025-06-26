import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFinetune
from dspy.utils.dummies import DummyLM


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]
trainset = [examples[0]]


def test_bootstrap_finetune_initialization():
    """Test BootstrapFinetune initialization with various parameters."""
    bootstrap = BootstrapFinetune(metric=simple_metric)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"
    assert bootstrap.multitask == True, "Multitask should default to True"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_error_handling_during_bootstrap():
    """Test error handling during the bootstrapping process."""

    class BuggyModule(dspy.Module):
        def __init__(self, signature):
            super().__init__()
            self.predictor = Predict(signature)

        def forward(self, **kwargs):
            raise RuntimeError("Simulated error")

    student = SimpleModule("input -> output")
    teacher = BuggyModule("input -> output")

    # Setup DummyLM to simulate an error scenario
    lm = DummyLM(
        [
            {"output": "Initial thoughts"},  # Simulate initial teacher's prediction
        ]
    )
    dspy.settings.configure(lm=lm)

    bootstrap = BootstrapFinetune(
        metric=simple_metric,
        max_errors=1,
    )

    with pytest.raises(RuntimeError, match="Simulated error"):
        bootstrap.compile(student, teacher=teacher, trainset=trainset)
