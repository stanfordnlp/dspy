import pytest
from unittest.mock import patch

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

    # Setup DummyLM to simulate an error scenario
    lm = DummyLM(
        [
            {"output": "Initial thoughts"},  # Simulate initial teacher's prediction
        ]
    )
    dspy.settings.configure(lm=lm)

    student = SimpleModule("input -> output")
    teacher = BuggyModule("input -> output")
    
    # Set LM for the student module
    student.set_lm(lm)
    teacher.set_lm(lm)

    bootstrap = BootstrapFinetune(
        metric=simple_metric,
    )

    # Mock the fine-tuning process since DummyLM doesn't support it
    with patch.object(bootstrap, 'finetune_lms') as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        
        # The bootstrap should complete successfully even with a buggy teacher
        # because we now handle exceptions gracefully
        compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)
        assert compiled_student is not None, "Bootstrap should complete successfully despite teacher errors"
        
        # Verify that fine-tuning was attempted (but with empty data due to the failed teacher)
        mock_finetune.assert_called_once()
