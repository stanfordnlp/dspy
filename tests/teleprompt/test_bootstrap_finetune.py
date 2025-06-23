import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFinetune
from dspy.utils.dummies import DummyLM


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!"),
]
trainset = [examples[0]]


def test_bootstrap_finetune_initialization():
    # Initialize BootstrapFinetune with a dummy metric and minimal setup
    bootstrap = BootstrapFinetune(metric=simple_metric)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"
    assert bootstrap.multitask == True, "Multitask should default to True"


class SimpleModule(dspy.Module):
    def __init__(self, signature, lm=None):
        super().__init__()
        self.predictor = Predict(signature)
        if lm:
            self.predictor.lm = lm

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_compile_with_predict_instances_no_explicit_lm():
    """Test BootstrapFinetune compile with predictors that don't have explicit LMs."""
    from unittest.mock import patch
    
    # Create student and teacher modules without explicit LMs in predictors
    # This tests the fix: lm = pred.lm or settings.lm
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    # Set up LM in settings - this will be the fallback
    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    original_lm = dspy.settings.lm
    dspy.settings.configure(lm=lm)


    # Verify that the predictor doesn't have an explicit LM
    assert student.predictor.lm is None

    # Initialize BootstrapFinetune - this should work without AttributeError
    bootstrap = BootstrapFinetune(metric=simple_metric)
    
    # Mock all the components that would fail without proper setup
    with patch('dspy.teleprompt.bootstrap_finetune.all_predictors_have_lms'), \
            patch('dspy.teleprompt.bootstrap_finetune.prepare_teacher', return_value=teacher), \
            patch('dspy.teleprompt.bootstrap_finetune.bootstrap_trace_data', return_value=[]), \
            patch.object(bootstrap, '_prepare_finetune_data', return_value=([], 'openai')), \
            patch.object(bootstrap, 'finetune_lms') as mock_finetune_lms:
        
        # Mock the finetune_lms to return a mapping from training key to LM
        mock_finetune_lms.return_value = {(lm, None): lm}
        
        # This should not raise AttributeError due to the fix
        compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)
        
        assert compiled_student is not None, "Failed to compile student"
        # Verify that finetune_lms was called
        mock_finetune_lms.assert_called_once()
        

