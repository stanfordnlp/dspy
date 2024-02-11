import textwrap
import pytest
import dspy
from dspy.teleprompt.signature_opt_bayesian import BayesianSignatureOptimizer
from dspy.utils.dummies import DummyLM
from dspy import Example
import dsp

# Define a simple metric function for testing
def simple_metric(example, prediction):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output

# Example training and validation sets
trainset = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    Example(input="What is the capital of France?", output="Paris").with_inputs("input"),
    Example(input="What is the capital of Germany?", output="Berlin").with_inputs("input"),
]

def test_bayesian_signature_optimizer_initialization():
    optimizer = BayesianSignatureOptimizer(metric=simple_metric, n=10, init_temperature=1.4, verbose=True, track_stats=True)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.n == 10, "Incorrect 'n' parameter initialization"
    assert optimizer.init_temperature == 1.4, "Initial temperature not correctly initialized"
    assert optimizer.verbose is True, "Verbose flag not correctly initialized"
    assert optimizer.track_stats is True, "Track stats flag not correctly initialized"

class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        # SignatureOptimizer doesn't work with dspy.Predict
        self.predictor = dspy.ChainOfThought(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)

def test_signature_optimizer_optimization_process():
    dsp.settings.lm = DummyLM(["Optimized instruction 1", "Optimized instruction 2"])
    
    student = SimpleModule(signature='input -> output')
    
    optimizer = BayesianSignatureOptimizer(metric=simple_metric, n=10, init_temperature=1.4, verbose=False, track_stats=False)
    
    # Adjustments: Include required parameters for the compile method
    optimized_student = optimizer.compile(
        student=student, 
        devset=trainset, 
        optuna_trials_num=10, 
        max_bootstrapped_demos=3, 
        max_labeled_demos=5, 
        eval_kwargs={"num_threads": 1, "display_progress": False}
    )
    
    assert optimized_student != student, "Optimization did not modify the student"

def test_signature_optimizer_statistics_tracking():
    dsp.settings.lm = DummyLM(["Optimized instruction"])
    
    optimizer = BayesianSignatureOptimizer(metric=simple_metric, n=10, init_temperature=1.4, verbose=False, track_stats=True)
    
    student = SimpleModule(signature="input -> output")
    
    optimized_student = optimizer.compile(
        student=student, 
        devset=trainset, 
        optuna_trials_num=10, 
        max_bootstrapped_demos=3, 
        max_labeled_demos=5, 
        eval_kwargs={"num_threads": 1, "display_progress": False}
    )
    
    # Assuming the tracking mechanism is implemented correctly in BayesianSignatureOptimizer
    assert hasattr(optimized_student, 'trial_logs'), "Optimizer did not track optimization statistics"

def test_optimization_and_output_verification():
    optimizer = BayesianSignatureOptimizer(metric=simple_metric, n=10, init_temperature=1.4, verbose=False, track_stats=True,
                                           )
    dsp.settings.lm = DummyLM([
        "Optimized Prompt",
        "Optimized Prefix",
    ])
    
    student = SimpleModule("input -> output")
    
    # Compile the student with the optimizer
    optimized_student = optimizer.compile(
        student=student, 
        devset=trainset, 
        optuna_trials_num=10, 
        max_bootstrapped_demos=3, 
        max_labeled_demos=5, 
        eval_kwargs={"num_threads": 1, "display_progress": False}
    )
    
    # Simulate calling the optimized student with a new input
    test_input = "What is the capital of France?"
    prediction = optimized_student(input=test_input)

    print(dsp.settings.lm.get_convo(-1))
    
    assert prediction.output == "No more responses"

    assert dsp.settings.lm.get_convo(-1) == textwrap.dedent("""\
        Given the fields `input`, produce the fields `output`.

        ---

        Follow the following format.

        Input: ${input}
        Reasoning: Let's think step by step in order to ${produce the output}. We ...
        Output: ${output}

        ---

        Input: What is the capital of Germany?
        Output: Berlin

        Input: What is the color of the sky?
        Output: blue

        Input: What is the capital of France?
        Output: Paris

        Input: What does the fox say?
        Output: Ring-ding-ding-ding-dingeringeding!

        Input: What is the capital of France?
        Reasoning: Let's think step by step in order to No more responses
        Output: No more responses""")
