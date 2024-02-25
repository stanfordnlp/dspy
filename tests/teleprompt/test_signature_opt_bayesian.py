import textwrap
import pytest
import dspy
from dspy.teleprompt.signature_opt_bayesian import BayesianSignatureOptimizer
from dspy.utils.dummies import DummyLM
from dspy import Example


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    # Simplified metric for testing: true if prediction matches expected output
    return example.output == prediction.output


# Example training and validation sets
trainset = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(
        input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!"
    ).with_inputs("input"),
    Example(input="What is the capital of France?", output="Paris").with_inputs(
        "input"
    ),
    Example(input="What is the capital of Germany?", output="Berlin").with_inputs(
        "input"
    ),
]


def test_bayesian_signature_optimizer_initialization():
    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric, n=10, init_temperature=1.4, verbose=True, track_stats=True
    )
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.n == 10, "Incorrect 'n' parameter initialization"
    assert (
        optimizer.init_temperature == 1.4
    ), "Initial temperature not correctly initialized"
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
    # Make LM that is always right
    dspy.settings.configure(lm=DummyLM({ex.input: ex.output for ex in trainset}))

    student = SimpleModule(signature="input -> output")

    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    )

    # Adjustments: Include required parameters for the compile method
    optimized_student = optimizer.compile(
        student=student,
        devset=trainset,
        optuna_trials_num=10,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )

    assert len(optimized_student.predictor.demos) == 4


def test_signature_optimizer_bad_lm():
    dspy.settings.configure(
        lm=DummyLM([f"Optimized instruction {i}" for i in range(30)])
    )
    student = SimpleModule(signature="input -> output")
    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    )

    # Krista: when the code tries to generate bootstrapped examples, the examples are generated using DummyLM,
    # which only outputs "Optimized instruction i" this means that none of the bootstrapped examples are successful,
    # and therefore the set of examples that we're using to generate new prompts is empty
    with pytest.raises(ValueError):
        _optimized_student = optimizer.compile(
            student=student,
            devset=trainset,
            optuna_trials_num=10,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
            eval_kwargs={"num_threads": 1, "display_progress": False},
        )


def test_optimization_and_output_verification():
    # Make a language model that is always right, except on the last
    # example in the train set.
    lm = DummyLM({ex.input: ex.output for ex in trainset[:-1]}, follow_examples=True)
    dspy.settings.configure(lm=lm)

    optimizer = BayesianSignatureOptimizer(
        metric=simple_metric,
        n=10,
        init_temperature=1.4,
        verbose=False,
        track_stats=True,
    )

    student = SimpleModule("input -> output")

    # Compile the student with the optimizer
    optimized_student = optimizer.compile(
        student=student,
        devset=trainset,
        optuna_trials_num=10,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
        eval_kwargs={"num_threads": 1, "display_progress": False},
    )

    # Simulate calling the optimized student with a new input
    test_input = "What is the capital of France?"
    prediction = optimized_student(input=test_input)

    print("CORRECT ANSWER")
    print(lm.get_convo(-1))

    assert prediction.output == "blue"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields `input`, produce the fields `output`.

        ---

        Input: What does the fox say?
        Output: Ring-ding-ding-ding-dingeringeding!

        Input: What is the capital of Germany?
        Output: Berlin

        Input: What is the capital of France?
        Output: Paris

        ---

        Follow the following format.

        Input: ${input}
        Reasoning: Let's think step by step in order to ${produce the output}. We ...
        Output: ${output}

        ---

        Input: What is the color of the sky?
        Reasoning: Let's think step by step in order to blue
        Output: blue

        ---

        Input: What is the capital of France?
        Reasoning: Let's think step by step in order to blue
        Output: blue"""
    )
