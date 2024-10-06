import textwrap

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShot
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
valset = [examples[1]]


def test_bootstrap_initialization():
    # Initialize BootstrapFewShot with a dummy metric and minimal setup
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_compile_with_predict_instances():
    # Create Predict instances for student and teacher
    # Note that dspy.Predict is not itself a module, so we can't use it directly here
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.settings.configure(lm=lm)

    # Initialize BootstrapFewShot and compile the student
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    assert compiled_student is not None, "Failed to compile student"
    assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"


def test_bootstrap_effectiveness():
    # This test verifies if the bootstrapping process improves the student's predictions
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")
    lm = DummyLM(
        ["[[ ## output ## ]]\nblue", "[[ ## output ## ]]\nRing-ding-ding-ding-dingeringeding!"], follow_examples=True
    )
    dspy.settings.configure(lm=lm, trace=[])

    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    # Check that the compiled student has the correct demos
    assert len(compiled_student.predictor.demos) == 1
    assert compiled_student.predictor.demos[0].input == trainset[0].input
    assert compiled_student.predictor.demos[0].output == trainset[0].output

    # Test the compiled student's prediction.
    # We are using a DummyLM with follow_examples=True, which means that
    # even though it would normally reply with "Ring-ding-ding-ding-dingeringeding!"
    # on the second output, if it seems an example that perfectly matches the
    # prompt, it will use that instead. That is why we expect "blue" here.
    prediction = compiled_student(input=trainset[0].input)
    assert prediction.output == trainset[0].output

    assert lm.get_convo(-1)[0] == [
        {
            "role": "system",
            "content": textwrap.dedent(
                """\
                Your input fields are:
                1. `input` (str)

                Your output fields are:
                1. `output` (str)

                All interactions will be structured in the following way, with the appropriate values filled in.

                [[ ## input ## ]]
                {input}

                [[ ## output ## ]]
                {output}

                [[ ## completed ## ]]

                In adhering to this structure, your objective is: 
                        Given the fields `input`, produce the fields `output`."""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## input ## ]]
                What is the color of the sky?

                Respond with the corresponding output fields, starting with the field `output`, and then ending with the marker for `completed`."""
            ),
        },
        {
            "role": "assistant",
            "content": textwrap.dedent(
                """\
                [[ ## output ## ]]
                blue

                [[ ## completed ## ]]"""
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                """\
                [[ ## input ## ]]
                What is the color of the sky?

                Respond with the corresponding output fields, starting with the field `output`, and then ending with the marker for `completed`."""
            ),
        },
    ]


def test_error_handling_during_bootstrap():
    """
    Test to verify error handling during the bootstrapping process
    """

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
            "[[ ## output ## ]]\nInitial thoughts",  # Simulate initial teacher's prediction
        ]
    )
    dspy.settings.configure(lm=lm)

    bootstrap = BootstrapFewShot(
        metric=simple_metric,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        max_errors=1,
    )

    with pytest.raises(RuntimeError, match="Simulated error"):
        bootstrap.compile(student, teacher=teacher, trainset=trainset)


def test_validation_set_usage():
    """
    Test to ensure the validation set is correctly used during bootstrapping
    """
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(
        [
            "[[ ## output ## ]]\nInitial thoughts",
            "[[ ## output ## ]]\nFinish[blue]",  # Expected output for both training and validation
        ]
    )
    dspy.settings.configure(lm=lm)

    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

    # Check that validation examples are part of student's demos after compilation
    assert len(compiled_student.predictor.demos) >= len(valset), "Validation set not used in compiled student demos"
