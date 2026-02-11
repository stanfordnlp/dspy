import random

from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    """Attach labeled few-shot demonstrations to each predictor in a program.

    A simple teleprompter that assigns ``k`` examples from a training set to
    every predictor in the student program as demonstration examples. When
    sampling is enabled, uses a deterministic random selection (seed=0) for
    reproducibility.

    Args:
        k: Maximum number of demonstrations to attach to each predictor.
            Defaults to 16.

    Example:
        Compile a program with few-shot examples:

        ```python
        import dspy

        # Define a simple QA program
        qa = dspy.ChainOfThought("question -> answer")

        # Prepare training data
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
        ]

        # Compile with few-shot demonstrations
        teleprompter = dspy.LabeledFewShot(k=2)
        compiled_qa = teleprompter.compile(qa, trainset=trainset)
        ```
    """

    def __init__(self, k=16):
        self.k = k

    def compile(self, student, *, trainset, sample=True):
        self.student = student.reset_copy()
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        for predictor in self.student.predictors():
            if sample:
                predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

        return self.student


# NOTE: I believe templatev2 keeps rdemos as long as they have the last field.
# This may change later, especially with the introduction of required vs optional fields.
# NOTE: Since we're relying on downstream code to handle the demos, this sampling may be sub-sampled.
