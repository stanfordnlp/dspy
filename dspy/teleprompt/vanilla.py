import random

from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    """A simple teleprompter that assigns labeled training examples as few-shot demos.

    Unlike bootstrapping-based optimizers, ``LabeledFewShot`` does not run the
    student program or evaluate any metric. It simply selects up to ``k`` examples
    from the provided ``trainset`` and attaches them directly as demonstrations to
    every predictor in the student program.

    Args:
        k: Maximum number of labeled demonstrations to attach to each predictor.
            Defaults to 16.

    Examples:
        ```python
        import dspy

        qa = dspy.ChainOfThought("question -> answer")
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What color is the sky?", answer="Blue").with_inputs("question"),
        ]

        teleprompter = dspy.LabeledFewShot(k=2)
        compiled_qa = teleprompter.compile(qa, trainset=trainset)
        ```
    """

    def __init__(self, k=16):
        self.k = k

    def compile(self, student, *, trainset, sample=True):
        """Compile the student program with labeled few-shot demonstrations.

        Attaches up to ``k`` examples from ``trainset`` as demonstrations to each
        predictor in a reset copy of ``student``. When ``sample=True`` (default),
        examples are drawn randomly (seeded for reproducibility); otherwise the
        first ``k`` examples are used in order.

        Args:
            student: The DSPy program to compile. Each of its predictors will
                receive the selected demonstrations.
            trainset: A list of labeled ``dspy.Example`` objects to draw
                demonstrations from.
            sample: If ``True``, randomly sample up to ``k`` examples from
                ``trainset``. If ``False``, take the first ``k`` examples in
                order. Defaults to ``True``.

        Returns:
            A compiled copy of ``student`` whose predictors have their ``demos``
            field populated with labeled examples from ``trainset``.
        """
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
