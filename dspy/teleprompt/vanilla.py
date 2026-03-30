import random

from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    """A simple teleprompter that assigns labeled training examples as few-shot demonstrations.

    Randomly samples up to ``k`` examples from the training set and attaches them
    as demonstrations to each predictor in the student program. No bootstrapping
    or optimization is performed—the labeled examples are used directly.

    Args:
        k: Maximum number of labeled demonstrations to attach per predictor.
            Defaults to 16.
    """

    def __init__(self, k=16):
        self.k = k

    def compile(self, student, *, trainset, sample=True):
        """Compile the student program by attaching labeled demonstrations.

        Args:
            student: The student :class:`dspy.Module` to compile.
            trainset: A list of :class:`dspy.Example` objects to sample demonstrations from.
            sample: If ``True``, randomly sample up to ``k`` examples. If ``False``,
                take the first ``k`` examples in order. Defaults to ``True``.

        Returns:
            The compiled student module with demonstrations attached to its predictors.
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
