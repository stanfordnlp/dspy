import random

from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    """Teleprompter that attaches labeled demonstrations to each predictor.

    `LabeledFewShot` is the simplest prompt optimizer in DSPy. It selects up to
    ``k`` examples from a labeled training set and attaches them as fixed
    demonstrations to every predictor in the student program. No LM calls are
    made during compilation.

    Use this when you have a small, high-quality labeled dataset and want a
    deterministic, cost-free baseline before trying heavier optimizers such as
    :class:`~dspy.teleprompt.BootstrapFewShot`.

    Attributes:
        k: Maximum number of demonstrations to attach per predictor.

    Example:
        ```python
        import dspy

        # Define a simple QA module
        qa = dspy.ChainOfThought("question -> answer")

        # A small labeled training set
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
            dspy.Example(question="Color of the sky?", answer="Blue").with_inputs("question"),
        ]

        optimizer = dspy.LabeledFewShot(k=2)
        compiled_qa = optimizer.compile(qa, trainset=trainset)

        result = compiled_qa(question="What is 3+3?")
        print(result.answer)
        ```
    """

    def __init__(self, k: int = 16):
        """Initialize LabeledFewShot.

        Args:
            k: Maximum number of demonstrations to sample from the training set
                and attach to each predictor. When the training set has fewer
                than ``k`` examples, all examples are used. Defaults to ``16``.
        """
        self.k = k

    def compile(self, student, *, trainset, sample: bool = True):
        """Compile the student program by attaching labeled demonstrations.

        Selects up to ``self.k`` examples from ``trainset`` and assigns them as
        ``demos`` to every predictor inside ``student``. When ``sample=True`` the
        selection is a random sample (seeded for reproducibility); otherwise the
        first ``k`` examples in list order are used.

        If ``trainset`` is empty the student is returned unchanged.

        Args:
            student: A DSPy :class:`~dspy.Module` whose predictors will receive
                the sampled demonstrations.
            trainset: Sequence of :class:`~dspy.Example` objects to draw
                demonstrations from. Must be a keyword argument.
            sample: When ``True`` (default) draw a random sample of at most
                ``self.k`` examples from ``trainset``. When ``False``, take the
                first ``self.k`` examples in insertion order.

        Returns:
            The compiled student program with demonstrations attached to each
            of its predictors.

        Example:
            ```python
            import dspy

            qa = dspy.ChainOfThought("question -> answer")
            trainset = [
                dspy.Example(question="1+1?", answer="2").with_inputs("question"),
                dspy.Example(question="2+2?", answer="4").with_inputs("question"),
            ]

            optimizer = dspy.LabeledFewShot(k=1)

            # Random sample (default)
            compiled = optimizer.compile(qa, trainset=trainset)

            # Deterministic first-k selection
            compiled_det = optimizer.compile(qa, trainset=trainset, sample=False)
            ```
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
