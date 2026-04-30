import random

from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""


class Ensemble(Teleprompter):
    """A teleprompter that combines multiple compiled programs into a single ensemble.

    At inference time, the returned ``EnsembledProgram`` runs a subset (or all) of
    the compiled programs and aggregates their outputs via ``reduce_fn``. A typical
    use-case is majority voting with ``reduce_fn=dspy.majority``.

    Args:
        reduce_fn: A callable that takes a list of program outputs and returns a
            single aggregated output. A common choice is ``dspy.majority``.
            If ``None``, the raw list of outputs is returned. Defaults to ``None``.
        size: If set, randomly sample this many programs from the ensemble at each
            forward call. If ``None``, all programs are used. Defaults to ``None``.
        deterministic: Reserved for future use. Must be ``False``. Defaults to ``False``.

    Examples:
        ```python
        import dspy

        # Suppose we have several compiled QA programs
        programs = [compiled_qa_v1, compiled_qa_v2, compiled_qa_v3]

        teleprompter = dspy.Ensemble(reduce_fn=dspy.majority)
        ensemble_qa = teleprompter.compile(programs)

        result = ensemble_qa(question="What is the capital of France?")
        ```
    """

    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        """Combine a list of programs into a single ``EnsembledProgram``.

        Args:
            programs: A list of compiled DSPy programs (``dspy.Module`` instances)
                to ensemble together.

        Returns:
            An ``EnsembledProgram`` that, on each forward call, runs ``size``
            randomly sampled programs (or all programs if ``size`` is ``None``)
            and returns either the reduced result (when ``reduce_fn`` is set) or
            the raw list of outputs.
        """
        size = self.size
        reduce_fn = self.reduce_fn

        import dspy

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                """Run the ensemble and return the aggregated (or raw) outputs."""
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

        return EnsembledProgram()
