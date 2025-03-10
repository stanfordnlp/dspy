import random

from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn

        import dspy

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

        return EnsembledProgram()
