import random
from typing import Optional, TypedDict

import dspy

from .teleprompt import Teleprompter


class LabeledFewShotCompileKwargs(TypedDict):
    trainset: list[dspy.Example]
    sample: bool


class LabeledFewShot(Teleprompter):
    def __init__(self, k: int = 16):
        self.k = k

        self.student: Optional[dspy.Module] = None
        self.trainset: Optional[list[dspy.Example]] = None

    def compile(
        self,
        student: dspy.Module,
        *,
        trainset: Optional[list[dspy.Example]] = None,
        sample: bool = True,
        **_,
    ) -> dspy.Module:
        self.student = student.reset_copy()
        assert self.student is not None, "self.student was None!"
        self.trainset = trainset if trainset else []

        if not self.trainset:
            return self.student

        rng = random.Random(0)

        for predictor in self.student.predictors():
            if sample:
                predictor.demos = rng.sample(
                    self.trainset,
                    min(self.k, len(self.trainset)),
                )
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

        return self.student


# NOTE: I believe templatev2 keeps rdemos as long as they have the last field.
# This may change later, especially with the introduction of required vs optional fields.
# NOTE: Since we're relying on downstream code to handle the demos, this sampling may be sub-sampled.
