import dsp
import random

from .teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    def __init__(self, k=16, only_reset_uncompiled=False):
        self.k = k
        self.only_reset_uncompiled = only_reset_uncompiled

    def compile(self, student, *, trainset, sample=True):
        self.student = student.reset_copy(only_reset_uncompiled=self.only_reset_uncompiled)
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        for predictor in self.student.predictors(only_uncompiled=True):
            if sample:
                predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                predictor.demos = self.trainset[:min(self.k, len(self.trainset))]

        return self.student
    
# NOTE: I believe templatev2 keeps rdemos as long as they have the last field.
# This may change later, especially with the introduction of required vs optional fields.
# NOTE: Since we're relying on downstream code to handle the demos, this sampling may be sub-sampled.
