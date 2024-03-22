import types
from typing import List

import dsp
from dspy.teleprompt import BootstrapFewShot

from .teleprompt import Teleprompter


class KNNFewShot(Teleprompter):
    def __init__(self, KNN, k: int, trainset: List[dsp.Example]):
        self.KNN = KNN(k, trainset)

    def compile(self, student, *, teacher=None, trainset, valset=None):
        student_copy = student.reset_copy()

        def forward_pass(*args, **kwargs):
            knn_trainset = self.KNN(**kwargs)
            few_shot_bootstrap = BootstrapFewShot()
            compiled_program = few_shot_bootstrap.compile(
                student,
                teacher=teacher,
                trainset=knn_trainset,
                valset=valset,
            )
            return compiled_program(**kwargs)

        student_copy.forward = types.MethodType(forward_pass, student_copy)
        return student_copy
