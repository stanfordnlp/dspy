
from typing import Callable


class Constraint:
    def __init__(self, fn: Callable, desc: str, soft_constraint: bool = False):
        self.fn = fn
        self.desc = desc
        self.soft_constraint = soft_constraint

    def __call__(self, inputs, outputs):
        return self.fn(inputs, outputs)
