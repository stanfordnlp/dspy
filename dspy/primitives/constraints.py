
from typing import Callable


class Constraint:
    def __init__(self, fn: Callable, desc: str, soft: bool = False):
        self.fn = fn
        self.desc = desc
        self.soft = soft

    def __call__(self, inputs, outputs):
        return self.fn(inputs, outputs)
