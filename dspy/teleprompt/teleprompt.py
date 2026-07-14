from typing import Any

from dspy.primitives import Example, Module
from dspy.utils.callback import with_callbacks


class Teleprompter:
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        compile_method = cls.__dict__.get("compile")
        if compile_method is not None:
            cls.compile = with_callbacks(compile_method, _callback_kind="optimizer")

    def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
        """
        Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            teacher: The teacher program to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.
        """
        return self.__dict__
