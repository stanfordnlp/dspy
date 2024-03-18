from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import dspy

SomeModule = TypeVar("SomeModule", bound=dspy.Module)


class Teleprompter(ABC, Generic[SomeModule]):
    @abstractmethod
    def compile(self, student: SomeModule, **kwargs: Any) -> SomeModule:
        raise NotImplementedError()
