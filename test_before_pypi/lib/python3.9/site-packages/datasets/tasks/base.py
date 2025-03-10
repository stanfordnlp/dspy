import abc
import copy
import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Dict, Type, TypeVar

from ..features import Features


T = TypeVar("T", bound="TaskTemplate")


@dataclass(frozen=True)
class TaskTemplate(abc.ABC):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task: str
    input_schema: ClassVar[Features]
    label_schema: ClassVar[Features]

    def align_with_features(self: T, features: Features) -> T:
        """
        Align features with the task template.
        """
        # No-op
        return copy.deepcopy(self)

    @property
    def features(self) -> Features:
        return Features(**self.input_schema, **self.label_schema)

    @property
    @abc.abstractmethod
    def column_mapping(self) -> Dict[str, str]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls: Type[T], template_dict: dict) -> T:
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in template_dict.items() if k in field_names})
