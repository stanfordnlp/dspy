import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ..features import ClassLabel, Features, Value
from .base import TaskTemplate


@dataclass(frozen=True)
class TextClassification(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task: str = field(default="text-classification", metadata={"include_in_asdict_even_if_is_default": True})
    input_schema: ClassVar[Features] = Features({"text": Value("string")})
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    text_column: str = "text"
    label_column: str = "labels"

    def align_with_features(self, features):
        if self.label_column not in features:
            raise ValueError(f"Column {self.label_column} is not present in features.")
        if not isinstance(features[self.label_column], ClassLabel):
            raise ValueError(f"Column {self.label_column} is not a ClassLabel.")
        task_template = copy.deepcopy(self)
        label_schema = self.label_schema.copy()
        label_schema["labels"] = features[self.label_column]
        task_template.__dict__["label_schema"] = label_schema
        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.text_column: "text",
            self.label_column: "labels",
        }
