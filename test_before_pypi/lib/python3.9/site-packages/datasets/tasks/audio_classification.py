import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ..features import Audio, ClassLabel, Features
from .base import TaskTemplate


@dataclass(frozen=True)
class AudioClassification(TaskTemplate):
    task: str = field(default="audio-classification", metadata={"include_in_asdict_even_if_is_default": True})
    input_schema: ClassVar[Features] = Features({"audio": Audio()})
    label_schema: ClassVar[Features] = Features({"labels": ClassLabel})
    audio_column: str = "audio"
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
            self.audio_column: "audio",
            self.label_column: "labels",
        }
