import copy
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ..features import Audio, Features, Value
from .base import TaskTemplate


@dataclass(frozen=True)
class AutomaticSpeechRecognition(TaskTemplate):
    task: str = field(default="automatic-speech-recognition", metadata={"include_in_asdict_even_if_is_default": True})
    input_schema: ClassVar[Features] = Features({"audio": Audio()})
    label_schema: ClassVar[Features] = Features({"transcription": Value("string")})
    audio_column: str = "audio"
    transcription_column: str = "transcription"

    def align_with_features(self, features):
        if self.audio_column not in features:
            raise ValueError(f"Column {self.audio_column} is not present in features.")
        if not isinstance(features[self.audio_column], Audio):
            raise ValueError(f"Column {self.audio_column} is not an Audio type.")
        task_template = copy.deepcopy(self)
        input_schema = self.input_schema.copy()
        input_schema["audio"] = features[self.audio_column]
        task_template.__dict__["input_schema"] = input_schema
        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.audio_column: "audio", self.transcription_column: "transcription"}
