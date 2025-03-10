from dataclasses import dataclass, field
from typing import ClassVar, Dict

from ..features import Features, Sequence, Value
from .base import TaskTemplate


@dataclass(frozen=True)
class QuestionAnsweringExtractive(TaskTemplate):
    # `task` is not a ClassVar since we want it to be part of the `asdict` output for JSON serialization
    task: str = field(default="question-answering-extractive", metadata={"include_in_asdict_even_if_is_default": True})
    input_schema: ClassVar[Features] = Features({"question": Value("string"), "context": Value("string")})
    label_schema: ClassVar[Features] = Features(
        {
            "answers": Sequence(
                {
                    "text": Value("string"),
                    "answer_start": Value("int32"),
                }
            )
        }
    )
    question_column: str = "question"
    context_column: str = "context"
    answers_column: str = "answers"

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {self.question_column: "question", self.context_column: "context", self.answers_column: "answers"}
