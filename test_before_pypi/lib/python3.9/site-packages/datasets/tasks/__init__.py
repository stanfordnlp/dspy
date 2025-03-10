from typing import Optional

from ..utils.logging import get_logger
from .audio_classification import AudioClassification
from .automatic_speech_recognition import AutomaticSpeechRecognition
from .base import TaskTemplate
from .image_classification import ImageClassification
from .language_modeling import LanguageModeling
from .question_answering import QuestionAnsweringExtractive
from .summarization import Summarization
from .text_classification import TextClassification


__all__ = [
    "AutomaticSpeechRecognition",
    "AudioClassification",
    "ImageClassification",
    "LanguageModeling",
    "QuestionAnsweringExtractive",
    "Summarization",
    "TaskTemplate",
    "TextClassification",
]

logger = get_logger(__name__)


NAME2TEMPLATE = {
    AutomaticSpeechRecognition.task: AutomaticSpeechRecognition,
    AudioClassification.task: AudioClassification,
    ImageClassification.task: ImageClassification,
    LanguageModeling.task: LanguageModeling,
    QuestionAnsweringExtractive.task: QuestionAnsweringExtractive,
    Summarization.task: Summarization,
    TextClassification.task: TextClassification,
}


def task_template_from_dict(task_template_dict: dict) -> Optional[TaskTemplate]:
    """Create one of the supported task templates in :py:mod:`datasets.tasks` from a dictionary."""
    task_name = task_template_dict.get("task")
    if task_name is None:
        logger.warning(f"Couldn't find template for task '{task_name}'. Available templates: {list(NAME2TEMPLATE)}")
        return None
    template = NAME2TEMPLATE.get(task_name)
    return template.from_dict(task_template_dict)
