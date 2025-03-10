from typing import Dict, Literal

from ._common import TaskProviderHelper
from .black_forest_labs import BlackForestLabsTextToImageTask
from .fal_ai import (
    FalAIAutomaticSpeechRecognitionTask,
    FalAITextToImageTask,
    FalAITextToSpeechTask,
    FalAITextToVideoTask,
)
from .fireworks_ai import FireworksAIConversationalTask
from .hf_inference import HFInferenceBinaryInputTask, HFInferenceConversational, HFInferenceTask
from .hyperbolic import HyperbolicTextGenerationTask, HyperbolicTextToImageTask
from .nebius import NebiusConversationalTask, NebiusTextGenerationTask, NebiusTextToImageTask
from .novita import NovitaConversationalTask, NovitaTextGenerationTask
from .replicate import ReplicateTask, ReplicateTextToSpeechTask
from .sambanova import SambanovaConversationalTask
from .together import TogetherConversationalTask, TogetherTextGenerationTask, TogetherTextToImageTask


PROVIDER_T = Literal[
    "black-forest-labs",
    "fal-ai",
    "fireworks-ai",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "replicate",
    "sambanova",
    "together",
]

PROVIDERS: Dict[PROVIDER_T, Dict[str, TaskProviderHelper]] = {
    "black-forest-labs": {
        "text-to-image": BlackForestLabsTextToImageTask(),
    },
    "fal-ai": {
        "automatic-speech-recognition": FalAIAutomaticSpeechRecognitionTask(),
        "text-to-image": FalAITextToImageTask(),
        "text-to-speech": FalAITextToSpeechTask(),
        "text-to-video": FalAITextToVideoTask(),
    },
    "fireworks-ai": {
        "conversational": FireworksAIConversationalTask(),
    },
    "hf-inference": {
        "text-to-image": HFInferenceTask("text-to-image"),
        "conversational": HFInferenceConversational(),
        "text-generation": HFInferenceTask("text-generation"),
        "text-classification": HFInferenceTask("text-classification"),
        "question-answering": HFInferenceTask("question-answering"),
        "audio-classification": HFInferenceBinaryInputTask("audio-classification"),
        "automatic-speech-recognition": HFInferenceBinaryInputTask("automatic-speech-recognition"),
        "fill-mask": HFInferenceTask("fill-mask"),
        "feature-extraction": HFInferenceTask("feature-extraction"),
        "image-classification": HFInferenceBinaryInputTask("image-classification"),
        "image-segmentation": HFInferenceBinaryInputTask("image-segmentation"),
        "document-question-answering": HFInferenceTask("document-question-answering"),
        "image-to-text": HFInferenceBinaryInputTask("image-to-text"),
        "object-detection": HFInferenceBinaryInputTask("object-detection"),
        "audio-to-audio": HFInferenceBinaryInputTask("audio-to-audio"),
        "zero-shot-image-classification": HFInferenceBinaryInputTask("zero-shot-image-classification"),
        "zero-shot-classification": HFInferenceTask("zero-shot-classification"),
        "image-to-image": HFInferenceBinaryInputTask("image-to-image"),
        "sentence-similarity": HFInferenceTask("sentence-similarity"),
        "table-question-answering": HFInferenceTask("table-question-answering"),
        "tabular-classification": HFInferenceTask("tabular-classification"),
        "text-to-speech": HFInferenceTask("text-to-speech"),
        "token-classification": HFInferenceTask("token-classification"),
        "translation": HFInferenceTask("translation"),
        "summarization": HFInferenceTask("summarization"),
        "visual-question-answering": HFInferenceBinaryInputTask("visual-question-answering"),
    },
    "hyperbolic": {
        "text-to-image": HyperbolicTextToImageTask(),
        "conversational": HyperbolicTextGenerationTask("conversational"),
        "text-generation": HyperbolicTextGenerationTask("text-generation"),
    },
    "nebius": {
        "text-to-image": NebiusTextToImageTask(),
        "conversational": NebiusConversationalTask(),
        "text-generation": NebiusTextGenerationTask(),
    },
    "novita": {
        "text-generation": NovitaTextGenerationTask(),
        "conversational": NovitaConversationalTask(),
    },
    "replicate": {
        "text-to-image": ReplicateTask("text-to-image"),
        "text-to-speech": ReplicateTextToSpeechTask(),
        "text-to-video": ReplicateTask("text-to-video"),
    },
    "sambanova": {
        "conversational": SambanovaConversationalTask(),
    },
    "together": {
        "text-to-image": TogetherTextToImageTask(),
        "conversational": TogetherConversationalTask(),
        "text-generation": TogetherTextGenerationTask(),
    },
}


def get_provider_helper(provider: PROVIDER_T, task: str) -> TaskProviderHelper:
    """Get provider helper instance by name and task.

    Args:
        provider (str): Name of the provider
        task (str): Name of the task

    Returns:
        TaskProviderHelper: Helper instance for the specified provider and task

    Raises:
        ValueError: If provider or task is not supported
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Provider '{provider}' not supported. Available providers: {list(PROVIDERS.keys())}")
    if task not in PROVIDERS[provider]:
        raise ValueError(
            f"Task '{task}' not supported for provider '{provider}'. "
            f"Available tasks: {list(PROVIDERS[provider].keys())}"
        )
    return PROVIDERS[provider][task]
