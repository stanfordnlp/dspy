"""Normalized language model implementations and types."""

from dspy.clients.language_models.base import LanguageModel, LMCapabilities
from dspy.clients.language_models.types import (
    Assistant,
    AsyncLMStream,
    Developer,
    LMAudioDelta,
    LMAudioPart,
    LMBasePart,
    LMCacheConfig,
    LMCitationDelta,
    LMCitationPart,
    LMConfig,
    LMDelta,
    LMFilePart,
    LMHistoryEntry,
    LMImageDelta,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMOutputBuilder,
    LMPart,
    LMPromptCacheConfig,
    LMReasoningConfig,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMStream,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMTextPart,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
    System,
    LMToolCall,
    ToolResult,
    User,
)

# Curated set of names promoted to the top-level `dspy.*` namespace.
# Everything in `__all__` (below) remains importable from `dspy.clients.language_models`
# (and via the `dspy.lm` re-export shim), but only this subset becomes `dspy.X`.
# Hybrid policy decision from the migration plan (commit 1, refined commit 2:
# `LanguageModel` is internal scaffolding — users subclass `BaseLM`).
TOP_LEVEL_EXPORTS = [
    "LMCapabilities",
    "OpenAIChatLM",
    "OpenAITextLM",
    "OpenAIResponsesLM",
    "AnthropicLM",
    "GenAILM",
    "LiteLLMLM",
    "LMRequest",
    "LMResponse",
    "LMMessage",
    "LMConfig",
    "LMUsage",
    "LMHistoryEntry",
    "System",
    "Developer",
    "User",
    "Assistant",
    "LMToolCall",
    "ToolResult",
]

__all__ = [
    "LanguageModel",
    "LMCapabilities",
    "OpenAIChatLM",
    "OpenAITextLM",
    "OpenAIResponsesLM",
    "TOP_LEVEL_EXPORTS",
    "AnthropicLM",
    "GenAILM",
    "LiteLLMLM",
    "LMRouter",
    "register_lm_backend",
    "LMBasePart",
    "LMTextPart",
    "LMImagePart",
    "LMAudioPart",
    "LMFilePart",
    "LMToolCallPart",
    "LMToolResultPart",
    "LMThinkingPart",
    "LMCitationPart",
    "LMRefusalPart",
    "LMPart",
    "LMMessage",
    "LMToolSpec",
    "LMReasoningConfig",
    "LMToolChoice",
    "LMCacheConfig",
    "LMPromptCacheConfig",
    "LMConfig",
    "LMRequest",
    "LMUsage",
    "LMOutput",
    "LMResponse",
    "LMDelta",
    "LMTextDelta",
    "LMThinkingDelta",
    "LMToolCallDelta",
    "LMCitationDelta",
    "LMImageDelta",
    "LMAudioDelta",
    "LMStreamEvent",
    "LMStreamStartEvent",
    "LMStreamDeltaEvent",
    "LMStreamOutputEndEvent",
    "LMStreamEndEvent",
    "LMStreamErrorEvent",
    "LMOutputBuilder",
    "LMStream",
    "AsyncLMStream",
    "LMHistoryEntry",
    "System",
    "Developer",
    "User",
    "Assistant",
    "LMToolCall",
    "ToolResult",
]


# Lazy-load backend subclasses to avoid a circular import with `dspy.clients.base_lm`
# (backends subclass `BaseLM`, which itself imports from this package).
_LAZY_ATTRS = {
    "OpenAIChatLM": ("dspy.clients.language_models.openai", "OpenAIChatLM"),
    "OpenAITextLM": ("dspy.clients.language_models.openai", "OpenAITextLM"),
    "OpenAIResponsesLM": ("dspy.clients.language_models.openai", "OpenAIResponsesLM"),
    "AnthropicLM": ("dspy.clients.language_models.anthropic", "AnthropicLM"),
    "GenAILM": ("dspy.clients.language_models.gemini", "GenAILM"),
    "LiteLLMLM": ("dspy.clients.language_models.litellm", "LiteLLMLM"),
    "LMRouter": ("dspy.clients.language_models.router", "LMRouter"),
    "register_lm_backend": ("dspy.clients.language_models.router", "register_lm_backend"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib

        module_path, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
