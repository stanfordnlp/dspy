"""Shared infrastructure for Apple on-device language model adapters.

This module contains the OpenAI-compatible response dataclasses and helpers
shared by :mod:`dspy.clients.apple_fm` and :mod:`dspy.clients.apple_local`,
plus the :class:`_AppleBaseLM` base that both concrete adapters inherit from.

These are private implementation details; only ``AppleFoundationLM`` and
``AppleLocalLM`` are part of the public DSPy API.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Iterator

from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic OpenAI-compatible response types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FMMessage:
    """A single message in a completion response, mirroring OpenAI's format.

    Attributes:
        content: The text content of the message.
        tool_calls: Optional list of tool-call objects returned by the model.
    """

    content: str
    tool_calls: list[Any] | None = None


@dataclasses.dataclass
class _FMChoice:
    """One completion choice, mirroring OpenAI's ``Choice`` object.

    Attributes:
        message: The assistant message produced for this choice.
    """

    message: _FMMessage


@dataclasses.dataclass
class _FMUsage:
    """Token-usage statistics for a completion, mirroring OpenAI's ``Usage`` object.

    Implements the mapping protocol (``__iter__``) so that ``dict(usage)``
    works as expected by ``BaseLM`` history tracking.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Yield (key, value) pairs so that ``dict(usage)`` works as expected by BaseLM.

        Yields:
            Two-tuples of field name and integer value for each token count.
        """
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclasses.dataclass
class _FMResponse:
    """A completion response object compatible with ``BaseLM._process_completion``.

    Mirrors the subset of OpenAI's ``ChatCompletion`` that DSPy accesses.

    Attributes:
        choices: List of completion choices (always length 1 for Apple adapters).
        usage: Token-usage breakdown for the request.
        model: Model identifier string, echoed from the request.
        _hidden_params: DSPy internal metadata dict.  ``response_cost`` is set
            to ``0.0`` because on-device inference has no monetary cost.
    """

    choices: list[_FMChoice]
    usage: _FMUsage
    model: str
    # BaseLM reads getattr(response, "_hidden_params", {}).get("response_cost").
    # On-device inference has no monetary cost; declaring the field explicitly
    # satisfies the interface without relying on the getattr fallback.
    _hidden_params: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten a list of role/content message dicts into a single prompt string.

    Apple's ``LanguageModelSession.respond()`` takes a plain string rather than
    a structured message list.  System instructions are included as plain context
    at the top — bracket prefixes such as ``[System]:`` trigger Apple's on-device
    content guardrails and must be avoided.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts following the
            OpenAI chat format.  Multi-modal ``content`` lists are supported;
            only text blocks are extracted.

    Returns:
        A single string with all non-empty message contents joined by ``"\\n\\n"``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content blocks — extract text parts only.
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        if not content:
            continue
        if role == "system":
            # No bracket prefix — "[System]:" triggers Apple's on-device content
            # guardrails (pattern-matched as a jailbreak attempt).  System content
            # is included as plain context at the top of the prompt, which is
            # semantically correct for a flat string.
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class _AppleBaseLM(BaseLM):
    """Shared base class for Apple on-device language model adapters.

    Provides the ``_build_response`` factory method used by both
    :class:`~dspy.clients.apple_fm.AppleFoundationLM` and
    :class:`~dspy.clients.apple_local.AppleLocalLM`.  Not intended for direct
    instantiation — use one of the concrete subclasses instead.
    """

    def _build_response(self, text: str, usage: _FMUsage | None = None) -> _FMResponse:
        """Wrap a raw text string in an OpenAI-compatible ``_FMResponse``.

        Args:
            text: The model's generated text.
            usage: Pre-computed token-usage statistics.  Pass ``None`` (or omit)
                to get zeroed counters — appropriate when the underlying SDK does
                not expose token counts (e.g. ``AppleFoundationLM``).

        Returns:
            An ``_FMResponse`` with a single choice and ``response_cost=0.0``.
        """
        return _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content=text))],
            usage=usage or _FMUsage(),
            model=self.model,
            _hidden_params={"response_cost": 0.0},
        )
