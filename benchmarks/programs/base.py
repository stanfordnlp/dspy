"""
Base classes and utilities for DSPy programs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

if TYPE_CHECKING:
    from dspy import Module, Prediction


class BaseProgram:
    """Base class for DSPy programs."""
    
    @property  
    def name(self) -> str:
        """Return the program name."""
        raise NotImplementedError("Subclasses must implement the name property")


def format_context(context: dict | list[str] | str | None) -> str:
    """Format context into a string suitable for prompting.
    
    Handles multiple formats:
    - HotPotQA dict format: {'title': [...], 'sentences': [[...], ...]}
    - List of strings: ["passage 1", "passage 2", ...]
    - Single string: "passage text"
    """
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return "\n\n".join(context)
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        # HotPotQA format
        passages = []
        for title, sentences in zip(context["title"], context["sentences"]):
            text = " ".join(sentences)
            passages.append(f"[{title}]\n{text}")
        return "\n\n".join(passages)
    return str(context)