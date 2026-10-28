from __future__ import annotations

import functools
import sys
import types
from typing import Any

from dspy.utils.lazy_import import require


@functools.cache
def _configure_litellm_defaults(litellm: types.ModuleType) -> None:
    """Apply DSPy's global LiteLLM defaults once when LiteLLM is first imported."""
    litellm.telemetry = False
    litellm.cache = None  # By default we disable LiteLLM cache and use DSPy on-disk cache.
    if not getattr(litellm, "_dspy_logging_configured", False):
        litellm.suppress_debug_info = True
        litellm._dspy_logging_configured = True


def _materialize_litellm(litellm: types.ModuleType) -> None:
    """Force LiteLLM's lazy module to execute, or raise the missing dependency error."""
    # `require()` returns either an importlib LazyLoader-backed module or a _MissingModule.
    # Accessing a real LiteLLM attribute forces LazyLoader execution; on _MissingModule it raises
    # the helpful install-hint ImportError immediately at the DSPy call site.
    _completion = litellm.completion


@functools.cache
def get_litellm(*, feature: str) -> Any:
    """Import LiteLLM, apply DSPy's defaults once, and return the module."""
    litellm = require("litellm", extra="litellm", feature=feature)
    _materialize_litellm(litellm)
    _configure_litellm_defaults(litellm)
    return litellm


def is_litellm_context_window_error(error: Exception) -> bool:
    """Return whether an exception is LiteLLM's context-window error, if LiteLLM is loaded."""
    litellm_module = sys.modules.get("litellm")
    context_window_error = getattr(litellm_module, "ContextWindowExceededError", None)
    return context_window_error is not None and isinstance(error, context_window_error)
