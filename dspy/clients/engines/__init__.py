"""Execution backends for DSPy's LM layer.

The outer LM layer selects these engines and owns caching, retries, fan-out
and bookkeeping. No engine silently delegates a failed request to another
backend.
"""

from dspy.clients.engines.base import AsyncEngine, Engine
from dspy.clients.engines.legacy_engine import AsyncLegacyEngine, LegacyEngine
from dspy.clients.engines.litellm_engine import AsyncLiteLLMEngine, LiteLLMEngine
from dspy.clients.engines.lm15_engine import AsyncLM15Engine, LM15Engine

__all__ = [
    "Engine",
    "AsyncEngine",
    "LM15Engine",
    "AsyncLM15Engine",
    "LiteLLMEngine",
    "AsyncLiteLLMEngine",
    "LegacyEngine",
    "AsyncLegacyEngine",
]
