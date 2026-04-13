"""GoodMem integration utilities for DSPy.

Provides :class:`GoodMemClient` (HTTP wrapper) and
:func:`make_goodmem_tools` (tool factory for ``dspy.Tool`` / ``dspy.ReAct``).
"""

from dspy.utils.goodmem.client import GoodMemClient
from dspy.utils.goodmem.tools import make_goodmem_tools

__all__ = ["GoodMemClient", "make_goodmem_tools"]
