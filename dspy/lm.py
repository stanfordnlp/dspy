"""Public namespace for the normalized DSPy LM type system.

`dspy.lm.*` re-exports the full set of LM request/response/streaming types and
helpers from `dspy.clients.language_models`. The most common user-facing names
are also promoted to top-level `dspy.*` (see
`dspy.clients.language_models.TOP_LEVEL_EXPORTS` for the curated subset). Use
`dspy.lm.X` to reach the rest (parts, deltas, stream events, internal
config types) without polluting the top-level namespace.
"""

from dspy.clients.language_models import *  # noqa: F401,F403
from dspy.clients.language_models import __all__ as _full_all

__all__ = list(_full_all)
