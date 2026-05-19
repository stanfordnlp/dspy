"""Shared global LM history list.

The unified `BaseLM` and the underlying `LanguageModel` scaffolding both append
call entries to a single list. Hosting it in this leaf module avoids the
circular import that arises from importing across the two base modules.
"""

from typing import Any

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY: list[dict[str, Any]] = []
