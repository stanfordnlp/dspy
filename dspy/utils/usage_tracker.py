"""Usage tracking utilities for DSPy."""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel

from dspy.dsp.utils.settings import settings


class UsageTracker:
    """Tracks LM usage data within a context."""

    def __init__(self):
        # Map of LM name to list of usage entries. For example:
        # {
        #     "openai/gpt-4o-mini": [
        #         {"prompt_tokens": 100, "completion_tokens": 200},
        #         {"prompt_tokens": 300, "completion_tokens": 400},
        #     ],
        # }
        self.usage_data = defaultdict(list)

    def _flatten_usage_entry(self, usage_entry: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in usage_entry.items():
            if isinstance(value, BaseModel):
                # Convert Pydantic models to dicts, like `PromptTokensDetailsWrapper` from litellm.
                result[key] = value.model_dump()
            else:
                result[key] = value
        return result

    def _is_summable(self, value: Any) -> bool:
        """Return True for values that should be numerically summed when merging entries.

        bool is excluded despite being an int subclass: tallying `is_byok=True` twice
        would yield 2 instead of True, which is wrong.
        """
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _merge_usage_entries(
        self, usage_entry1: dict[str, Any] | None, usage_entry2: dict[str, Any] | None
    ) -> dict[str, Any]:
        if usage_entry1 is None or len(usage_entry1) == 0:
            return dict(usage_entry2)
        if usage_entry2 is None or len(usage_entry2) == 0:
            return dict(usage_entry1)

        result = dict(usage_entry2)
        for k, v in usage_entry1.items():
            current_v = result.get(k)
            if isinstance(v, dict) or isinstance(current_v, dict):
                result[k] = self._merge_usage_entries(current_v, v)
            elif self._is_summable(v) or self._is_summable(current_v):
                result[k] = (current_v or 0) + (v or 0)
            # else: non-summable (str, bool, …) — usage_entry2's value already in result; leave it.
        return result

    def add_usage(self, lm: str, usage_entry: dict[str, Any]) -> None:
        """Add a usage entry to the tracker."""
        if len(usage_entry) > 0:
            self.usage_data[lm].append(self._flatten_usage_entry(usage_entry))

    def merge(self, other: "UsageTracker") -> None:
        """Absorb every usage entry recorded by `other` into this tracker.

        Entries stored on a tracker are already flattened by `_flatten_usage_entry`,
        so they can be appended directly without going through `add_usage` (which would
        re-flatten them and re-apply the `len > 0` guard unnecessarily).
        Used to roll a nested `track_usage()` scope up into its parent.
        """
        for lm, usage_entries in other.usage_data.items():
            self.usage_data[lm].extend(usage_entries)

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        """Calculate total tokens from all tracked usage."""
        total_usage_by_lm = {}
        for lm, usage_entries in self.usage_data.items():
            total_usage = {}
            for usage_entry in usage_entries:
                total_usage = self._merge_usage_entries(total_usage, usage_entry)
            total_usage_by_lm[lm] = total_usage
        return total_usage_by_lm


@contextmanager
def track_usage() -> Generator[UsageTracker, None, None]:
    """Context manager for tracking LM usage.

    Each LM call is recorded by the innermost active tracker. On exit, a nested tracker
    rolls its usage up into the enclosing one, so an outer block always reflects
    everything that happened inside it (including work done by nested scopes).
    """
    tracker = UsageTracker()
    parent_tracker = settings.usage_tracker

    try:
        with settings.context(usage_tracker=tracker):
            yield tracker
    finally:
        # Roll up even when the block raises: those tokens were still spent.
        if parent_tracker is not None:
            parent_tracker.merge(tracker)
