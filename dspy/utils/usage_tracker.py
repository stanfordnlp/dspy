"""Usage tracking utilities for DSPy."""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any

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

    def _flatten_usage_entry(self, usage_entry) -> dict[str, dict[str, Any]]:
        result = dict(usage_entry)

        if result.get("completion_tokens_details"):
            result["completion_tokens_details"] = dict(result["completion_tokens_details"])
        if result.get("prompt_tokens_details"):
            result["prompt_tokens_details"] = dict(result["prompt_tokens_details"])
        return result

    def _merge_usage_entries(self, usage_entry1, usage_entry2) -> dict[str, dict[str, Any]]:
        if usage_entry1 is None or len(usage_entry1) == 0:
            return dict(usage_entry2)
        if usage_entry2 is None or len(usage_entry2) == 0:
            return dict(usage_entry1)

        result = dict(usage_entry2)
        for k, v in usage_entry1.items():
            if k in result:
                if isinstance(v, dict):
                    result[k] = self._merge_usage_entries(result[k], v)
                else:
                    result[k] = result[k] or 0
                    result[k] += v if v else 0
        return result

    def add_usage(self, lm: str, usage_entry: dict):
        """Add a usage entry to the tracker."""
        if len(usage_entry) > 0:
            self.usage_data[lm].append(self._flatten_usage_entry(usage_entry))

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
def track_usage():
    """Context manager for tracking LM usage."""
    tracker = UsageTracker()

    with settings.context(usage_tracker=tracker):
        yield tracker
