"""Usage tracking utilities for DSPy."""

from contextlib import contextmanager

from dspy.dsp.utils.settings import settings


class UsageTracker:
    """Tracks LM usage data within a context."""

    def __init__(self):
        self.usage_data = []

    def add_usage(self, usage_entry: dict):
        """Add a usage entry to the tracker."""
        self.usage_data.append(usage_entry)

    @property
    def total_cost(self) -> float:
        """Calculate total cost from all tracked usage."""
        return sum(entry.get("cost", 0) or 0 for entry in self.usage_data)

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens from all tracked usage."""
        return sum(
            sum(entry.get("usage", {}).get(k, 0) for k in ["prompt_tokens", "completion_tokens"])
            for entry in self.usage_data
        )


@contextmanager
def track_usage():
    """Context manager for tracking LM usage."""
    tracker = UsageTracker()

    with settings.context(usage_tracker=tracker):
        yield tracker

