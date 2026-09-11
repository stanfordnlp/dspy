"""Usage tracking utilities for DSPy."""

import copy
import threading
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
        # Map of LM name to accumulated response cost, e.g. {"openai/gpt-4o-mini": 0.0021}.
        self.total_cost_by_model = defaultdict(float)
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        copied = type(self)()
        memo[id(self)] = copied
        with self._lock:
            copied.usage_data = copy.deepcopy(self.usage_data, memo)
            copied.total_cost_by_model = copy.deepcopy(self.total_cost_by_model, memo)
        return copied

    def _flatten_usage_entry(self, usage_entry: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in usage_entry.items():
            if isinstance(value, BaseModel):
                # Convert Pydantic models to dicts, like `PromptTokensDetailsWrapper` from litellm.
                result[key] = value.model_dump()
            else:
                result[key] = value
        return result

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
            elif current_v is not None or v is not None:
                result[k] = (current_v or 0) + (v or 0)
        return result

    def add_usage(self, lm: str, usage_entry: dict[str, Any], cost: float | None = None) -> None:
        """Add a usage entry, and optionally its response cost, to the tracker."""
        flattened_usage = self._flatten_usage_entry(usage_entry) if usage_entry else None
        normalized_cost = float(cost) if cost is not None else None
        with self._lock:
            if flattened_usage is not None:
                self.usage_data[lm].append(flattened_usage)
            if normalized_cost is not None:
                self.total_cost_by_model[lm] += normalized_cost

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        """Calculate total tokens from all tracked usage."""
        with self._lock:
            usage_data = copy.deepcopy(self.usage_data)

        total_usage_by_lm = {}
        for lm, usage_entries in usage_data.items():
            total_usage = {}
            for usage_entry in usage_entries:
                total_usage = self._merge_usage_entries(total_usage, usage_entry)
            total_usage_by_lm[lm] = total_usage
        return total_usage_by_lm

    def get_total_cost(self) -> float:
        """Calculate the total cost from all tracked usage, 0.0 when no costs were reported."""
        with self._lock:
            return sum(self.total_cost_by_model.values(), 0.0)


@contextmanager
def track_usage() -> Generator[UsageTracker, None, None]:
    """Context manager for tracking LM usage."""
    tracker = UsageTracker()

    with settings.context(usage_tracker=tracker):
        yield tracker
