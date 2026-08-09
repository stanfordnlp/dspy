"""Usage tracking utilities for DSPy."""

import copy
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel

from dspy.dsp.utils.settings import settings


class UsageTracker:
    """Tracks LM usage data within a context.

    Args:
        mirror: Optional enclosing tracker. Every entry added here is also forwarded to
            ``mirror`` immediately, so usage recorded under a nested or per-thread tracker
            reaches enclosing trackers in real time — it survives exceptions raised later in
            the nested scope and is visible to the enclosing scope as soon as each LM call
            completes, not only when the nested scope finishes.
    """

    def __init__(self, mirror: "UsageTracker | None" = None):
        # Map of LM name to list of usage entries. For example:
        # {
        #     "openai/gpt-4o-mini": [
        #         {"prompt_tokens": 100, "completion_tokens": 200},
        #         {"prompt_tokens": 300, "completion_tokens": 400},
        #     ],
        # }
        self.usage_data = defaultdict(list)
        # Trackers can be read and mirrored into across threads, so guard mutations.
        self._lock = threading.Lock()
        self._mirror = mirror

    def __deepcopy__(self, memo):
        # The copy is an independent tracker: it keeps the recorded data but not the
        # mirror link or the (uncopyable) lock.
        new_tracker = UsageTracker()
        with self._lock:
            new_tracker.usage_data = copy.deepcopy(self.usage_data, memo)
        return new_tracker

    def __getstate__(self):
        # Locks aren't picklable and the mirror chain shouldn't be serialized; trackers
        # can end up in pickled settings snapshots.
        state = self.__dict__.copy()
        del state["_lock"]
        del state["_mirror"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._mirror = None

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

    def add_usage(self, lm: str, usage_entry: dict[str, Any]) -> None:
        """Add a usage entry to the tracker (and to its mirror chain, if any)."""
        if len(usage_entry) > 0:
            with self._lock:
                self.usage_data[lm].append(self._flatten_usage_entry(usage_entry))
            if self._mirror is not None:
                self._mirror.add_usage(lm, usage_entry)

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        """Calculate total tokens from all tracked usage."""
        with self._lock:
            entries_by_lm = {lm: list(entries) for lm, entries in self.usage_data.items()}
        total_usage_by_lm = {}
        for lm, usage_entries in entries_by_lm.items():
            total_usage = {}
            for usage_entry in usage_entries:
                total_usage = self._merge_usage_entries(total_usage, usage_entry)
            total_usage_by_lm[lm] = total_usage
        return total_usage_by_lm

    def get_call_counts(self) -> dict[str, int]:
        """Number of tracked LM calls per LM (cache hits are never tracked)."""
        with self._lock:
            return {lm: len(entries) for lm, entries in self.usage_data.items()}


@contextmanager
def track_usage(mirror: UsageTracker | None = None) -> Generator[UsageTracker, None, None]:
    """Context manager for tracking LM usage.

    Args:
        mirror: Optional enclosing tracker that also receives, in real time, every entry
            recorded inside this context.
    """
    tracker = UsageTracker(mirror=mirror)

    with settings.context(usage_tracker=tracker):
        yield tracker
