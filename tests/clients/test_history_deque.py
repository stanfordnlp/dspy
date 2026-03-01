"""Tests for deque-based history in BaseLM and GLOBAL_HISTORY.

Verifies that history uses collections.deque for O(1) eviction instead of
list.pop(0) which is O(n).
"""

from collections import deque

import pytest

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY, MAX_HISTORY_SIZE
from dspy.utils.dummies import DummyLM


@pytest.fixture(autouse=True)
def clear_history():
    GLOBAL_HISTORY.clear()
    yield


class TestGlobalHistoryDeque:
    def test_global_history_is_deque(self):
        assert isinstance(GLOBAL_HISTORY, deque)

    def test_global_history_has_maxlen(self):
        assert GLOBAL_HISTORY.maxlen == MAX_HISTORY_SIZE

    def test_global_history_auto_evicts(self):
        """GLOBAL_HISTORY deque with maxlen automatically evicts oldest entries."""
        maxlen = GLOBAL_HISTORY.maxlen

        # Append more entries than maxlen to trigger auto-eviction on GLOBAL_HISTORY.
        for i in range(maxlen + 2):
            GLOBAL_HISTORY.append({"i": i})

        assert len(GLOBAL_HISTORY) == maxlen

        # After overfilling by 2, the first retained value should correspond to i == 2.
        remaining_indices = [entry["i"] for entry in GLOBAL_HISTORY]
        assert remaining_indices == list(range(2, maxlen + 2))

    def test_global_history_append_and_iterate(self):
        lm = DummyLM([{"response": "a"}, {"response": "b"}])
        dspy.configure(lm=lm)
        predictor = dspy.Predict("q: str -> response: str")
        predictor(q="first")
        predictor(q="second")

        assert len(GLOBAL_HISTORY) == 2
        entries = list(GLOBAL_HISTORY)
        assert all(isinstance(e, dict) for e in entries)

    def test_global_history_supports_negative_indexing(self):
        lm = DummyLM([{"response": "a"}, {"response": "b"}])
        dspy.configure(lm=lm)
        predictor = dspy.Predict("q: str -> response: str")
        predictor(q="first")
        predictor(q="second")

        last = GLOBAL_HISTORY[-1]
        assert isinstance(last, dict)
        assert "messages" in last

    def test_global_history_clear(self):
        lm = DummyLM([{"response": "a"}])
        dspy.configure(lm=lm)
        predictor = dspy.Predict("q: str -> response: str")
        predictor(q="test")

        assert len(GLOBAL_HISTORY) > 0
        GLOBAL_HISTORY.clear()
        assert len(GLOBAL_HISTORY) == 0
        assert isinstance(GLOBAL_HISTORY, deque)


class TestLMHistoryDeque:
    def test_lm_history_is_deque(self):
        lm = DummyLM([])
        assert isinstance(lm.history, deque)

    def test_lm_history_append_and_access(self):
        lm = DummyLM([{"response": "a"}, {"response": "b"}])
        dspy.configure(lm=lm)
        predictor = dspy.Predict("q: str -> response: str")
        predictor(q="first")
        predictor(q="second")

        assert isinstance(lm.history, deque)
        assert len(lm.history) == 2
        assert lm.history[0]["outputs"] is not None
        assert lm.history[-1]["outputs"] is not None

    def test_lm_history_eviction_uses_popleft(self):
        """Verify that history eviction uses O(1) popleft, not O(n) pop(0)."""
        lm = DummyLM([{"response": str(i)} for i in range(5)])
        dspy.configure(lm=lm, max_history_size=3)
        predictor = dspy.Predict("q: str -> response: str")

        for i in range(5):
            predictor(q=f"query_{i}")

        assert len(lm.history) <= 3

    def test_lm_copy_has_empty_deque_history(self):
        lm = DummyLM([{"response": "a"}])
        dspy.configure(lm=lm)
        predictor = dspy.Predict("q: str -> response: str")
        predictor(q="test")

        assert len(lm.history) > 0

        lm_copy = lm.copy()
        assert isinstance(lm_copy.history, deque)
        assert len(lm_copy.history) == 0


class TestModuleHistoryDeque:
    def test_module_history_is_deque(self):
        module = dspy.Predict("q -> a")
        assert isinstance(module.history, deque)

    def test_module_history_after_deserialization(self):
        """Module.__setstate__ should restore history as deque."""
        module = dspy.Predict("q -> a")
        state = module.__getstate__()
        assert "history" not in state

        module.__setstate__(state)
        assert isinstance(module.history, deque)
