"""Tests for cost tracking utilities."""

from unittest import mock

import pytest

import dspy
from dspy.utils.cost_tracker import BudgetExceededError, CostTracker, track_cost


def test_add_cost_entry():
    """Test adding cost entries to the tracker."""
    tracker = CostTracker()

    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4o-mini", 0.00012)
    tracker.add_cost("openai/gpt-4", 0.0025)

    assert len(tracker.cost_data["openai/gpt-4o-mini"]) == 2
    assert len(tracker.cost_data["openai/gpt-4"]) == 1
    assert tracker.cost_data["openai/gpt-4o-mini"][0] == 0.00015
    assert tracker.cost_data["openai/gpt-4o-mini"][1] == 0.00012


def test_add_cost_with_none():
    """Test that None costs are ignored."""
    tracker = CostTracker()

    tracker.add_cost("openai/gpt-4o-mini", None)
    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4o-mini", 0)  # Zero cost should also be ignored

    assert len(tracker.cost_data["openai/gpt-4o-mini"]) == 1


def test_total_cost():
    """Test calculating total cost."""
    tracker = CostTracker()

    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4o-mini", 0.00012)
    tracker.add_cost("openai/gpt-4", 0.0025)

    # 0.00015 + 0.00012 + 0.0025 = 0.00277
    assert abs(tracker.total_cost - 0.00277) < 1e-10


def test_get_costs_by_model():
    """Test getting costs broken down by model."""
    tracker = CostTracker()

    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4o-mini", 0.00012)
    tracker.add_cost("openai/gpt-4", 0.0025)

    costs = tracker.get_costs_by_model()

    assert "openai/gpt-4o-mini" in costs
    assert "openai/gpt-4" in costs
    assert abs(costs["openai/gpt-4o-mini"] - 0.00027) < 1e-10
    assert abs(costs["openai/gpt-4"] - 0.0025) < 1e-10


def test_add_cost_with_usage():
    """Test adding cost with usage data."""
    tracker = CostTracker()

    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    tracker.add_cost("openai/gpt-4o-mini", 0.00015, usage=usage)

    assert len(tracker.usage_data["openai/gpt-4o-mini"]) == 1
    assert tracker.usage_data["openai/gpt-4o-mini"][0] == usage


def test_call_counts():
    """Test that call counts are tracked correctly."""
    tracker = CostTracker()

    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4o-mini", None)  # None cost still counts as a call
    tracker.add_cost("openai/gpt-4", 0.0025)

    assert tracker.call_counts["openai/gpt-4o-mini"] == 2
    assert tracker.call_counts["openai/gpt-4"] == 1


def test_get_cost_summary():
    """Test getting a comprehensive cost summary."""
    tracker = CostTracker()

    tracker.add_cost(
        "openai/gpt-4o-mini",
        0.00015,
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    tracker.add_cost(
        "openai/gpt-4o-mini",
        0.00012,
        usage={"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
    )
    tracker.add_cost(
        "openai/gpt-4",
        0.0025,
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
    )

    summary = tracker.get_cost_summary()

    assert abs(summary["total_cost"] - 0.00277) < 1e-10
    assert "openai/gpt-4o-mini" in summary["costs_by_model"]
    assert "openai/gpt-4" in summary["costs_by_model"]
    assert summary["calls_by_model"]["openai/gpt-4o-mini"] == 2
    assert summary["calls_by_model"]["openai/gpt-4"] == 1
    assert "openai/gpt-4o-mini" in summary["average_cost_per_call"]
    assert summary["usage_by_model"]["openai/gpt-4o-mini"]["prompt_tokens"] == 180
    assert summary["usage_by_model"]["openai/gpt-4o-mini"]["completion_tokens"] == 90


def test_budget_exceeded_total():
    """Test that BudgetExceededError is raised when total budget is exceeded."""
    tracker = CostTracker(budget=0.001)

    tracker.add_cost("openai/gpt-4o-mini", 0.0005)

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.add_cost("openai/gpt-4o-mini", 0.0006)  # This exceeds the $0.001 budget

    assert exc_info.value.budget == 0.001
    assert exc_info.value.current_cost > 0.001
    assert exc_info.value.model is None


def test_budget_exceeded_per_model():
    """Test that BudgetExceededError is raised when per-model budget is exceeded."""
    tracker = CostTracker(budget_per_model={"openai/gpt-4": 0.001})

    tracker.add_cost("openai/gpt-4o-mini", 0.01)  # Different model, no limit
    tracker.add_cost("openai/gpt-4", 0.0005)

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.add_cost("openai/gpt-4", 0.0006)  # This exceeds the $0.001 budget for gpt-4

    assert exc_info.value.budget == 0.001
    assert exc_info.value.model == "openai/gpt-4"


def test_track_cost_context_manager():
    """Test the track_cost context manager."""
    with track_cost() as tracker:
        assert isinstance(tracker, CostTracker)
        assert dspy.settings.cost_tracker is tracker

    # After exiting, cost_tracker should be None
    assert dspy.settings.cost_tracker is None


def test_track_cost_context_manager_with_budget():
    """Test track_cost with budget parameter."""
    with track_cost(budget=1.0) as tracker:
        assert tracker.budget == 1.0


def test_track_cost_context_manager_with_per_model_budget():
    """Test track_cost with per-model budget."""
    with track_cost(budget_per_model={"openai/gpt-4": 0.5}) as tracker:
        assert tracker.budget_per_model["openai/gpt-4"] == 0.5


def test_repr():
    """Test the string representation of CostTracker."""
    tracker = CostTracker()
    tracker.add_cost("openai/gpt-4o-mini", 0.00015)
    tracker.add_cost("openai/gpt-4", 0.0025)

    repr_str = repr(tracker)

    assert "CostTracker" in repr_str
    assert "total=$0.0027" in repr_str  # Approximately
    assert "gpt-4o-mini" in repr_str
    assert "gpt-4" in repr_str


def test_budget_exceeded_error_message():
    """Test the error message format for BudgetExceededError."""
    error = BudgetExceededError(0.015, 0.01, model="openai/gpt-4")
    assert "gpt-4" in str(error)
    assert "0.015" in str(error)
    assert "0.01" in str(error)

    error_no_model = BudgetExceededError(0.015, 0.01)
    assert "gpt-4" not in str(error_no_model)


def test_track_cost_with_mocked_lm():
    """Test cost tracking with a mocked LM that returns cost data."""
    # Create a mock response with cost data
    mock_response = mock.MagicMock()
    mock_response.cache_hit = False
    mock_response._hidden_params = {"response_cost": 0.00025}
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4o-mini"

    # Mock LM
    mock_lm = mock.MagicMock(spec=dspy.LM)
    mock_lm.return_value = ["Test response"]
    mock_lm.kwargs = {}
    mock_lm.model = "openai/gpt-4o-mini"

    dspy.configure(lm=mock_lm, adapter=dspy.ChatAdapter())

    with track_cost() as tracker:
        # Manually add a cost to simulate what the LM would do
        tracker.add_cost(
            "openai/gpt-4o-mini",
            0.00025,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

    assert tracker.total_cost == 0.00025
    assert tracker.call_counts["openai/gpt-4o-mini"] == 1


def test_empty_tracker():
    """Test that an empty tracker has zero cost."""
    tracker = CostTracker()

    assert tracker.total_cost == 0
    assert tracker.get_costs_by_model() == {}
    assert tracker.call_counts == {}


def test_multiple_contexts_independent():
    """Test that multiple track_cost contexts are independent."""
    with track_cost() as tracker1:
        tracker1.add_cost("model-a", 0.001)

        with track_cost() as tracker2:
            tracker2.add_cost("model-b", 0.002)
            assert tracker2.total_cost == 0.002
            assert "model-a" not in tracker2.cost_data

        # tracker1 should still have its own data
        assert tracker1.total_cost == 0.001
        assert "model-b" not in tracker1.cost_data
