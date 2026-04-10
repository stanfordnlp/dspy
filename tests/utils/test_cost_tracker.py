"""Comprehensive tests for the CostTracker and cost tracking utilities."""

import threading
from unittest import mock

import pytest

import dspy
from dspy.utils.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    _lookup_litellm_pricing,
    register_model_pricing,
    track_cost,
)


# ---------------------------------------------------------------------------
# CostTracker initialization tests
# ---------------------------------------------------------------------------


class TestCostTrackerInit:
    def test_default_init(self):
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.budget is None
        assert tracker.budget_action == "warn"
        assert tracker.num_calls == 0

    def test_init_with_budget(self):
        tracker = CostTracker(budget=5.0)
        assert tracker.budget == 5.0
        assert tracker.budget_remaining == 5.0
        assert tracker.budget_utilization == 0.0

    def test_init_with_budget_stop(self):
        tracker = CostTracker(budget=10.0, budget_action="stop")
        assert tracker.budget_action == "stop"

    def test_invalid_budget_action(self):
        with pytest.raises(ValueError, match="budget_action must be"):
            CostTracker(budget_action="invalid")

    def test_negative_budget(self):
        with pytest.raises(ValueError, match="budget must be positive"):
            CostTracker(budget=-1.0)

    def test_zero_budget(self):
        with pytest.raises(ValueError, match="budget must be positive"):
            CostTracker(budget=0.0)


# ---------------------------------------------------------------------------
# add_usage and cost calculation tests
# ---------------------------------------------------------------------------


class TestAddUsage:
    def test_add_single_usage(self):
        tracker = CostTracker()
        tracker.set_model_pricing("test-model", 0.001, 0.002)
        tracker.add_usage("test-model", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert tracker.num_calls == 1
        # cost = 100 * 0.001 + 50 * 0.002 = 0.1 + 0.1 = 0.2
        assert abs(tracker.total_cost - 0.2) < 1e-9

    def test_add_multiple_usages_same_model(self):
        tracker = CostTracker()
        tracker.set_model_pricing("test-model", 0.001, 0.002)

        tracker.add_usage("test-model", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        tracker.add_usage("test-model", {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
        })

        assert tracker.num_calls == 2
        # cost = (100*0.001 + 50*0.002) + (200*0.001 + 100*0.002) = 0.2 + 0.4 = 0.6
        assert abs(tracker.total_cost - 0.6) < 1e-9

    def test_add_usages_multiple_models(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)
        tracker.set_model_pricing("model-b", 0.01, 0.02)

        tracker.add_usage("model-a", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        tracker.add_usage("model-b", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })

        costs = tracker.get_costs_by_model()
        assert abs(costs["model-a"] - 0.2) < 1e-9
        assert abs(costs["model-b"] - 2.0) < 1e-9
        assert abs(tracker.total_cost - 2.2) < 1e-9

    def test_add_empty_usage(self):
        tracker = CostTracker()
        tracker.add_usage("test-model", {})
        assert tracker.num_calls == 0
        assert tracker.total_cost == 0.0

    def test_add_usage_with_none_tokens(self):
        tracker = CostTracker()
        tracker.set_model_pricing("test-model", 0.001, 0.002)
        tracker.add_usage("test-model", {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        })
        assert tracker.num_calls == 1
        assert tracker.total_cost == 0.0

    def test_add_usage_unknown_model_no_pricing(self):
        tracker = CostTracker()
        tracker.add_usage("unknown/model-xyz", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        # Should still track tokens, but cost is None
        assert tracker.num_calls == 1
        assert tracker.total_cost == 0.0
        tokens = tracker.get_tokens_by_model()
        assert tokens["unknown/model-xyz"]["prompt_tokens"] == 100


# ---------------------------------------------------------------------------
# Token tracking tests
# ---------------------------------------------------------------------------


class TestTokenTracking:
    def test_get_tokens_by_model(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.add_usage("model-a", {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})

        tokens = tracker.get_tokens_by_model()
        assert tokens["model-a"]["prompt_tokens"] == 300
        assert tokens["model-a"]["completion_tokens"] == 150
        assert tokens["model-a"]["total_tokens"] == 450

    def test_get_total_tokens_compatibility(self):
        """Test that get_total_tokens works for UsageTracker compatibility."""
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

        total = tracker.get_total_tokens()
        assert total["model-a"]["prompt_tokens"] == 100
        assert total["model-a"]["completion_tokens"] == 50
        assert total["model-a"]["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Call log tests
# ---------------------------------------------------------------------------


class TestCallLog:
    def test_call_log_entries(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

        log = tracker.get_call_log()
        assert len(log) == 1
        assert log[0]["model"] == "model-a"
        assert log[0]["prompt_tokens"] == 100
        assert log[0]["completion_tokens"] == 50
        assert log[0]["cost"] is not None
        assert abs(log[0]["cost"] - 0.2) < 1e-9

    def test_call_log_multiple_entries(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)
        tracker.set_model_pricing("model-b", 0.01, 0.02)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        tracker.add_usage("model-b", {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})

        log = tracker.get_call_log()
        assert len(log) == 2
        assert log[0]["model"] == "model-a"
        assert log[1]["model"] == "model-b"


# ---------------------------------------------------------------------------
# Budget enforcement tests
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_budget_exceeded_warn(self):
        tracker = CostTracker(budget=0.1, budget_action="warn")
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        with pytest.warns(UserWarning, match="Budget of \\$0.1000 exceeded"):
            tracker.add_usage("model-a", {
                "prompt_tokens": 500,
                "completion_tokens": 500,
                "total_tokens": 1000,
            })

    def test_budget_exceeded_stop(self):
        tracker = CostTracker(budget=0.1, budget_action="stop")
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        with pytest.raises(BudgetExceededError, match="Budget of \\$0.1000 exceeded"):
            tracker.add_usage("model-a", {
                "prompt_tokens": 500,
                "completion_tokens": 500,
                "total_tokens": 1000,
            })

    def test_budget_not_exceeded(self):
        tracker = CostTracker(budget=10.0, budget_action="stop")
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        # Should not raise
        tracker.add_usage("model-a", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert tracker.total_cost < tracker.budget

    def test_budget_warning_only_once(self):
        tracker = CostTracker(budget=0.05, budget_action="warn")
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        with pytest.warns(UserWarning, match="Budget"):
            tracker.add_usage("model-a", {
                "prompt_tokens": 500,
                "completion_tokens": 500,
                "total_tokens": 1000,
            })

        # Second call should NOT warn again
        # (no pytest.warns context - if it warned, the test would still pass,
        # but we verify the flag)
        tracker.add_usage("model-a", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert tracker._budget_warned is True

    def test_budget_remaining(self):
        tracker = CostTracker(budget=1.0)
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        # cost = 100*0.001 + 50*0.002 = 0.2
        assert abs(tracker.budget_remaining - 0.8) < 1e-9

    def test_budget_utilization(self):
        tracker = CostTracker(budget=1.0)
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        assert abs(tracker.budget_utilization - 0.2) < 1e-9

    def test_no_budget_remaining_is_none(self):
        tracker = CostTracker()
        assert tracker.budget_remaining is None
        assert tracker.budget_utilization is None

    def test_budget_exceeded_error_attributes(self):
        err = BudgetExceededError(budget=1.0, total_cost=1.5, model="gpt-4o")
        assert err.budget == 1.0
        assert err.total_cost == 1.5
        assert err.model == "gpt-4o"
        assert "gpt-4o" in str(err)


# ---------------------------------------------------------------------------
# Custom pricing tests
# ---------------------------------------------------------------------------


class TestCustomPricing:
    def test_set_model_pricing_override(self):
        tracker = CostTracker()
        # Even if litellm has pricing for gpt-4o-mini, custom pricing should take precedence
        tracker.set_model_pricing("gpt-4o-mini", 0.005, 0.01)

        tracker.add_usage("gpt-4o-mini", {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        # cost = 100 * 0.005 + 50 * 0.01 = 0.5 + 0.5 = 1.0
        assert abs(tracker.total_cost - 1.0) < 1e-9

    def test_register_global_pricing(self):
        register_model_pricing("custom/my-model", 0.0001, 0.0002)

        tracker = CostTracker()
        tracker.add_usage("custom/my-model", {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        })
        # cost = 1000 * 0.0001 + 500 * 0.0002 = 0.1 + 0.1 = 0.2
        assert abs(tracker.total_cost - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Litellm pricing lookup tests
# ---------------------------------------------------------------------------


class TestLitellmPricing:
    def test_lookup_known_model(self):
        pricing = _lookup_litellm_pricing("gpt-4o-mini")
        if pricing is not None:
            assert "input_cost_per_token" in pricing
            assert "output_cost_per_token" in pricing
            assert pricing["input_cost_per_token"] > 0
            assert pricing["output_cost_per_token"] > 0

    def test_lookup_with_provider_prefix(self):
        pricing = _lookup_litellm_pricing("openai/gpt-4o-mini")
        # Should find it by stripping the prefix
        if pricing is not None:
            assert pricing["input_cost_per_token"] > 0

    def test_lookup_unknown_model(self):
        pricing = _lookup_litellm_pricing("totally-unknown-model-xyz-999")
        assert pricing is None


# ---------------------------------------------------------------------------
# Summary and reporting tests
# ---------------------------------------------------------------------------


class TestSummaryReporting:
    def test_get_summary_no_budget(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

        summary = tracker.get_summary()
        assert "total_cost" in summary
        assert "num_calls" in summary
        assert "costs_by_model" in summary
        assert "tokens_by_model" in summary
        assert "budget" not in summary
        assert summary["num_calls"] == 1

    def test_get_summary_with_budget(self):
        tracker = CostTracker(budget=5.0)
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

        summary = tracker.get_summary()
        assert summary["budget"] == 5.0
        assert summary["budget_remaining"] > 0
        assert summary["budget_utilization"] > 0

    def test_repr(self):
        tracker = CostTracker(budget=5.0)
        tracker.set_model_pricing("model-a", 0.001, 0.002)
        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})

        repr_str = repr(tracker)
        assert "CostTracker" in repr_str
        assert "$" in repr_str
        assert "budget" in repr_str


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_data(self):
        tracker = CostTracker(budget=5.0)
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        tracker.add_usage("model-a", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
        assert tracker.total_cost > 0
        assert tracker.num_calls > 0

        tracker.reset()
        assert tracker.total_cost == 0.0
        assert tracker.num_calls == 0
        assert len(tracker.get_costs_by_model()) == 0
        assert len(tracker.get_tokens_by_model()) == 0
        assert len(tracker.get_call_log()) == 0


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_usage(self):
        tracker = CostTracker()
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        num_threads = 10
        calls_per_thread = 100

        def add_calls():
            for _ in range(calls_per_thread):
                tracker.add_usage("model-a", {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                })

        threads = [threading.Thread(target=add_calls) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.num_calls == num_threads * calls_per_thread
        expected_cost = num_threads * calls_per_thread * (10 * 0.001 + 5 * 0.002)
        assert abs(tracker.total_cost - expected_cost) < 1e-6


# ---------------------------------------------------------------------------
# track_cost context manager tests
# ---------------------------------------------------------------------------


class TestTrackCostContextManager:
    def test_track_cost_basic(self):
        with track_cost() as tracker:
            assert isinstance(tracker, CostTracker)
            # Verify the tracker is set as the usage_tracker in settings
            from dspy.dsp.utils.settings import settings
            assert settings.usage_tracker is tracker
            assert settings.track_usage is True

    def test_track_cost_with_budget(self):
        with track_cost(budget=10.0, budget_action="stop") as tracker:
            assert tracker.budget == 10.0
            assert tracker.budget_action == "stop"

    def test_track_cost_context_restores_settings(self):
        from dspy.dsp.utils.settings import settings

        original_tracker = settings.get("usage_tracker")

        with track_cost() as tracker:
            assert settings.usage_tracker is tracker

        # After context, the setting should be restored
        assert settings.get("usage_tracker") is not tracker

    def test_track_cost_with_manual_usage(self):
        with track_cost() as tracker:
            tracker.set_model_pricing("test-model", 0.001, 0.002)
            tracker.add_usage("test-model", {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })

        assert tracker.total_cost > 0
        assert tracker.num_calls == 1

    def test_track_cost_invalid_budget_action(self):
        with pytest.raises(ValueError, match="budget_action must be"):
            with track_cost(budget_action="explode") as tracker:
                pass


# ---------------------------------------------------------------------------
# Integration-style tests with mocked LM
# ---------------------------------------------------------------------------


class TestIntegrationWithMockedLM:
    def test_cost_tracked_via_settings(self):
        """Test that CostTracker receives usage data through the settings mechanism."""
        tracker = CostTracker()
        tracker.set_model_pricing("openai/gpt-4o-mini", 0.00015, 0.0006)

        from dspy.dsp.utils.settings import settings

        with settings.context(usage_tracker=tracker, track_usage=True):
            # Simulate what LM.forward does after a completion
            tracker.add_usage("openai/gpt-4o-mini", {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "total_tokens": 600,
            })
            tracker.add_usage("openai/gpt-4o-mini", {
                "prompt_tokens": 300,
                "completion_tokens": 200,
                "total_tokens": 500,
            })

        assert tracker.num_calls == 2
        # cost = (500*0.00015 + 100*0.0006) + (300*0.00015 + 200*0.0006)
        #      = (0.075 + 0.06) + (0.045 + 0.12) = 0.135 + 0.165 = 0.3
        assert abs(tracker.total_cost - 0.3) < 1e-9

        costs = tracker.get_costs_by_model()
        assert "openai/gpt-4o-mini" in costs

        tokens = tracker.get_tokens_by_model()
        assert tokens["openai/gpt-4o-mini"]["prompt_tokens"] == 800
        assert tokens["openai/gpt-4o-mini"]["completion_tokens"] == 300

    def test_budget_stop_raises_during_execution(self):
        """Test that budget_action='stop' actually halts execution."""
        tracker = CostTracker(budget=0.01, budget_action="stop")
        tracker.set_model_pricing("model-a", 0.001, 0.002)

        with pytest.raises(BudgetExceededError):
            for _ in range(100):
                tracker.add_usage("model-a", {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                })

        # Should have stopped before completing all 100 calls
        assert tracker.num_calls < 100
