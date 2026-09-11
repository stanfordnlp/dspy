import copy
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from pydantic import BaseModel

import dspy
from dspy.utils.usage_tracker import UsageTracker, track_usage


def test_add_usage_entry():
    """Test adding usage entries to the tracker."""
    tracker = UsageTracker()

    # Test with a single usage entry
    usage_entry = {
        "prompt_tokens": 1117,
        "completion_tokens": 46,
        "total_tokens": 1163,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }

    tracker.add_usage("gpt-4o-mini", usage_entry)
    assert len(tracker.usage_data["gpt-4o-mini"]) == 1
    assert tracker.usage_data["gpt-4o-mini"][0] == usage_entry


def test_add_usage_with_cost():
    """Test accumulating response costs across calls and models."""
    tracker = UsageTracker()

    tracker.add_usage("gpt-4o-mini", {"prompt_tokens": 10, "completion_tokens": 5}, cost=0.001)
    tracker.add_usage("gpt-4o-mini", {"prompt_tokens": 20, "completion_tokens": 10}, cost=0.002)
    tracker.add_usage("gpt-3.5-turbo", {"prompt_tokens": 5, "completion_tokens": 5}, cost=0.0005)

    assert tracker.total_cost_by_model["gpt-4o-mini"] == pytest.approx(0.003)
    assert tracker.total_cost_by_model["gpt-3.5-turbo"] == pytest.approx(0.0005)
    assert tracker.get_total_cost() == pytest.approx(0.0035)


def test_add_usage_with_none_cost():
    """None costs are not accumulated, and the total defaults to 0.0."""
    tracker = UsageTracker()

    assert tracker.get_total_cost() == 0.0

    tracker.add_usage("gpt-4o-mini", {"prompt_tokens": 10, "completion_tokens": 5})
    tracker.add_usage("gpt-4o-mini", {"prompt_tokens": 20, "completion_tokens": 10}, cost=None)

    assert len(tracker.usage_data["gpt-4o-mini"]) == 2
    assert len(tracker.total_cost_by_model) == 0
    assert tracker.get_total_cost() == 0.0


def test_add_usage_cost_without_usage_entry():
    """Cost is tracked even when the provider reports no token usage."""
    tracker = UsageTracker()

    tracker.add_usage("gpt-4o-mini", {}, cost=0.01)

    assert len(tracker.usage_data) == 0
    assert tracker.get_total_cost() == pytest.approx(0.01)


def test_usage_and_cost_updates_are_thread_safe():
    tracker = UsageTracker()
    update_count = 1_000

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(
            executor.map(
                lambda _: tracker.add_usage("model", {"total_tokens": 1}, cost=0.01),
                range(update_count),
            )
        )

    assert tracker.get_total_tokens()["model"]["total_tokens"] == update_count
    assert tracker.get_total_cost() == pytest.approx(update_count * 0.01)


def test_usage_tracker_can_be_deep_copied():
    tracker = UsageTracker()
    tracker.add_usage("model", {"total_tokens": 1}, cost=0.01)

    copied = copy.deepcopy(tracker)
    copied.add_usage("model", {"total_tokens": 2}, cost=0.02)

    assert tracker.get_total_tokens()["model"]["total_tokens"] == 1
    assert tracker.get_total_cost() == pytest.approx(0.01)
    assert copied.get_total_tokens()["model"]["total_tokens"] == 3
    assert copied.get_total_cost() == pytest.approx(0.03)


def test_get_total_tokens():
    """Test calculating total tokens from usage entries."""
    tracker = UsageTracker()

    # Add multiple usage entries for the same model
    usage_entries = [
        {
            "prompt_tokens": 1117,
            "completion_tokens": 46,
            "total_tokens": 1163,
            "prompt_tokens_details": {"cached_tokens": 200, "audio_tokens": 50},
            "completion_tokens_details": {
                "reasoning_tokens": 20,
                "audio_tokens": 10,
                "accepted_prediction_tokens": 16,
                "rejected_prediction_tokens": 0,
            },
        },
        {
            "prompt_tokens": 800,
            "completion_tokens": 100,
            "total_tokens": 900,
            "prompt_tokens_details": {"cached_tokens": 300, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 50,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 40,
                "rejected_prediction_tokens": 10,
            },
        },
        {
            "prompt_tokens": 500,
            "completion_tokens": 80,
            "total_tokens": 580,
            "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 25},
            "completion_tokens_details": {
                "reasoning_tokens": 30,
                "audio_tokens": 15,
                "accepted_prediction_tokens": 25,
                "rejected_prediction_tokens": 10,
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage("gpt-4o-mini", entry)

    total_usage = tracker.get_total_tokens()
    assert "gpt-4o-mini" in total_usage
    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2417  # 1117 + 800 + 500
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 226  # 46 + 100 + 80
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 2643  # 1163 + 900 + 580
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 600  # 200 + 300 + 100
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["audio_tokens"] == 75  # 50 + 0 + 25
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 100  # 20 + 50 + 30
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 25  # 10 + 0 + 15
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 81  # 16 + 40 + 25
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 20  # 0 + 10 + 10


def test_track_usage_with_multiple_models():
    """Test tracking usage across multiple models."""
    tracker = UsageTracker()

    # Add usage entries for different models
    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        },
        {
            "model": "gpt-3.5-turbo",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage(entry["model"], entry["usage"])

    total_usage = tracker.get_total_tokens()
    assert "gpt-4o-mini" in total_usage
    assert "gpt-3.5-turbo" in total_usage
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 1163
    assert total_usage["gpt-3.5-turbo"]["total_tokens"] == 900


def test_track_usage_context_manager(lm_for_test):
    lm = dspy.LM(lm_for_test, cache=False, temperature=0.0)
    dspy.configure(lm=lm)

    predict = dspy.ChainOfThought("question -> answer")
    with track_usage() as tracker:
        predict(question="What is the capital of France?")
        predict(question="What is the capital of Italy?")

    assert len(tracker.usage_data) > 0
    assert len(tracker.usage_data[lm_for_test]) == 2

    total_usage = tracker.get_total_tokens()
    assert lm_for_test in total_usage
    assert len(total_usage.keys()) == 1
    assert isinstance(total_usage[lm_for_test], dict)


def test_merge_usage_entries_with_new_keys():
    """Ensure merging usage entries preserves unseen keys."""
    tracker = UsageTracker()

    tracker.add_usage("model-x", {"prompt_tokens": 5})
    tracker.add_usage("model-x", {"completion_tokens": 2})

    total_usage = tracker.get_total_tokens()

    assert total_usage["model-x"]["prompt_tokens"] == 5
    assert total_usage["model-x"]["completion_tokens"] == 2


def test_merge_usage_entries_with_none_values():
    """Test tracking usage across multiple models."""
    tracker = UsageTracker()

    # Add usage entries for different models
    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": None,
                "completion_tokens_details": {},
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 100,
                "total_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 50, "audio_tokens": 50},
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": None,
                "completion_tokens_details": {
                    "reasoning_tokens": 1,
                    "audio_tokens": 1,
                    "accepted_prediction_tokens": 1,
                    "rejected_prediction_tokens": 1,
                },
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage(entry["model"], entry["usage"])

    total_usage = tracker.get_total_tokens()

    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2817
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 346
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 3163
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["audio_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 1


def test_merge_usage_entries_with_pydantic_models():
    """Test merging usage entries with Pydantic model objects, like `PromptTokensDetailsWrapper` from litellm."""
    tracker = UsageTracker()

    # Here we define a simplified version of the Pydantic models from litellm to avoid the dependency change on litellm.
    class CacheCreationTokenDetails(BaseModel):
        ephemeral_5m_input_tokens: int
        ephemeral_1h_input_tokens: int

    class PromptTokensDetailsWrapper(BaseModel):
        audio_tokens: int | None
        cached_tokens: int
        text_tokens: int | None
        image_tokens: int | None
        cache_creation_tokens: int
        cache_creation_token_details: CacheCreationTokenDetails

    # Add usage entries for different models
    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": {},
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": None,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "total_tokens": 900,
                "prompt_tokens_details": PromptTokensDetailsWrapper(
                    audio_tokens=None,
                    cached_tokens=3,
                    text_tokens=None,
                    image_tokens=None,
                    cache_creation_tokens=0,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_5m_input_tokens=5, ephemeral_1h_input_tokens=0
                    ),
                ),
                "completion_tokens_details": {
                    "reasoning_tokens": 1,
                    "audio_tokens": 1,
                    "accepted_prediction_tokens": 1,
                    "rejected_prediction_tokens": 1,
                },
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage(entry["model"], entry["usage"])

    total_usage = tracker.get_total_tokens()

    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2717
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 246
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 2963
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 9
    assert (
        total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cache_creation_token_details"]["ephemeral_5m_input_tokens"]
        == 15
    )
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 1


def test_parallel_executor_with_usage_tracker():
    """Test that usage tracking works correctly with ParallelExecutor and mocked LM calls."""

    parent_tracker = UsageTracker()

    # Mock LM with different responses
    mock_lm = mock.MagicMock(spec=dspy.LM)
    mock_lm.return_value = ['{"answer": "Mocked answer"}']
    mock_lm.kwargs = {}
    mock_lm.model = "openai/gpt-4o-mini"

    dspy.configure(lm=mock_lm, adapter=dspy.JSONAdapter())

    def task1():
        # Simulate LM usage tracking for task 1
        dspy.settings.usage_tracker.add_usage(
            "openai/gpt-4o-mini",
            {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
        )
        return dspy.settings.usage_tracker.get_total_tokens()

    def task2():
        # Simulate LM usage tracking for task 2 with different values
        dspy.settings.usage_tracker.add_usage(
            "openai/gpt-4o-mini",
            {
                "prompt_tokens": 80,
                "completion_tokens": 15,
                "total_tokens": 95,
            },
        )
        return dspy.settings.usage_tracker.get_total_tokens()

    # Execute tasks in parallel
    with dspy.context(track_usage=True, usage_tracker=parent_tracker):
        executor = dspy.Parallel()
        results = executor([(task1, {}), (task2, {})])
    # Verify that the two workers had different usage
    usage1 = results[0]
    usage2 = results[1]

    # Task 1 should have 50 prompt tokens, task 2 should have 80
    assert usage1["openai/gpt-4o-mini"]["prompt_tokens"] == 50
    assert usage1["openai/gpt-4o-mini"]["completion_tokens"] == 10
    assert usage2["openai/gpt-4o-mini"]["prompt_tokens"] == 80
    assert usage2["openai/gpt-4o-mini"]["completion_tokens"] == 15

    # Parent tracker should remain unchanged (workers have independent copies)
    assert len(parent_tracker.usage_data) == 0
