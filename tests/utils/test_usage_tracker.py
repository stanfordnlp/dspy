import pytest
import dspy
from unittest.mock import Mock, patch, MagicMock
from dspy.utils.usage_tracker import UsageTracker, track_usage
from dspy.utils.dummies import DummyLM
import os


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


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Skip the test if OPENAI_API_KEY is not set.",
)
def test_track_usage_context_manager():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.settings.configure(lm=lm)

    predict = dspy.ChainOfThought("question -> answer")
    with track_usage() as tracker:
        predict(question="What is the capital of France?")
        predict(question="What is the capital of Italy?")

    assert len(tracker.usage_data) > 0
    assert len(tracker.usage_data["openai/gpt-4o-mini"]) == 2

    total_usage = tracker.get_total_tokens()
    assert "openai/gpt-4o-mini" in total_usage
    assert len(total_usage.keys()) == 1
    assert isinstance(total_usage["openai/gpt-4o-mini"], dict)
