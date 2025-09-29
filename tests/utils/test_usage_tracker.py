from litellm.types.llms.openai import InputTokensDetails, OutputTokensDetails

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
    lm = dspy.LM(lm_for_test, cache=False)
    dspy.settings.configure(lm=lm)

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

    assert total_usage["gpt-4o-mini"]["prompt_tokens"] == 2717
    assert total_usage["gpt-4o-mini"]["completion_tokens"] == 246
    assert total_usage["gpt-4o-mini"]["total_tokens"] == 2963
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["cached_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["prompt_tokens_details"]["audio_tokens"] == 50
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["audio_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["accepted_prediction_tokens"] == 1
    assert total_usage["gpt-4o-mini"]["completion_tokens_details"]["rejected_prediction_tokens"] == 1


def test_merge_input_output_token_details_aggregation():
    """Ensure input/output token details are aggregated across entries."""
    tracker = UsageTracker()

    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": InputTokensDetails(
                    audio_tokens=0,
                    cached_tokens=200,
                    text_tokens=1000,
                ),
                "output_tokens": 300,
                "output_tokens_details": OutputTokensDetails(
                    reasoning_tokens=100,
                    text_tokens=200,
                ),
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "input_tokens": 500,
                "input_tokens_details": InputTokensDetails(
                    audio_tokens=10,
                    cached_tokens=50,
                    text_tokens=400,
                ),
                "output_tokens": 100,
                "output_tokens_details": OutputTokensDetails(
                    reasoning_tokens=20,
                    text_tokens=80,
                ),
            },
        },
        {
            # Missing input details entirely; only output parts present
            "model": "gpt-4o-mini",
            "usage": {
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        },
        {
            # Input tokens without details to ensure numeric-only fields still sum
            "model": "gpt-4o-mini",
            "usage": {
                "input_tokens": 200,
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage(entry["model"], entry["usage"])

    total_usage = tracker.get_total_tokens()["gpt-4o-mini"]

    # Totals for numeric input/output fields
    assert total_usage["input_tokens"] == 1700  # 1000 + 500 + 200
    assert total_usage["output_tokens"] == 401  # 300 + 100 + 1

    # Aggregated details
    assert total_usage["input_tokens_details"]["audio_tokens"] == 10  # 0 + 10
    assert total_usage["input_tokens_details"]["cached_tokens"] == 250  # 200 + 50
    assert total_usage["input_tokens_details"]["text_tokens"] == 1400  # 1000 + 400

    assert total_usage["output_tokens_details"]["reasoning_tokens"] == 121  # 100 + 20 + 1
    assert total_usage["output_tokens_details"]["text_tokens"] == 280  # 200 + 80


def test_merge_input_output_token_details_with_none_values():
    """None values for details should be ignored while preserving later dict merges."""
    tracker = UsageTracker()

    usage_entries = [
        {
            "model": "gpt-4o-mini",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
            },
        },
        {
            "model": "gpt-4o-mini",
            "usage": {
                "input_tokens": 50,
                "input_tokens_details": InputTokensDetails(
                    audio_tokens=1,
                    cached_tokens=2,
                    text_tokens=3,
                ),
                "output_tokens": 5,
                "output_tokens_details": OutputTokensDetails(
                    reasoning_tokens=4,
                    text_tokens=6,
                ),
            },
        },
    ]

    for entry in usage_entries:
        tracker.add_usage(entry["model"], entry["usage"])

    total_usage = tracker.get_total_tokens()["gpt-4o-mini"]

    assert total_usage["input_tokens"] == 150
    assert total_usage["output_tokens"] == 15

    assert total_usage["input_tokens_details"]["audio_tokens"] == 1
    assert total_usage["input_tokens_details"]["cached_tokens"] == 2
    assert total_usage["input_tokens_details"]["text_tokens"] == 3

    assert total_usage["output_tokens_details"]["reasoning_tokens"] == 4
    assert total_usage["output_tokens_details"]["text_tokens"] == 6
