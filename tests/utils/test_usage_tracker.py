from unittest import mock

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


def test_merge_usage_entries_with_mixed_types():
    """Test merging usage entries where nested keys have mixed types (dict, int, None).

    This test verifies the fix for the bug where _merge_usage_entries would fail
    with TypeError when trying to call len() on a non-dict value during recursive merging.
    When types are mixed, the new value (right value) should override the old value.
    """
    # Case 1: None first, then dict - new value (dict) should override
    tracker1 = UsageTracker()
    tracker1.add_usage("test-1", {"prompt_tokens": 100, "details": None})
    tracker1.add_usage("test-1", {"prompt_tokens": 200, "details": {"cached_tokens": 10}})
    total1 = tracker1.get_total_tokens()
    assert total1["test-1"]["prompt_tokens"] == 300
    assert isinstance(total1["test-1"]["details"], dict)
    assert total1["test-1"]["details"]["cached_tokens"] == 10

    # Case 2: Dict first, then None - old value (dict) should be preserved (None is meaningless)
    tracker2 = UsageTracker()
    tracker2.add_usage("test-2", {"prompt_tokens": 100, "details": {"cached_tokens": 10}})
    tracker2.add_usage("test-2", {"prompt_tokens": 200, "details": None})
    total2 = tracker2.get_total_tokens()
    assert total2["test-2"]["prompt_tokens"] == 300
    assert isinstance(total2["test-2"]["details"], dict)
    assert total2["test-2"]["details"]["cached_tokens"] == 10

    # Case 3: Dict first, then int - new value (int) should override
    tracker3 = UsageTracker()
    tracker3.add_usage("test-3", {"prompt_tokens": 100, "details": {"cached_tokens": 10}})
    tracker3.add_usage("test-3", {"prompt_tokens": 200, "details": 15})
    total3 = tracker3.get_total_tokens()
    assert total3["test-3"]["prompt_tokens"] == 300
    assert total3["test-3"]["details"] == 15

    # Case 4: Int first, then dict - new value (dict) should override
    tracker4 = UsageTracker()
    tracker4.add_usage("test-4", {"prompt_tokens": 100, "details": 20})
    tracker4.add_usage("test-4", {"prompt_tokens": 200, "details": {"cached_tokens": 10}})
    total4 = tracker4.get_total_tokens()
    assert total4["test-4"]["prompt_tokens"] == 300
    assert isinstance(total4["test-4"]["details"], dict)
    assert total4["test-4"]["details"]["cached_tokens"] == 10


def test_merge_usage_entries_with_nested_none_dict_int():
    """Test merging usage entries with complex nested structures containing None, dict, and int.

    When types are mixed, the new value (right value) should override the old value.
    """
    tracker = UsageTracker()

    # First usage entry: nested structure with None
    usage_entry1 = {
        "prompt_tokens": 100,
        "details": {
            "cached_tokens": None,
            "audio_tokens": 5,
            "nested": {
                "value": 10,
            },
        },
    }

    # Second usage entry: nested structure with dict and int
    usage_entry2 = {
        "prompt_tokens": 200,
        "details": {
            "cached_tokens": 20,  # None -> int: new value (int) should override
            "audio_tokens": {
                "input": 3,  # int -> dict: new value (dict) should override
                "output": 2,
            },
            "nested": 15,  # dict -> int: new value (int) should override
        },
    }

    tracker.add_usage("test-model", usage_entry1)
    tracker.add_usage("test-model", usage_entry2)

    total_usage = tracker.get_total_tokens()

    assert "test-model" in total_usage
    assert total_usage["test-model"]["prompt_tokens"] == 300  # 100 + 200
    assert isinstance(total_usage["test-model"]["details"], dict)
    # None + int = int (new value overrides)
    assert total_usage["test-model"]["details"]["cached_tokens"] == 20  # 0 + 20
    # int -> dict: new value (dict) should override old value (int)
    assert isinstance(total_usage["test-model"]["details"]["audio_tokens"], dict)
    assert total_usage["test-model"]["details"]["audio_tokens"]["input"] == 3
    assert total_usage["test-model"]["details"]["audio_tokens"]["output"] == 2
    # dict -> int: new value (int) should override old value (dict)
    assert total_usage["test-model"]["details"]["nested"] == 15
