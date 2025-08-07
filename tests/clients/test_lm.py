import time
from unittest import mock
from unittest.mock import patch

import litellm
import pydantic
import pytest
from litellm.utils import Choices, Message, ModelResponse
from openai import RateLimitError

import dspy
from dspy.utils.usage_tracker import track_usage


def test_chat_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert azure_openai_lm("azure openai query") == expected_response


@pytest.mark.parametrize(
    ("cache", "cache_in_memory"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_litellm_cache(litellm_test_server, cache, cache_in_memory):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    original_cache = dspy.cache
    dspy.clients.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=cache_in_memory,
        enable_litellm_cache=cache,
    )

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
        cache=cache,
        cache_in_memory=cache_in_memory,
    )
    assert openai_lm("openai query") == expected_response

    # Reset the cache configuration
    dspy.cache = original_cache


def test_dspy_cache(litellm_test_server, tmp_path):
    api_base, _ = litellm_test_server

    original_cache = dspy.cache
    dspy.clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        enable_litellm_cache=False,
        disk_cache_dir=tmp_path / ".disk_cache",
    )
    cache = dspy.cache

    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    with track_usage() as usage_tracker:
        lm("Query")

    assert len(cache.memory_cache) == 1
    cache_key = next(iter(cache.memory_cache.keys()))
    assert cache_key in cache.disk_cache
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm("Query")

    assert len(usage_tracker.usage_data) == 0

    dspy.cache = original_cache


def test_text_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    openai_lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert azure_openai_lm("azure openai query") == expected_response


def test_lm_calls_support_callables(litellm_test_server):
    api_base, _ = litellm_test_server

    with mock.patch("litellm.completion", autospec=True, wraps=litellm.completion) as spy_completion:

        def azure_ad_token_provider(*args, **kwargs):
            return None

        lm_with_callable = dspy.LM(
            model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            azure_ad_token_provider=azure_ad_token_provider,
            cache=False,
        )

        lm_with_callable("Query")

        spy_completion.assert_called_once()
        call_args = spy_completion.call_args.kwargs
        assert call_args["model"] == "openai/dspy-test-model"
        assert call_args["api_base"] == api_base
        assert call_args["api_key"] == "fakekey"
        assert call_args["azure_ad_token_provider"] is azure_ad_token_provider


def test_lm_calls_support_pydantic_models(litellm_test_server):
    api_base, _ = litellm_test_server

    class ResponseFormat(pydantic.BaseModel):
        response: str

    lm = dspy.LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        response_format=ResponseFormat,
    )
    lm("Query")


def test_retry_number_set_correctly():
    lm = dspy.LM("openai/gpt-4o-mini", num_retries=3)
    with mock.patch("litellm.completion") as mock_completion:
        lm("query")

    assert mock_completion.call_args.kwargs["num_retries"] == 3


def test_retry_made_on_system_errors():
    retry_tracking = [0]  # Using a list to track retries

    def mock_create(*args, **kwargs):
        retry_tracking[0] += 1
        # These fields are called during the error handling
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=250, num_retries=3)
    with mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create):
        with pytest.raises(RateLimitError):
            lm("question")

    assert retry_tracking[0] == 4


def test_reasoning_model_token_parameter():
    test_cases = [
        ("openai/o1", True),
        ("openai/o1-mini", True),
        ("openai/o1-2023-01-01", True),
        ("openai/o3", True),
        ("openai/o3-mini-2023-01-01", True),
        ("openai/gpt-4", False),
        ("anthropic/claude-2", False),
    ]

    for model_name, is_reasoning_model in test_cases:
        lm = dspy.LM(
            model=model_name,
            temperature=1.0 if is_reasoning_model else 0.7,
            max_tokens=20_000 if is_reasoning_model else 1000,
        )
        if is_reasoning_model:
            assert "max_completion_tokens" in lm.kwargs
            assert "max_tokens" not in lm.kwargs
            assert lm.kwargs["max_completion_tokens"] == 20_000
        else:
            assert "max_completion_tokens" not in lm.kwargs
            assert "max_tokens" in lm.kwargs
            assert lm.kwargs["max_tokens"] == 1000


def test_reasoning_model_requirements():
    # Should raise assertion error if temperature or max_tokens requirements not met
    with pytest.raises(AssertionError) as exc_info:
        dspy.LM(
            model="openai/o1",
            temperature=0.7,  # Should be 1.0
            max_tokens=1000,  # Should be >= 20_000
        )
    assert "reasoning models require passing temperature=1.0 and max_tokens >= 20_000" in str(exc_info.value)

    # Should pass with correct parameters
    lm = dspy.LM(
        model="openai/o1",
        temperature=1.0,
        max_tokens=20_000,
    )
    assert lm.kwargs["max_completion_tokens"] == 20_000


def test_dump_state():
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
        launch_kwargs={"temperature": 1},
        train_kwargs={"temperature": 5},
    )

    assert lm.dump_state() == {
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        "cache": True,
        "cache_in_memory": True,
        "finetuning_model": None,
        "launch_kwargs": {"temperature": 1},
        "train_kwargs": {"temperature": 5},
    }


def test_exponential_backoff_retry():
    time_counter = []

    def mock_create(*args, **kwargs):
        time_counter.append(time.time())
        # These fields are called during the error handling
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = dspy.LM(model="openai/gpt-3.5-turbo", max_tokens=250, num_retries=3)
    with mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create):
        with pytest.raises(RateLimitError):
            lm("question")

    # The first retry happens immediately regardless of the configuration
    for i in range(1, len(time_counter) - 1):
        assert time_counter[i + 1] - time_counter[i] >= 2 ** (i - 1)


def test_logprobs_included_when_requested():
    lm = dspy.LM(model="dspy-test-model", logprobs=True, cache=False)
    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(content="test answer"),
                    logprobs={
                        "content": [
                            {"token": "test", "logprob": 0.1, "top_logprobs": [{"token": "test", "logprob": 0.1}]},
                            {"token": "answer", "logprob": 0.2, "top_logprobs": [{"token": "answer", "logprob": 0.2}]},
                        ]
                    },
                )
            ],
            model="dspy-test-model",
        )
        result = lm("question")
        assert result[0]["text"] == "test answer"
        assert result[0]["logprobs"].model_dump() == {
            "content": [
                {
                    "token": "test",
                    "bytes": None,
                    "logprob": 0.1,
                    "top_logprobs": [{"token": "test", "bytes": None, "logprob": 0.1}],
                },
                {
                    "token": "answer",
                    "bytes": None,
                    "logprob": 0.2,
                    "top_logprobs": [{"token": "answer", "bytes": None, "logprob": 0.2}],
                },
            ]
        }
        assert mock_completion.call_args.kwargs["logprobs"]


@pytest.mark.asyncio
async def test_async_lm_call():
    from litellm.utils import Choices, Message, ModelResponse

    mock_response = ModelResponse(choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini")

    with patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = mock_response

        lm = dspy.LM(model="openai/gpt-4o-mini", cache=False)
        result = await lm.acall("question")

        assert result == ["answer"]
        mock_acompletion.assert_called_once()


@pytest.mark.asyncio
async def test_async_lm_call_with_cache(tmp_path):
    """Test the async LM call with caching."""
    original_cache = dspy.cache
    dspy.clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        enable_litellm_cache=False,
        disk_cache_dir=tmp_path / ".disk_cache",
    )
    cache = dspy.cache

    lm = dspy.LM(model="openai/gpt-4o-mini")

    with mock.patch("dspy.clients.lm.alitellm_completion") as mock_alitellm_completion:
        mock_alitellm_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini"
        )
        mock_alitellm_completion.__qualname__ = "alitellm_completion"
        await lm.acall("Query")

        assert len(cache.memory_cache) == 1
        cache_key = next(iter(cache.memory_cache.keys()))
        assert cache_key in cache.disk_cache
        assert mock_alitellm_completion.call_count == 1

        await lm.acall("Query")
        # Second call should hit the cache, so no new call to LiteLLM is made.
        assert mock_alitellm_completion.call_count == 1

        # Test that explicitly disabling memory cache works
        await lm.acall("New query", cache_in_memory=False)

        # There should be a new call to LiteLLM on new query, but the memory cache shouldn't be written to.
        assert len(cache.memory_cache) == 1
        assert mock_alitellm_completion.call_count == 2

    dspy.cache = original_cache


def test_lm_history_size_limit():
    lm = dspy.LM(model="openai/gpt-4o-mini")
    with dspy.context(max_history_size=5):
        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="test answer"))],
                model="openai/gpt-4o-mini",
            )

            for _ in range(10):
                lm("query")

    assert len(lm.history) == 5


def test_disable_history():
    lm = dspy.LM(model="openai/gpt-4o-mini")
    with dspy.context(disable_history=True):
        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="test answer"))],
                model="openai/gpt-4o-mini",
            )
            for _ in range(10):
                lm("query")

    assert len(lm.history) == 0

    with dspy.context(disable_history=False):
        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="test answer"))],
                model="openai/gpt-4o-mini",
            )
