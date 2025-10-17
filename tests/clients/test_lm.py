import json
import tempfile
import time
import warnings
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import litellm
import pydantic
import pytest
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.utils import Choices, Message, ModelResponse
from openai import RateLimitError
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

import dspy
from dspy.utils.dummies import DummyLM
from dspy.utils.usage_tracker import track_usage


def make_response(output_blocks):
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/dspy-test-model",
        object="response",
        output=output_blocks,
        metadata = {},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        user=None,
    )


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


def test_dspy_cache(litellm_test_server, tmp_path):
    api_base, _ = litellm_test_server

    original_cache = dspy.cache
    dspy.clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
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


def test_disabled_cache_skips_cache_key(monkeypatch):
    original_cache = dspy.cache
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    cache = dspy.cache

    try:
        with mock.patch.object(cache, "cache_key", wraps=cache.cache_key) as cache_key_spy, \
             mock.patch.object(cache, "get", wraps=cache.get) as cache_get_spy, \
             mock.patch.object(cache, "put", wraps=cache.put) as cache_put_spy:

            def fake_completion(*, cache, num_retries, retry_strategy, **request):
                return ModelResponse(
                    choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    model="dummy",
                )

            monkeypatch.setattr(litellm, "completion", fake_completion)

            dummy_lm = DummyLM([{"answer": "ignored"}])
            # TODO(isaacbmiller): Change from dummy_lm.forward to just dummy_lm.__call__ #8864
            dummy_lm.forward(messages=[{"role": "user", "content": "Hello"}])

            cache_key_spy.assert_not_called()
            cache_get_spy.assert_called_once()
            cache_put_spy.assert_called_once()
    finally:
        dspy.cache = original_cache


def test_rollout_id_bypasses_cache(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_completion(*, cache, num_retries, retry_strategy, **request):
        calls.append(request)
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/dspy-test-model",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    original_cache = dspy.cache
    dspy.clients.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path / ".disk_cache",
    )

    lm = dspy.LM(model="openai/dspy-test-model", model_type="chat")

    with track_usage() as usage_tracker:
        lm(messages=[{"role": "user", "content": "Query"}], rollout_id=1)
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(messages=[{"role": "user", "content": "Query"}], rollout_id=1)
    assert len(usage_tracker.usage_data) == 0

    with track_usage() as usage_tracker:
        lm(messages=[{"role": "user", "content": "Query"}], rollout_id=2)
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(messages=[{"role": "user", "content": "NoRID"}])
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm(messages=[{"role": "user", "content": "NoRID"}], rollout_id=None)
    assert len(usage_tracker.usage_data) == 0

    assert len(dspy.cache.memory_cache) == 3
    assert all("rollout_id" not in r for r in calls)
    dspy.cache = original_cache


def test_zero_temperature_rollout_warns_once(monkeypatch):
    def fake_completion(*, cache, num_retries, retry_strategy, **request):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/dspy-test-model",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    lm = dspy.LM(model="openai/dspy-test-model", model_type="chat")
    with pytest.warns(UserWarning, match="rollout_id has no effect"):
        lm("Query", rollout_id=1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm("Query", rollout_id=2)
        assert len(record) == 0


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
        ("openai/gpt-5", True),
        ("openai/gpt-5-mini", True),
        ("openai/gpt-5-nano", True),
        ("openai/gpt-4", False),
        ("anthropic/claude-2", False),
    ]

    for model_name, is_reasoning_model in test_cases:
        lm = dspy.LM(
            model=model_name,
            temperature=1.0 if is_reasoning_model else 0.7,
            max_tokens=16_000 if is_reasoning_model else 1000,
        )
        if is_reasoning_model:
            assert "max_completion_tokens" in lm.kwargs
            assert "max_tokens" not in lm.kwargs
            assert lm.kwargs["max_completion_tokens"] == 16_000
        else:
            assert "max_completion_tokens" not in lm.kwargs
            assert "max_tokens" in lm.kwargs
            assert lm.kwargs["max_tokens"] == 1000

@pytest.mark.parametrize("model_name", ["openai/o1", "openai/gpt-5-nano"])
def test_reasoning_model_requirements(model_name):
    # Should raise assertion error if temperature or max_tokens requirements not met
    with pytest.raises(
        ValueError,
        match="reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None",
    ):
        dspy.LM(
            model=model_name,
            temperature=0.7,  # Should be 1.0
            max_tokens=1000,  # Should be >= 16_000
        )

    # Should pass with correct parameters
    lm = dspy.LM(
        model=model_name,
        temperature=1.0,
        max_tokens=16_000,
    )
    assert lm.kwargs["max_completion_tokens"] == 16_000

    # Should pass with no parameters
    lm = dspy.LM(
        model=model_name,
    )
    assert lm.kwargs["temperature"] == None
    assert lm.kwargs["max_completion_tokens"] == None


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

        # A new query should result in a new LiteLLM call and a new cache entry.
        await lm.acall("New query")

        assert len(cache.memory_cache) == 2
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

def test_responses_api():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                **{
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "This is a test answer from responses API.", "annotations": []}
                    ],
                },
            ),
            ResponseReasoningItem(
                **{
                    "id": "reasoning_1",
                    "type": "reasoning",
                    "summary": [Summary(**{"type": "summary_text", "text": "This is a dummy reasoning."})],
                },
            ),
        ]
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = dspy.LM(
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )
        lm_result = lm("openai query")

        assert lm_result == [
            {
                "text": "This is a test answer from responses API.",
                "reasoning_content": "This is a dummy reasoning.",
            }
        ]

        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/gpt-5-mini"


def test_lm_replaces_system_with_developer_role():
    with mock.patch(
        "dspy.clients.lm.litellm_responses_completion", return_value={"choices": []}
    ) as mock_completion:
        lm = dspy.LM(
            "openai/gpt-4o-mini",
            cache=False,
            model_type="responses",
            use_developer_role=True,
        )
        lm.forward(messages=[{"role": "system", "content": "hi"}])
        assert (
            mock_completion.call_args.kwargs["request"]["messages"][0]["role"]
            == "developer"
        )


def test_responses_api_tool_calls(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_tool_call = {
        "type": "function_call",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris"}),
        "call_id": "call_1",
        "status": "completed",
        "id": "call_1",
    }
    expected_response = [{"tool_calls": [expected_tool_call]}]

    api_response = make_response(
        output_blocks=[expected_tool_call],
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = dspy.LM(
            model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            model_type="responses",
            cache=False,
        )
        assert lm("openai query") == expected_response

        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/dspy-test-model"


def test_api_key_not_saved_in_json():
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1.0,
        max_tokens=100,
        api_key="sk-test-api-key-12345",
    )

    predict = dspy.Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)

        with open(json_path) as f:
            saved_state = json.load(f)

        # Verify API key is not in the saved state
        assert "api_key" not in saved_state.get("lm", {}), "API key should not be saved in JSON"

        # Verify other attributes are saved
        assert saved_state["lm"]["model"] == "openai/gpt-4o-mini"
        assert saved_state["lm"]["temperature"] == 1.0
        assert saved_state["lm"]["max_tokens"] == 100
