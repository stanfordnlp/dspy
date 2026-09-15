import json
import tempfile
import warnings
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import litellm
import pydantic
import pytest
import tenacity
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.utils import Choices, Message, ModelResponse
from openai import RateLimitError
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

import dspy
from dspy.utils.usage_tracker import track_usage
from tests.test_utils.engines import litellm_response, recording_lm
from tests.test_utils.engines import make_response as make_lm_response


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
        metadata={},
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
        engine="litellm", model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="chat",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        engine="litellm", model="azure/dspy-test-model",
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
        engine="litellm", model="openai/dspy-test-model",
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
        with (
            mock.patch.object(cache, "cache_key", wraps=cache.cache_key) as cache_key_spy,
            mock.patch.object(cache, "get", wraps=cache.get) as cache_get_spy,
            mock.patch.object(cache, "put", wraps=cache.put) as cache_put_spy,
        ):

            def fake_completion(*, cache, num_retries, **request):
                return ModelResponse(
                    choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    model="dummy",
                )

            monkeypatch.setattr(litellm, "completion", fake_completion)

            lm = dspy.LM("dummy", engine="litellm", model_type="chat")
            lm("Hello")

            cache_key_spy.assert_not_called()
            cache_get_spy.assert_called_once()
            cache_put_spy.assert_called_once()
    finally:
        dspy.cache = original_cache


def test_rollout_id_bypasses_cache(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_completion(*, cache, num_retries, **request):
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

    lm = dspy.LM(engine="litellm", model="openai/dspy-test-model", model_type="chat")

    with track_usage() as usage_tracker:
        lm("Query", rollout_id=1)
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm("Query", rollout_id=1)
    assert len(usage_tracker.usage_data) == 0

    with track_usage() as usage_tracker:
        lm("Query", rollout_id=2)
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm("NoRID")
    assert len(usage_tracker.usage_data) == 1

    with track_usage() as usage_tracker:
        lm("NoRID", rollout_id=None)
    assert len(usage_tracker.usage_data) == 0

    assert len(dspy.cache.memory_cache) == 3
    assert all("rollout_id" not in r for r in calls)
    dspy.cache = original_cache


def test_zero_temperature_rollout_warns_once(monkeypatch):
    def fake_completion(*, cache, num_retries, **request):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/dspy-test-model",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    lm = dspy.LM(engine="litellm", model="openai/dspy-test-model", model_type="chat", temperature=0)
    with pytest.warns(UserWarning, match="rollout_id has no effect"):
        lm("Query", rollout_id=1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm("Query", rollout_id=2)
        assert len(record) == 0


def test_rollout_id_with_default_temperature_does_not_warn(monkeypatch):
    def fake_completion(*, cache, num_retries, **request):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/gpt-5-nano",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm = dspy.LM(engine="litellm", model="openai/gpt-5-nano", model_type="chat", rollout_id=1)
        lm("Query")
        assert len(record) == 0


def test_text_lms_can_be_queried(litellm_test_server):
    api_base, _ = litellm_test_server
    expected_response = ["Hi!"]

    openai_lm = dspy.LM(
        engine="litellm", model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert openai_lm("openai query") == expected_response

    azure_openai_lm = dspy.LM(
        engine="litellm", model="azure/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        model_type="text",
    )
    assert azure_openai_lm("azure openai query") == expected_response


def test_lm_calls_support_callables(litellm_test_server):
    api_base, _ = litellm_test_server

    real_completion = litellm.completion

    def call_through(**kwargs):
        return real_completion(**kwargs)

    with mock.patch("litellm.completion", side_effect=call_through) as spy_completion:

        def azure_ad_token_provider(*args, **kwargs):
            return None

        lm_with_callable = dspy.LM(
            engine="litellm", model="openai/dspy-test-model",
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
        engine="litellm", model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        response_format=ResponseFormat,
    )
    lm("Query")


def _litellm_failure(lm, error):
    with mock.patch("litellm.completion", side_effect=error):
        with pytest.raises(dspy.LMError) as exc_info:
            lm("question")
    return exc_info.value


def test_lm_wraps_litellm_errors_with_metadata():
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, num_retries=0)
    response = mock.Mock()
    response.status_code = 429
    response.headers = {"x-request-id": "req-123", "retry-after": "2.5"}

    error = litellm.RateLimitError(message="too many requests", llm_provider="openai", model="gpt-4o", response=response)
    wrapped = _litellm_failure(lm, error)

    assert isinstance(wrapped, dspy.LMRateLimitError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"
    assert wrapped.status == 429
    assert wrapped.request_id == "req-123"
    assert wrapped.retry_after == 2.5


def test_lm_wraps_litellm_context_window_error():
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, num_retries=0)
    error = litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model="gpt-4o")
    wrapped = _litellm_failure(lm, error)

    assert isinstance(wrapped, dspy.ContextWindowExceededError)
    assert isinstance(wrapped, dspy.LMError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"


def test_lm_wraps_unknown_boundary_error_as_unexpected_error():
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, num_retries=0)
    wrapped = _litellm_failure(lm, RuntimeError("local boundary failure"))

    assert isinstance(wrapped, dspy.LMUnexpectedError)
    assert wrapped.code == "unexpected"
    assert wrapped.model == "openai/gpt-4o-mini"


def test_lm_preserves_existing_lm_error_without_self_cause():
    error = dspy.LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False)

    with mock.patch("litellm.completion", side_effect=error):
        with pytest.raises(dspy.LMRateLimitError) as exc_info:
            lm("question")

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_lm_preserves_existing_lm_error_without_self_cause_async():
    error = dspy.LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False)

    with mock.patch("litellm.acompletion", side_effect=error):
        with pytest.raises(dspy.LMRateLimitError) as exc_info:
            await lm.acall("question")

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_retry_number_set_correctly():
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", num_retries=3)
    with mock.patch("litellm.completion", return_value=litellm_response("ok")) as mock_completion:
        lm("query")

    # DSPy owns retries; every individual backend attempt disables them.
    assert mock_completion.call_args.kwargs["num_retries"] == 0


def test_retry_made_on_system_errors():
    retry_tracking = [0]  # Using a list to track retries

    def mock_create(*args, **kwargs):
        retry_tracking[0] += 1
        # These fields are called during the error handling
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    original_retrying = tenacity.Retrying

    def immediate_retrying(*args, **kwargs):
        kwargs["sleep"] = lambda _: None
        return original_retrying(*args, **kwargs)

    lm = dspy.LM(engine="litellm", model="openai/gpt-4o-mini", max_tokens=250, num_retries=3)
    with (
        mock.patch("tenacity.Retrying", side_effect=immediate_retrying),
        mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create),
    ):
        with pytest.raises(dspy.LMRateLimitError):
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
        ("azure/gpt-5-chat", False),  # gpt-5-chat is NOT a reasoning model
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


def test_lm_supports_reasoning_with_litellm_capability_api():
    lm = dspy.LM("anthropic/claude-3-7-sonnet-20250219")
    assert lm.supports_reasoning is True


@pytest.mark.parametrize("model_name", ["openai/o1", "openai/gpt-5-nano", "openai/gpt-5-mini"])
def test_reasoning_model_requirements(model_name):
    # Should raise assertion error if temperature or max_tokens requirements not met
    with pytest.raises(
        dspy.LMConfigurationError,
        match=r"reasoning models require passing temperature=1\.0 or None and max_tokens >= 16000 or None",
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
    assert lm.kwargs["temperature"] is None
    assert lm.kwargs["max_completion_tokens"] is None


def test_gpt_5_chat_not_reasoning_model():
    """Test that gpt-5-chat is NOT treated as a reasoning model."""
    # Should NOT raise validation error - gpt-5-chat is not a reasoning model
    lm = dspy.LM(
        model="openai/gpt-5-chat",
        temperature=0.7,  # Can be any value
        max_tokens=1000,  # Can be any value
    )
    # Should use max_tokens, not max_completion_tokens
    assert "max_completion_tokens" not in lm.kwargs
    assert "max_tokens" in lm.kwargs
    assert lm.kwargs["max_tokens"] == 1000
    assert lm.kwargs["temperature"] == 0.7


def test_base_lm_init_uses_lm_defaults_and_isolates_callback_list():
    callbacks = [object()]
    lm = dspy.BaseLM("custom-model", callbacks=callbacks)

    assert lm.kwargs == {"temperature": None, "max_tokens": None}
    assert lm.num_retries == 3
    assert lm.callbacks == callbacks
    assert lm.callbacks is not callbacks
















# BaseLM engine tests: a custom LM is a BaseLM bound to an engine.


def test_base_lm_runs_its_engine_and_returns_list_outputs():
    lm = recording_lm(["Hi!"], model="custom-model")
    assert lm("Query") == ["Hi!"]
    [request] = lm.engine.requests
    assert request.messages[0].text == "Query"


def test_base_lm_rejects_engines_that_do_not_return_responses():
    class BadEngine:
        def complete(self, request):
            return ["not a response"]

    lm = dspy.LM("custom-model", engine=BadEngine(), cache=False, num_retries=0)
    with pytest.raises(dspy.LMUnexpectedError, match=r"must return dspy\.lm15\.Response"):
        lm("Query")


def test_base_lm_tracks_usage_for_custom_engines():
    from dspy.lm15 import Usage

    lm = recording_lm([make_lm_response("Hi!", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2))],
                      model="custom-model")

    with track_usage() as usage_tracker:
        lm("Query")

    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 1
    assert total_usage["total_tokens"] == 2


def test_base_lm_copy_is_shallow_runtime_copy_with_isolated_dspy_state():
    class CustomLM(dspy.BaseLM):
        pass

    callback = object()
    client = object()
    lm = CustomLM(model="custom-model", callbacks=[callback], temperature=0.1)
    lm.client = client
    lm.extra_state = {"mutable": []}
    lm.history = [{"prompt": "original"}]

    copied_lm = lm.copy(temperature=0.2, rollout_id=1)

    assert copied_lm is not lm
    assert copied_lm.client is client
    assert copied_lm.extra_state is lm.extra_state
    assert copied_lm.history == []
    assert copied_lm.history is not lm.history
    assert copied_lm.callbacks == [callback]
    assert copied_lm.callbacks is not lm.callbacks
    assert copied_lm.kwargs == {"temperature": 0.2, "max_tokens": None, "rollout_id": 1}
    assert lm.kwargs == {"temperature": 0.1, "max_tokens": None}


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
        "_dspy_lm_class": "dspy.clients.lm.LM",
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


def test_reasoning_model_dump_state_uses_constructor_max_tokens():
    lm = dspy.LM(
        model="openai/gpt-5-nano",
        temperature=1.0,
        max_tokens=16_000,
        cache=False,
        num_retries=1,
    )

    state = lm.dump_state()

    assert lm.kwargs["max_completion_tokens"] == 16_000
    assert "max_completion_tokens" not in state
    assert state["max_tokens"] == 16_000


def test_dump_state_preserves_enabled_developer_role():
    lm = dspy.LM("openai/gpt-4o-mini", use_developer_role=True)

    assert lm.dump_state()["use_developer_role"] is True
    assert dspy.LM.load_state(lm.dump_state()).use_developer_role is True


def test_dump_state_ignores_internal_class_marker_kwarg():
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        **{"_dspy_lm_class": "malicious.module.LM"},
    )

    dumped_state = lm.dump_state()

    assert dumped_state["_dspy_lm_class"] == "dspy.clients.lm.LM"
    assert lm.kwargs["_dspy_lm_class"] == "malicious.module.LM"


def test_load_state():
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
        launch_kwargs={"temperature": 1},
        train_kwargs={"temperature": 5},
    )

    loaded_lm = dspy.LM.load_state(lm.dump_state())

    assert isinstance(loaded_lm, dspy.LM)
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_round_trips_canonical_state():
    lm = dspy.LM(
        model="openai/gpt-5-nano",
        temperature=1.0,
        max_tokens=16_000,
        cache=False,
        num_retries=1,
    )

    loaded_lm = dspy.BaseLM.load_state(lm.dump_state())

    assert isinstance(loaded_lm, dspy.LM)
    assert loaded_lm.kwargs["max_completion_tokens"] == 16_000
    assert loaded_lm.dump_state() == lm.dump_state()


def test_reasoning_model_load_state_accepts_max_completion_tokens_alias():
    state = {
        "_dspy_lm_class": "dspy.clients.lm.LM",
        "model": "openai/gpt-5-nano",
        "model_type": "chat",
        "cache": False,
        "num_retries": 1,
        "temperature": 1.0,
        "max_completion_tokens": 16_000,
        "finetuning_model": None,
        "launch_kwargs": {},
        "train_kwargs": {},
    }

    loaded_lm = dspy.BaseLM.load_state(state)

    assert isinstance(loaded_lm, dspy.LM)
    assert loaded_lm.kwargs["max_completion_tokens"] == 16_000
    assert "max_completion_tokens" not in loaded_lm.dump_state()
    assert loaded_lm.dump_state()["max_tokens"] == 16_000


def test_lm_load_state_forwards_allow_custom_lm_class(monkeypatch):
    calls = []
    original_load_state = dspy.BaseLM.load_state.__func__

    def spy_load_state(cls, state, *, allow_custom_lm_class=False):
        calls.append(allow_custom_lm_class)
        return original_load_state(cls, state, allow_custom_lm_class=allow_custom_lm_class)

    monkeypatch.setattr(dspy.BaseLM, "load_state", classmethod(spy_load_state))

    dspy.LM.load_state(dspy.LM("openai/gpt-4o-mini", engine="litellm").dump_state(), allow_custom_lm_class=True)

    assert calls == [True]


def test_exponential_backoff_retry():
    retry_delays = []

    def mock_create(*args, **kwargs):
        # These fields are called during the error handling
        mock_response = mock.Mock()
        mock_response.headers = {}
        mock_response.status_code = 429
        raise RateLimitError(response=mock_response, message="message", body="error")

    lm = dspy.LM(engine="litellm", model="openai/gpt-3.5-turbo", max_tokens=250, num_retries=3)
    with (
        # Replace this module's reference, not the process-wide time.sleep:
        # background SDK threads may also sleep while the request runs.
        mock.patch("dspy.clients.execution.time", mock.Mock(sleep=retry_delays.append)),
        mock.patch.object(litellm.OpenAIChatCompletion, "completion", side_effect=mock_create) as completion,
    ):
        with pytest.raises(dspy.LMRateLimitError):
            lm("question")

    assert retry_delays == [1, 2, 4]
    assert completion.call_count == 4


def test_logprobs_included_when_requested():
    lm = dspy.LM(engine="litellm", model="dspy-test-model", logprobs=True, cache=False)
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

        lm = dspy.LM(engine="litellm", model="openai/gpt-4o-mini", cache=False)
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

    lm = dspy.LM(engine="litellm", model="openai/gpt-4o-mini")

    with mock.patch("litellm.acompletion") as mock_alitellm_completion:
        mock_alitellm_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="answer"))], model="openai/gpt-4o-mini"
        )
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
    lm = dspy.LM(engine="litellm", model="openai/gpt-4o-mini")
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
    lm = dspy.LM(engine="litellm", model="openai/gpt-4o-mini")
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
            engine="litellm", model="openai/gpt-5-mini",
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
    from dspy.adapters import Prompt
    from dspy.clients.requests import build_request
    from dspy.lm15 import Message as LMMessage

    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, model_type="responses", use_developer_role=True)
    request = build_request(lm, Prompt(system="hi", messages=(LMMessage.user("q"),)), {})

    # The instruction travels as a leading developer message, not as Request.system.
    assert request.system is None
    assert request.messages[0].role == "developer"
    assert request.messages[0].text == "hi"
    with mock.patch("litellm.responses", return_value=_responses_text_response()) as mock_responses:
        lm(request)
    assert mock_responses.call_args.kwargs["input"][0]["role"] == "developer"


@pytest.mark.parametrize(
    "provider_fields",
    [
        {},
        {"caller": None, "namespace": None},
        {"caller": {"type": "direct"}, "namespace": "collaboration"},
    ],
    ids=["legacy", "empty-provider-fields", "populated-provider-fields"],
)
def test_responses_api_tool_calls(litellm_test_server, provider_fields):
    api_base, _ = litellm_test_server
    base_tool_call = {
        "type": "function_call",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris"}),
        "call_id": "call_1",
        "status": "completed",
        "id": "fc_1",
    }
    # Legacy outputs use the chat-unified tool-call shape regardless of which
    # provider fields the Responses item carries; the full provider item stays
    # available on the typed path via LMToolCallPart.provider_data.
    expected_response = [
        {
            "text": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    "id": "call_1",
                }
            ],
        }
    ]

    api_response = make_response(
        output_blocks=[{**base_tool_call, **provider_fields}],
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = dspy.LM(
            engine="litellm", model="openai/dspy-test-model",
            api_base=api_base,
            api_key="fakekey",
            model_type="responses",
            cache=False,
        )
        assert lm("openai query") == expected_response

        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/dspy-test-model"


def test_responses_api_cache_hit_preserves_outputs_and_skips_usage(tmp_path):
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "cached answer", "annotations": []}],
            ),
            ResponseReasoningItem(
                id="reasoning_1",
                type="reasoning",
                summary=[Summary(type="summary_text", text="cached reasoning")],
            ),
        ],
    )

    original_cache = dspy.cache
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=tmp_path / ".dspy_cache")
    try:
        with mock.patch("litellm.responses", autospec=True, return_value=api_response) as responses:
            lm = dspy.LM("openai/dspy-test-model", engine="litellm", model_type="responses")
            with track_usage() as first_usage:
                first = lm("cache me")
            with track_usage() as second_usage:
                second = lm("cache me")

        assert first == [{"text": "cached answer", "reasoning_content": "cached reasoning"}]
        assert second == first
        assert responses.call_count == 1
        # The fresh call records usage; the cache hit must not.
        assert len(first_usage.usage_data) == 1
        assert len(second_usage.usage_data) == 0
        assert lm.history[-1]["usage"] == {}
    finally:
        dspy.cache = original_cache


def test_responses_api_joins_multiple_text_outputs():
    """Multiple message items must join into one text string, never leak a list."""
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "part one. ", "annotations": []}],
            ),
            ResponseOutputMessage(
                id="msg_2",
                type="message",
                role="assistant",
                status="completed",
                content=[{"type": "output_text", "text": "part two.", "annotations": []}],
            ),
        ],
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response):
        lm = dspy.LM("openai/dspy-test-model", engine="litellm", model_type="responses", cache=False)
        outputs = lm("multi part query")

    assert outputs == [{"text": "part one. part two."}]


def test_reasoning_effort_responses_api():
    """Test that reasoning_effort gets normalized to reasoning format for Responses API."""
    with mock.patch("litellm.responses", return_value=make_response([])) as mock_responses:
        lm = dspy.LM(
            engine="litellm", model="openai/gpt-5", model_type="responses", reasoning_effort="low", max_tokens=16000, temperature=1.0
        )
        lm("openai query")
        call_kwargs = mock_responses.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["reasoning"] == {"effort": "low", "summary": "auto"}


def test_call_reasoning_model_with_chat_api():
    """Test that Chat API properly handles reasoning models and returns data in correct format."""
    # Create message with reasoning_content attribute
    message = Message(content="The answer is 4", role="assistant")
    # Add reasoning_content attribute
    message.reasoning_content = "Step 1: I need to add 2 + 2\nStep 2: 2 + 2 = 4\nTherefore, the answer is 4"

    # Create choice with the message
    mock_choice = Choices(message=message)

    # Mock response with reasoning content for chat completion
    mock_response = ModelResponse(
        choices=[mock_choice],
        model="anthropic/claude-3-7-sonnet-20250219",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )

    with mock.patch("litellm.completion", return_value=mock_response) as mock_completion:
        with mock.patch("litellm.supports_reasoning", return_value=True):
            # Create reasoning model with chat API
            lm = dspy.LM(
                engine="litellm", model="anthropic/claude-3-7-sonnet-20250219",
                model_type="chat",
                temperature=1.0,
                max_tokens=16000,
                reasoning_effort="low",
                cache=False,
            )

            # Test the call
            result = lm("What is 2 + 2?")

            # Verify the response format
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert "text" in result[0]
            assert "reasoning_content" in result[0]
            assert result[0]["text"] == "The answer is 4"
            assert "Step 1" in result[0]["reasoning_content"]

            # Verify mock was called with correct parameters
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args.kwargs
            assert call_kwargs["model"] == "anthropic/claude-3-7-sonnet-20250219"
            assert call_kwargs["reasoning_effort"] == "low"


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


def test_responses_api_with_image_input():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                **{
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "This is a test answer with image input.", "annotations": []}
                    ],
                },
            ),
        ]
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        lm = dspy.LM(
            engine="litellm", model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        # A request whose user message carries an image part
        from dspy.lm15 import ImagePart, Request
        from dspy.lm15 import Message as LMMessage

        request = Request(model=lm.model, messages=(LMMessage.user([
            "Describe this image",
            ImagePart(data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                      media_type="image/png"),
        ]),))

        lm_result = lm(request)

        assert lm_result.text == "This is a test answer with image input."

        dspy_responses.assert_called_once()
        call_args = dspy_responses.call_args.kwargs

        # Verify the request was converted correctly
        assert "input" in call_args
        content = call_args["input"][0]["content"]

        # Check that image was converted to input_image format
        image_content = [c for c in content if c.get("type") == "input_image"]
        assert len(image_content) == 1
        assert (
            image_content[0]["image_url"]
            == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )


def test_responses_api_with_pydantic_model_input():
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                **{
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"answer" : "This is a good test answer", "number" : 42}',
                            "annotations": [],
                        }
                    ],
                },
            ),
        ]
    )

    lm = dspy.LM(
        engine="litellm", model="openai/gpt-5-mini",
        model_type="responses",
        cache=False,
        temperature=1.0,
        max_tokens=16000,
    )

    class TestModel(pydantic.BaseModel):
        answer: str
        number: int

    with mock.patch("litellm.responses", autospec=True, return_value=api_response) as dspy_responses:
        # Test with messages containing a Pydantic model as response format
        lm_result = lm("What is a good test answer?", response_format=TestModel)

    # Try to validate to Pydantic model
    TestModel.model_validate_json(lm_result[0]["text"])

    dspy_responses.assert_called_once()
    call_args = dspy_responses.call_args.kwargs

    # Verify the request was converted correctly
    assert "text" in call_args
    response_format = call_args["text"]["format"]

    assert response_format == {
        "name": TestModel.__name__,
        "type": "json_schema",
        "schema": {**TestModel.model_json_schema(), "additionalProperties": False},
        "strict": True,
    }


def test_responses_api_with_none_usage():
    """Responses API returns usage=None for incomplete/truncated responses (e.g. max_output_tokens hit)."""
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details={"reason": "max_output_tokens"},
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                **{
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {"type": "output_text", "text": "Partial response that was truncated", "annotations": []}
                    ],
                },
            ),
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )

    with mock.patch("litellm.responses", autospec=True, return_value=api_response):
        lm = dspy.LM(
            engine="litellm", model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        with track_usage() as tracker:
            result = lm("test query")

        assert result == [{"text": "Partial response that was truncated"}]
        assert lm.history[-1]["usage"] == {}
        assert tracker.get_total_tokens() == {}


@pytest.mark.asyncio
async def test_responses_api_with_none_usage_async():
    """Async path: Responses API returns usage=None for incomplete/truncated responses."""
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0.0,
        error=None,
        incomplete_details={"reason": "max_output_tokens"},
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                **{
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete",
                    "content": [
                        {"type": "output_text", "text": "Partial async response", "annotations": []}
                    ],
                },
            ),
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )

    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response):
        lm = dspy.LM(
            engine="litellm", model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        with track_usage() as tracker:
            result = await lm.acall("test query")

        assert result == [{"text": "Partial async response"}]
        assert lm.history[-1]["usage"] == {}
        assert tracker.get_total_tokens() == {}


def test_litellm_engine_passes_headers_and_identifies_dspy():
    lm = dspy.LM("openai/gpt-4o-mini", engine="litellm", cache=False, headers={"Authorization": "Bearer my-custom-token"})
    with mock.patch("litellm.completion", return_value=litellm_response("ok")) as mock_completion:
        lm("test")
    headers = mock_completion.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer my-custom-token"
    assert headers["User-Agent"] == f"DSPy/{dspy.__version__}"


# ---------------------------------------------------------------------------
# Responses API request contract: the shapes DSPy must emit for tools,
# tool_choice, messages, and config on model_type="responses".
# ---------------------------------------------------------------------------


def _chat_shaped_weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _responses_text_response():
    return make_response(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                "id": "msg_1",
                "status": "completed",
            }
        ]
    )


class ContractSchema(pydantic.BaseModel):
    answer: str






def test_lm_responses_engine_writes_lm15_tools_and_documents():
    """The LiteLLM Responses engine serializes canonical tools and parts with lm15's own dialect."""
    from dspy.lm15 import BuiltinTool, DocumentPart, FunctionTool, Request, ToolChoice
    from dspy.lm15 import Message as LMMessage

    response = _responses_text_response()
    lm = dspy.LM("openai/dspy-test-model", engine="litellm", model_type="responses", cache=False)
    request = Request(
        model=lm.model,
        messages=(LMMessage.user(["Read this.", DocumentPart(data="JVBERi0xLjQK", media_type="application/pdf")]),),
        tools=(
            BuiltinTool("web_search", config={"search_context_size": "low"}),
            FunctionTool("get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}}),
        ),
        config=dspy.lm15.Config(tool_choice=ToolChoice(mode="required", allowed=("get_weather",))),
    )

    with mock.patch("litellm.responses", autospec=True, return_value=response) as responses:
        lm(request)

    sent = responses.call_args.kwargs
    assert sent["tools"][0]["type"].startswith("web_search")
    assert sent["tools"][1]["name"] == "get_weather"
    assert sent["tool_choice"] == {"type": "function", "name": "get_weather"}
    file_item = sent["input"][0]["content"][1]
    assert file_item["type"] == "input_file"
    assert file_item["file_data"] == "data:application/pdf;base64,JVBERi0xLjQK"


def test_lm_reads_responses_api_tool_spellings_from_options():
    """Flat function tools, hosted tools and their tool_choice forms become lm15 objects."""
    from dspy.clients.requests import build_request
    from dspy.lm15 import BuiltinTool, FunctionTool

    lm = dspy.LM("openai/dspy-test-model", engine="litellm", model_type="responses", cache=False)
    flat = {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}, "strict": True}

    request = build_request(lm, "hi", {"tools": [flat, {"type": "web_search_preview"}, _chat_shaped_weather_tool()],
                                       "tool_choice": {"type": "function", "name": "get_time"}})
    assert {(type(tool), tool.name) for tool in request.tools} == {
        (FunctionTool, "get_time"), (BuiltinTool, "web_search_preview"), (FunctionTool, "get_weather"),
    }
    assert request.config.tool_choice.allowed == ("get_time",)

    hosted = build_request(lm, "hi", {"tools": [{"type": "web_search_preview"}], "tool_choice": {"type": "web_search_preview"}})
    assert hosted.config.tool_choice.allowed == ("web_search_preview",)


def test_lm_responses_explicit_reasoning_wins_over_constructor_effort():
    response = _responses_text_response()

    with mock.patch("litellm.responses", autospec=True, return_value=response) as responses:
        lm = dspy.LM("openai/dspy-test-model", engine="litellm", model_type="responses", cache=False, reasoning_effort="low")
        lm("Say hi.", reasoning={"effort": "high"})

    sent = responses.call_args.kwargs
    assert sent["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_effort" not in sent


def test_lm_responses_does_not_validate_reasoning_temperature_client_side():
    response = _responses_text_response()

    with mock.patch("litellm.responses", autospec=True, return_value=response) as responses:
        lm = dspy.LM("openai/gpt-5-nano", engine="litellm", model_type="responses", cache=False, max_tokens=16000)
        lm("Say hi.", temperature=0.7, reasoning_effort="low")

    sent = responses.call_args.kwargs
    assert sent["temperature"] == 0.7
    assert sent["reasoning"] == {"effort": "low", "summary": "auto"}
