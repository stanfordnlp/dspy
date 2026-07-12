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
        with (
            mock.patch.object(cache, "cache_key", wraps=cache.cache_key) as cache_key_spy,
            mock.patch.object(cache, "get", wraps=cache.get) as cache_get_spy,
            mock.patch.object(cache, "put", wraps=cache.put) as cache_put_spy,
        ):

            def fake_completion(*, cache, num_retries, retry_strategy, **request):
                return ModelResponse(
                    choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    model="dummy",
                )

            monkeypatch.setattr(litellm, "completion", fake_completion)

            lm = dspy.LM("dummy", model_type="chat")
            lm(messages=[{"role": "user", "content": "Hello"}])

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

    lm = dspy.LM(model="openai/dspy-test-model", model_type="chat", temperature=0)
    with pytest.warns(UserWarning, match="rollout_id has no effect"):
        lm("Query", rollout_id=1)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm("Query", rollout_id=2)
        assert len(record) == 0


def test_rollout_id_with_default_temperature_does_not_warn(monkeypatch):
    def fake_completion(*, cache, num_retries, retry_strategy, **request):
        return ModelResponse(
            choices=[Choices(message=Message(role="assistant", content="Hi!"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            model="openai/gpt-5-nano",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        lm = dspy.LM(model="openai/gpt-5-nano", model_type="chat", rollout_id=1)
        lm("Query")
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


def test_lm_wraps_litellm_errors_with_metadata():
    lm = dspy.LM("openai/gpt-4o-mini")
    response = mock.Mock()
    response.status_code = 429
    response.headers = {"x-request-id": "req-123", "retry-after": "2.5"}

    error = litellm.RateLimitError(message="too many requests", llm_provider="openai", model="gpt-4o", response=response)
    wrapped = lm._wrap_litellm_exception(error)

    assert isinstance(wrapped, dspy.LMRateLimitError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"
    assert wrapped.status == 429
    assert wrapped.request_id == "req-123"
    assert wrapped.retry_after == 2.5


def test_lm_wraps_litellm_context_window_error():
    lm = dspy.LM("openai/gpt-4o-mini")
    error = litellm.ContextWindowExceededError(message="too long", llm_provider="openai", model="gpt-4o")
    wrapped = lm._wrap_litellm_exception(error)

    assert isinstance(wrapped, dspy.ContextWindowExceededError)
    assert isinstance(wrapped, dspy.LMError)
    assert wrapped.model == "gpt-4o"
    assert wrapped.provider == "openai"


def test_lm_wraps_unknown_boundary_error_as_unexpected_error():
    lm = dspy.LM("openai/gpt-4o-mini")
    wrapped = lm._wrap_litellm_exception(RuntimeError("local boundary failure"))

    assert isinstance(wrapped, dspy.LMUnexpectedError)
    assert wrapped.code == "unexpected"
    assert wrapped.model == "openai/gpt-4o-mini"


def test_lm_preserves_existing_lm_error_without_self_cause():
    error = dspy.LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    with mock.patch("dspy.clients.lm.litellm_completion", side_effect=error):
        with pytest.raises(dspy.LMRateLimitError) as exc_info:
            lm("question")

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_lm_preserves_existing_lm_error_without_self_cause_async():
    error = dspy.LMRateLimitError("rate limited", model="openai/gpt-4o-mini")
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    with mock.patch("dspy.clients.lm.alitellm_completion", side_effect=error):
        with pytest.raises(dspy.LMRateLimitError) as exc_info:
            await lm.acall("question")

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


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


def test_base_lm_forward_contract_defaults_to_legacy():
    class CustomLM(dspy.BaseLM):
        pass

    lm = CustomLM("custom-model")

    assert lm._get_forward_contract() == "legacy"
    assert not lm._declares_forward_contract()


def test_base_lm_forward_contract_accepts_explicit_values():
    class LegacyLM(dspy.BaseLM):
        forward_contract = "legacy"

    class TypedLM(dspy.BaseLM):
        forward_contract = "typed_lm"

    assert LegacyLM("custom-model")._get_forward_contract() == "legacy"
    assert LegacyLM("custom-model")._declares_forward_contract()
    assert TypedLM("custom-model")._get_forward_contract() == "typed_lm"
    assert TypedLM("custom-model")._declares_forward_contract()


def test_base_lm_forward_contract_rejects_unknown_values():
    class CustomLM(dspy.BaseLM):
        forward_contract = "normalized"

    with pytest.raises(ValueError, match="forward_contract must be 'legacy' or 'typed_lm'"):
        CustomLM("custom-model")._get_forward_contract()


def test_base_lm_validates_typed_lm_response():
    lm = dspy.BaseLM("custom-model")
    response = dspy.LMResponse.from_text("ok", model="custom-model")

    assert lm._validate_typed_lm_response(response) is response

    with pytest.raises(TypeError, match=r"requires forward\(request\).*dspy.LMResponse"):
        lm._validate_typed_lm_response(["ok"])


def test_base_lm_warns_when_inherited_legacy_forward_returns_lm_response():
    class CustomLM(dspy.BaseLM):
        pass

    lm = CustomLM("custom-model")
    response = dspy.LMResponse.from_text("ok", model="custom-model")

    with pytest.warns(DeprecationWarning, match="default legacy forward_contract"):
        assert lm._validate_legacy_lm_response(response) is response

    assert lm._validate_legacy_lm_response(["ok"]) is None


def test_base_lm_errors_when_explicit_legacy_forward_returns_lm_response():
    class CustomLM(dspy.BaseLM):
        forward_contract = "legacy"

    lm = CustomLM("custom-model")
    response = dspy.LMResponse.from_text("ok", model="custom-model")

    with pytest.raises(TypeError, match=r"forward_contract='legacy'.*got dspy.LMResponse"):
        lm._validate_legacy_lm_response(response)


def test_base_lm_inherited_legacy_forward_returning_lm_response_errors_on_direct_call():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return dspy.LMResponse.from_text(
                "ok",
                model="custom-model",
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            )

    lm = CustomLM("custom-model")

    with pytest.raises(TypeError, match="legacy direct path"):
        lm("Query")

    assert len(lm.history) == 0


# BaseLM direct-call compatibility tests.
#
# These cover the staged typed LM migration: legacy calls still return lists by default, explicit LMRequest calls and
# experimental direct calls return LMResponse, and typed-LM subclasses receive normalized LMRequest objects.


def test_base_lm_default_call_keeps_legacy_outputs():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            assert prompt == "Query"
            assert messages is None
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                model="custom-model",
            )

    assert CustomLM("custom-model")("Query") == ["Hi!"]


def test_base_lm_experimental_call_returns_lm_response_through_legacy_bridge():
    class CustomLM(dspy.BaseLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = None

        def forward(self, prompt=None, messages=None, **kwargs):
            self.seen = {"prompt": prompt, "messages": messages, "kwargs": kwargs}
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi!"), finish_reason="stop")],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                model="custom-model",
            )

    lm = CustomLM("custom-model", temperature=0.2)
    with dspy.context(experimental=True):
        response = lm("Query", rollout_id=7)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Hi!"
    assert response.output.finish_reason == "stop"
    assert lm.seen["prompt"] == "Query"
    assert lm.seen["messages"] is None
    assert lm.seen["kwargs"]["temperature"] == 0.2
    assert lm.seen["kwargs"]["cache"] is True
    assert lm.seen["kwargs"]["rollout_id"] == 7


def test_base_lm_explicit_lm_request_returns_lm_response_without_experimental():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                usage={},
                model="custom-model",
            )

    request = dspy.LMRequest.from_call(model="custom-model", prompt="Query")
    response = CustomLM("custom-model")(request)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Hi!"


def test_base_lm_legacy_bridge_records_typed_history_and_usage_once():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                model="custom-model",
            )

    lm = CustomLM("custom-model")
    request = dspy.LMRequest.from_call(model="custom-model", prompt="Query")

    with track_usage() as usage_tracker:
        response = lm(request)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Hi!"
    assert len(lm.history) == 1
    assert lm.history[0].request == request
    assert lm.history[0].response == response
    total_usage = usage_tracker.get_total_tokens()["custom-model"]
    assert total_usage["prompt_tokens"] == 1
    assert total_usage["completion_tokens"] == 2
    assert total_usage["total_tokens"] == 3


def test_base_lm_typed_forward_contract_uses_lm_request():
    class CustomLM(dspy.BaseLM):
        forward_contract = "typed_lm"

        def forward(self, request):
            assert isinstance(request, dspy.LMRequest)
            return dspy.LMResponse.from_text(f"model={request.model}; text={request.messages[0].text}")

    lm = CustomLM("custom-model")

    assert lm("Query") == ["model=custom-model; text=Query"]
    with dspy.context(experimental=True):
        response = lm("Query")
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "model=custom-model; text=Query"


def test_base_lm_typed_forward_contract_rejects_non_lm_response_at_call_time():
    class CustomLM(dspy.BaseLM):
        forward_contract = "typed_lm"

        def forward(self, request):
            return ["not typed"]

    with pytest.raises(TypeError, match="forward_contract='typed_lm'"):
        CustomLM("custom-model")("Query")


def test_base_lm_request_call_rejects_mixed_inputs():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            raise AssertionError("forward should not be called")

    request = dspy.LMRequest.from_call(model="custom-model", prompt="Query")
    with pytest.raises(ValueError, match="Pass either an LMRequest or direct-call inputs"):
        CustomLM("custom-model")(request, "extra")


def _model_response(text: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(role="assistant", content=text))],
        usage={},
        model="custom-model",
    )


class _TypedContractLM(dspy.BaseLM):
    """Test double that records normalized requests received through the typed LM contract."""

    forward_contract = "typed_lm"

    def __init__(self, *args, outputs: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = outputs
        self.requests = []

    def forward(self, request):
        assert isinstance(request, dspy.LMRequest)
        self.requests.append(request)
        return dspy.LMResponse.from_text(self.outputs[len(self.requests) - 1], model=request.model)


def _direct_lm_case(lm_kind: str, outputs: list[str]):
    """Return a direct-call test double and helpers for inspecting normalized messages."""
    if lm_kind == "current_lm":
        patcher = mock.patch(
            "dspy.clients.lm.litellm_completion",
            side_effect=[_model_response(output) for output in outputs],
        )
        completion = patcher.start()
        lm = dspy.LM("custom-model", cache=False)

        def get_messages(index: int) -> list[dict[str, object]]:
            return completion.call_args_list[index].kwargs["request"]["messages"]

        def get_request(index: int):
            return None

        return lm, get_messages, get_request, patcher

    if lm_kind == "typed_lm":
        lm = _TypedContractLM("custom-model", outputs=outputs)

        def get_messages(index: int) -> list[dict[str, object]]:
            from dspy.clients.openai_format import to_openai_chat_request

            return to_openai_chat_request(lm.requests[index])["messages"]

        def get_request(index: int):
            return lm.requests[index]

        return lm, get_messages, get_request, None

    raise ValueError(f"Unknown lm_kind: {lm_kind}")


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_support_system_user_and_assistant_turns(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(lm_kind, ["Five-word answer."])
    try:
        with dspy.context(experimental=True):
            response = lm(
                dspy.System("Be concise."),
                dspy.User("What is DSPy?"),
                dspy.Assistant("DSPy is a framework for programming LM pipelines."),
                dspy.User("Say that in five words."),
                temperature=0.2,
            )
    finally:
        if patcher is not None:
            patcher.stop()

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Five-word answer."
    assert get_messages(0) == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is DSPy?"},
        {"role": "assistant", "content": "DSPy is a framework for programming LM pipelines."},
        {"role": "user", "content": "Say that in five words."},
    ]
    if lm_kind == "typed_lm":
        assert get_request(0).config.temperature == 0.2


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_support_tool_call_transcripts(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(lm_kind, ["It is 22 C in Paris."])
    try:
        with dspy.context(experimental=True):
            response = lm(
                dspy.User("What is the weather in Paris?"),
                dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
                dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
                dspy.User("Summarize the result."),
            )
    finally:
        if patcher is not None:
            patcher.stop()

    assert response.text == "It is 22 C in Paris."
    assert get_messages(0) == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
                    "id": "call_1",
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": "22 C"}', "tool_call_id": "call_1", "name": "get_weather"},
        {"role": "user", "content": "Summarize the result."},
    ]
    if lm_kind == "typed_lm":
        assert isinstance(get_request(0), dspy.LMRequest)


@pytest.mark.parametrize("lm_kind", ["current_lm", "typed_lm"])
def test_base_lm_experimental_direct_messages_can_reuse_lm_response_as_assistant_turn(lm_kind):
    lm, get_messages, get_request, patcher = _direct_lm_case(
        lm_kind,
        ["DSPy programs LM pipelines.", "DSPy programs pipelines."],
    )
    try:
        with dspy.context(experimental=True):
            first = lm("Explain DSPy in one sentence.")
            follow_up = lm(
                dspy.User("Explain DSPy in one sentence."),
                first,
                dspy.User("Now make it even shorter."),
            )
    finally:
        if patcher is not None:
            patcher.stop()

    assert first.text == "DSPy programs LM pipelines."
    assert follow_up.text == "DSPy programs pipelines."
    assert get_messages(0) == [{"role": "user", "content": "Explain DSPy in one sentence."}]
    assert get_messages(1) == [
        {"role": "user", "content": "Explain DSPy in one sentence."},
        {"role": "assistant", "content": "DSPy programs LM pipelines."},
        {"role": "user", "content": "Now make it even shorter."},
    ]
    if lm_kind == "typed_lm":
        assert isinstance(get_request(1), dspy.LMRequest)


@pytest.mark.asyncio
async def test_base_lm_async_explicit_lm_request_returns_lm_response():
    class CustomLM(dspy.BaseLM):
        async def aforward(self, prompt=None, messages=None, **kwargs):
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi async!"))],
                usage={},
                model="custom-model",
            )

    request = dspy.LMRequest.from_call(model="custom-model", prompt="Query")
    response = await CustomLM("custom-model").acall(request)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Hi async!"


def test_base_lm_tracks_usage_for_custom_subclasses():
    class CustomLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="Hi!"))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                model="custom-model",
            )

    lm = CustomLM(model="custom-model")

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

    dspy.LM.load_state(dspy.LM("openai/gpt-4o-mini").dump_state(), allow_custom_lm_class=True)

    assert calls == [True]


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
        with pytest.raises(dspy.LMRateLimitError):
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
    with mock.patch("dspy.clients.lm.litellm_responses_completion", return_value={"choices": []}) as mock_completion:
        lm = dspy.LM(
            "openai/gpt-4o-mini",
            cache=False,
            model_type="responses",
            use_developer_role=True,
        )
        lm.forward(messages=[{"role": "system", "content": "hi"}])
        assert mock_completion.call_args.kwargs["request"]["messages"][0]["role"] == "developer"


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


def test_reasoning_effort_responses_api():
    """Test that reasoning_effort gets normalized to reasoning format for Responses API."""
    with mock.patch("litellm.responses") as mock_responses:
        # OpenAI model with Responses API - should normalize
        lm = dspy.LM(
            model="openai/gpt-5", model_type="responses", reasoning_effort="low", max_tokens=16000, temperature=1.0
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
                model="anthropic/claude-3-7-sonnet-20250219",
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


def test_responses_api_converts_images_correctly():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    # Test with base64 image
    request_with_base64_image = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_base64_image)

    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"

    content = result["input"][0]["content"]
    assert len(content) == 2

    # First item should be text converted to input_text format
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "What's in this image?"

    # Second item should be converted to input_image format
    assert content[1]["type"] == "input_image"
    assert (
        content[1]["image_url"]
        == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    # Test with URL image
    request_with_url_image = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_url_image)

    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"] == "https://example.com/image.jpg"


def test_responses_api_converts_files_correctly():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    # Test with file data (base64 encoded)
    request_with_file = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this file"},
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                            "filename": "test.txt",
                        },
                    },
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_file)

    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"

    content = result["input"][0]["content"]
    assert len(content) == 2

    # First item should be text converted to input_text format
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "Analyze this file"

    # Second item should be converted to input_file format
    assert content[1]["type"] == "input_file"
    assert content[1]["file_data"] == "data:text/plain;base64,SGVsbG8gV29ybGQ="
    assert content[1]["filename"] == "test.txt"

    # Test with file_id
    request_with_file_id = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_id": "file-abc123",
                            "filename": "document.pdf",
                        },
                    }
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_file_id)

    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_file"
    assert content[0]["file_id"] == "file-abc123"
    assert content[0]["filename"] == "document.pdf"

    # Test with all file fields
    request_with_all_fields = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                            "file_id": "file-xyz789",
                            "filename": "report.pdf",
                        },
                    }
                ],
            }
        ],
    }

    result = _convert_chat_request_to_responses_request(request_with_all_fields)

    content = result["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == "data:application/pdf;base64,JVBERi0xLjQ="
    assert content[0]["file_id"] == "file-xyz789"
    assert content[0]["filename"] == "report.pdf"


def test_responses_api_preserves_multi_message_structure():
    from dspy.clients.lm import _convert_chat_request_to_responses_request

    request = {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    }

    result = _convert_chat_request_to_responses_request(request)

    assert "input" in result
    assert len(result["input"]) == 4

    assert result["input"][0]["role"] == "system"
    assert result["input"][0]["content"] == [{"type": "input_text", "text": "You are a helpful assistant."}]

    assert result["input"][1]["role"] == "user"
    assert result["input"][1]["content"] == [{"type": "input_text", "text": "What is 2+2?"}]

    assert result["input"][2]["role"] == "assistant"
    assert result["input"][2]["content"] == [{"type": "input_text", "text": "4"}]

    assert result["input"][3]["role"] == "user"
    assert result["input"][3]["content"] == [{"type": "input_text", "text": "And 3+3?"}]


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
            model="openai/gpt-5-mini",
            model_type="responses",
            cache=False,
            temperature=1.0,
            max_tokens=16000,
        )

        # Test with messages containing an image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ]

        lm_result = lm(messages=messages)

        assert lm_result == [{"text": "This is a test answer with image input."}]

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
        model="openai/gpt-5-mini",
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
        "schema": TestModel.model_json_schema(),
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
            model="openai/gpt-5-mini",
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
            model="openai/gpt-5-mini",
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


@pytest.mark.asyncio
async def test_streaming_passes_headers_correctly():
    from dspy.clients.lm import _get_stream_completion_fn

    custom_headers = {"Authorization": "Bearer my-custom-token"}
    request = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "test"}],
    }

    mock_stream = mock.AsyncMock()
    mock_stream.send = mock.AsyncMock()

    async def empty_async_generator():
        return
        yield  # Make it a generator

    with mock.patch("dspy.settings") as mock_settings:
        mock_settings.send_stream = mock_stream
        mock_settings.caller_predict = None
        mock_settings.track_usage = False

        with mock.patch("litellm.acompletion") as mock_acompletion:
            mock_acompletion.return_value = empty_async_generator()

            stream_fn = _get_stream_completion_fn(request, {}, sync=False, headers=custom_headers)
            assert stream_fn is not None

            with mock.patch("litellm.stream_chunk_builder", return_value={}):
                await stream_fn()

            # Verify headers were passed to litellm.acompletion
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Bearer my-custom-token"
