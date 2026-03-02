"""Tests for Adapter retry logic on parse/validation errors."""

from unittest import mock

import pytest
from litellm.utils import Choices, Message, ModelResponse, Usage

import dspy
from dspy.utils.exceptions import AdapterParseError


def _model_response(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content))],
        model="openai/gpt-4o-mini",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


VALID_CHAT_RESPONSE = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
INVALID_RESPONSE = "some garbage that cannot be parsed"


class TestAdapterRetry:
    """Tests for retry-on-parse-error in Adapter.__call__."""

    def test_retry_succeeds_on_second_attempt(self):
        """First LM call returns unparseable output; retry returns valid output."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=3, use_json_adapter_fallback=False)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                _model_response(INVALID_RESPONSE),
                _model_response(VALID_CHAT_RESPONSE),
            ]
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        assert result == [{"answer": "Paris"}]
        assert mock_completion.call_count == 2

    def test_retry_exhausted_raises(self):
        """All retries fail, error is raised."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=2, use_json_adapter_fallback=False)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _model_response(INVALID_RESPONSE)
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)

            with pytest.raises(AdapterParseError):
                adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        # 1 initial + 2 retries = 3 calls
        assert mock_completion.call_count == 3

    def test_max_retries_zero_disables_retry(self):
        """With max_retries=0, no retry is attempted."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=0, use_json_adapter_fallback=False)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _model_response(INVALID_RESPONSE)
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)

            with pytest.raises(AdapterParseError):
                adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        assert mock_completion.call_count == 1

    def test_retry_appends_error_feedback_to_messages(self):
        """On retry, the failed response and error are appended to messages."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=1, use_json_adapter_fallback=False)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                _model_response(INVALID_RESPONSE),
                _model_response(VALID_CHAT_RESPONSE),
            ]
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        # Check the second call's messages include retry feedback
        second_call_messages = mock_completion.call_args_list[1].kwargs.get(
            "messages", mock_completion.call_args_list[1][1].get("messages", [])
        )
        # Should have: system, user (original), assistant (failed), user (error feedback)
        roles = [m["role"] for m in second_call_messages]
        assert roles[-2] == "assistant"
        assert roles[-1] == "user"
        assert INVALID_RESPONSE in second_call_messages[-2]["content"]
        assert "could not be parsed" in second_call_messages[-1]["content"]

    def test_non_parse_errors_propagate_immediately(self):
        """Errors other than AdapterParseError/ValidationError are not retried."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=3, use_json_adapter_fallback=False)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = RuntimeError("network error")
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)

            with pytest.raises(RuntimeError, match="network error"):
                adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        assert mock_completion.call_count == 1

    def test_retry_with_fallback_still_works(self):
        """Retries exhaust, then ChatAdapter falls back to JSONAdapter."""
        signature = dspy.make_signature("question->answer")
        # JSON-formatted response: ChatAdapter can't parse it, but JSONAdapter can
        json_response = '{"answer": "Paris"}'
        adapter = dspy.ChatAdapter(max_retries=1, use_json_adapter_fallback=True)

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _model_response(json_response)
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

        assert result == [{"answer": "Paris"}]
        # ChatAdapter: 1 initial + 1 retry = 2 calls
        # JSONAdapter fallback: 1 initial (succeeds) = 1 call
        # Total = 3
        assert mock_completion.call_count >= 3

    @pytest.mark.asyncio
    async def test_async_retry_succeeds_on_second_attempt(self):
        """Async retry: first call fails, second succeeds."""
        signature = dspy.make_signature("question->answer")
        adapter = dspy.ChatAdapter(max_retries=3, use_json_adapter_fallback=False)

        with mock.patch("litellm.acompletion") as mock_acompletion:
            mock_acompletion.side_effect = [
                _model_response(INVALID_RESPONSE),
                _model_response(VALID_CHAT_RESPONSE),
            ]
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})

        assert result == [{"answer": "Paris"}]
        assert mock_acompletion.call_count == 2

    def test_default_max_retries_is_three(self):
        """Default ChatAdapter has max_retries=3."""
        adapter = dspy.ChatAdapter()
        assert adapter.max_retries == 3

    def test_pydantic_validation_error_triggers_retry(self):
        """Pydantic validation errors (e.g., constraint violations) trigger retry."""
        import pydantic

        class StrictAnswer(pydantic.BaseModel):
            items: list[str] = pydantic.Field(max_length=2)

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            answer: StrictAnswer = dspy.OutputField()

        adapter = dspy.ChatAdapter(max_retries=1, use_json_adapter_fallback=False)

        # First response violates max_length=2, second is valid
        bad_json = '[[ ## answer ## ]]\n{"items": ["a", "b", "c"]}\n\n[[ ## completed ## ]]'
        good_json = '[[ ## answer ## ]]\n{"items": ["a", "b"]}\n\n[[ ## completed ## ]]'

        with mock.patch("litellm.completion") as mock_completion:
            mock_completion.side_effect = [
                _model_response(bad_json),
                _model_response(good_json),
            ]
            lm = dspy.LM("openai/gpt-4o-mini", cache=False)
            result = adapter(lm, {}, MySignature, [], {"question": "Give me items"})

        assert result[0]["answer"].items == ["a", "b"]
        assert mock_completion.call_count == 2
