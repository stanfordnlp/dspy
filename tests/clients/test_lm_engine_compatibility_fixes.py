"""Regression coverage for the first coordinated engine matrix failures."""

import pytest

import dspy
from dspy.clients.legacy_requests import chat_to_responses
from dspy.utils.dummies import DummyLM


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_dummy_forward_override_is_not_bypassed(asynchronous):
    error = dspy.LMRateLimitError("original error")

    class DownLM(DummyLM):
        def forward(self, *args, **kwargs):
            raise error

    lm = DownLM([])
    with pytest.raises(dspy.LMRateLimitError) as caught:
        if asynchronous:
            await lm.acall("hello")
        else:
            lm("hello")
    assert caught.value is error
    assert lm.history == []


@pytest.mark.asyncio
async def test_dummy_override_can_delegate_to_super_without_recursion():
    class RecordingLM(DummyLM):
        calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return super().forward(*args, **kwargs)

    lm = RecordingLM([{"answer": "first"}, {"answer": "second"}])
    assert lm("hello") == ["[[ ## answer ## ]]\nfirst"]
    assert await lm.acall("hello") == ["[[ ## answer ## ]]\nsecond"]
    assert lm.calls == 2
    assert len(lm.history) == 2


@pytest.mark.asyncio
async def test_dummy_instance_forward_override_is_respected():
    lm = DummyLM([])
    error = dspy.LMTransportError("offline")

    def fail(**kwargs):
        raise error

    lm.forward = fail
    with pytest.raises(dspy.LMTransportError) as caught:
        await lm.acall("hello")
    assert caught.value is error


@pytest.mark.asyncio
async def test_dummy_accepts_opaque_blocks_and_positional_message_lists():
    lm = DummyLM([{"answer": "first"}, {"answer": "second"}], reasoning=True)
    messages = [{"role": "user", "content": [
        {"type": "file", "file": {"file_id": "file-1", "filename": "report.txt"}},
        {"type": "custom_event", "payload": {"value": 1}},
    ]}]
    first = lm(messages)
    second = await lm.acall(messages=messages)
    assert first[0]["text"] == "[[ ## answer ## ]]\nfirst"
    assert second[0]["text"] == "[[ ## answer ## ]]\nsecond"
    assert first[0]["reasoning_content"] == "Some reasoning"
    assert messages[0]["content"][0]["file"]["filename"] == "report.txt"
    assert lm.history[-1]["messages"] == messages


def test_dictionary_responses_preserve_rich_outputs():
    class DictionaryLM(dspy.BaseLM):
        def forward(self, **kwargs):
            return {"model": "custom", "usage": {"total_tokens": 2}, "choices": [{
                "message": {"content": "answer", "reasoning_content": "reasoning",
                            "tool_calls": [{"id": "call-1", "type": "function", "function": {
                                "name": "lookup", "arguments": "{}"}}],
                            "provider_specific_fields": {"citations": [[{"cited_text": "source"}]]}},
                "logprobs": {"content": []}, "finish_reason": "stop",
            }]}

    lm = DictionaryLM("custom")
    result = lm("hello", logprobs=True)[0]
    assert result["text"] == "answer"
    assert result["reasoning_content"] == "reasoning"
    assert result["tool_calls"][0]["function"]["name"] == "lookup"
    assert result["citations"] == [{"cited_text": "source"}]
    assert result["logprobs"] == {"content": []}
    assert lm.history[0]["usage"] == {"total_tokens": 2}


def test_absent_tool_description_is_omitted_without_dropping_other_fields():
    original = {"model": "openai/example", "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {
                    "name": "lookup", "description": None, "strict": False, "vendor_field": None,
                    "parameters": {"type": "object"},
                }}]}
    tool = chat_to_responses(original)["tools"][0]
    assert "description" not in tool
    assert tool["strict"] is False
    assert "vendor_field" in tool and tool["vendor_field"] is None
    assert "description" in original["tools"][0]["function"]
