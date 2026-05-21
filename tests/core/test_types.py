import pytest
from pydantic import ValidationError

import dspy
from dspy.core.types import (
    LMConfig,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMRequest,
    LMThinkingPart,
    LMToolCallPart,
    LMToolResultPart,
    _document_dict_to_part,
)


def test_lm_config_normalizes_grouped_config_values():
    config = LMConfig.from_kwargs(
        reasoning_effort="low",
        tool_choice={"allowed": ["search"]},
        parallel_tool_calls=False,
        cache={"enabled": True, "rollout_id": 2},
        prompt_cache={"enabled": True, "key": "prompt-v1"},
    )

    assert config.stop is None
    assert config.reasoning.effort == "low"
    assert config.tool_choice.mode is None
    assert config.tool_choice.allowed == ["search"]
    assert config.tool_choice.parallel is False
    assert config.cache.enabled is True
    assert config.cache.rollout_id == 2
    assert config.prompt_cache.enabled is True
    assert config.prompt_cache.key == "prompt-v1"

    parallel_only = LMConfig.from_kwargs(parallel_tool_calls=False)
    assert parallel_only.tool_choice.model_dump(exclude_none=True) == {"parallel": False}


def test_none_companion_config_keys_do_not_create_nested_configs():
    config = LMConfig.from_kwargs(
        reasoning_effort=None,
        parallel_tool_calls=None,
        rollout_id=None,
        prompt_cache_key=None,
    )

    assert config.reasoning is None
    assert config.tool_choice is None
    assert config.cache is None
    assert config.prompt_cache is None


def test_public_type_constructors_are_exported():
    assert dspy.LMConfig is LMConfig
    assert dspy.System("system").role == "system"
    assert dspy.User("hello").parts[0].text == "hello"


def test_openai_assistant_tool_calls_without_content_normalize_to_parts():
    message = LMMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "DSPy"}'},
            }
        ],
    )

    assert message.parts == [LMToolCallPart(id="call_1", name="search", args={"query": "DSPy"})]


def test_openai_tool_result_message_normalizes_tool_call_id():
    message = LMMessage(role="tool", content="result", tool_call_id="call_1", name="search")

    assert message.parts == [
        LMToolResultPart(call_id="call_1", name="search", content=["result"]),
    ]


def test_lm_request_from_call_does_not_coerce_dspy_specific_objects():
    class Image:
        url = "https://example.com/image.png"

    with pytest.raises(TypeError, match="Cannot convert"):
        LMRequest.from_call(model="openai/gpt-5-nano", items=(Image(),))


def test_lm_request_config_overrides_preserve_nested_config():
    request = LMRequest(
        model="openai/gpt-5-nano",
        messages=[LMMessage(role="user", parts=["hello"])],
        config=LMConfig.from_kwargs(cache=False, prompt_cache=True, tool_choice="auto"),
    )

    updated = request.with_config_overrides(
        rollout_id=3,
        prompt_cache_key="prompt-v2",
        parallel_tool_calls=False,
    )

    assert updated.config.cache.enabled is False
    assert updated.config.cache.rollout_id == 3
    assert updated.config.prompt_cache.enabled is True
    assert updated.config.prompt_cache.key == "prompt-v2"
    assert updated.config.tool_choice.mode == "auto"
    assert updated.config.tool_choice.parallel is False


def test_source_parts_require_exactly_one_non_empty_source():
    with pytest.raises(ValidationError, match="requires exactly one"):
        LMImagePart(data="abc", url="https://example.com/image.png")

    with pytest.raises(ValidationError, match="must be non-empty"):
        LMImagePart(url="")


def test_document_string_source_preserves_url():
    part = _document_dict_to_part({"type": "document", "source": "https://example.com/file.pdf"})

    assert part.url == "https://example.com/file.pdf"
    assert part.data is None
    assert part.media_type == "application/pdf"


def test_thinking_part_value_stays_core_type():
    output = LMOutput(parts=[LMThinkingPart(text="reasoning")])

    assert output.to_value() == [LMThinkingPart(text="reasoning")]
