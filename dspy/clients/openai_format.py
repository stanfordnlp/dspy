"""Translate between DSPy's LM types and OpenAI-shaped JSON.

This module does not call any provider. It only maps data shapes:

```text
LMRequest -> OpenAI Chat / Responses / text-completion kwargs
provider response -> LMResponse
```

The concrete OpenAI backends import these functions, then add transport,
streaming, and caching around them. Read the file in
this order when learning it:

1. `to_openai_chat_request()` maps chat-completion requests.
2. `to_openai_responses_request()` maps Responses API requests.
3. `to_openai_text_request()` maps legacy text-completion requests.
4. `completion_to_lm_response()` and `responses_to_lm_response()` map outputs.
5. The final utility section handles media sources, data URIs, and object/dict
   access.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Any

import pydantic

from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMConfig,
    LMDocumentPart,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
    LMVideoPart,
)

__all__ = [
    "to_openai_chat_request",
    "to_openai_responses_request",
    "to_openai_text_request",
    "completion_to_lm_response",
    "responses_to_lm_response",
    "provider_tool_call_to_part",
    "responses_function_call_to_part",
    "usage_from_response",
]


# ---------------------------------------------------------------------------
# DSPy request -> OpenAI Chat Completions
#
# Chat Completions expects:
#   {"model": ..., "messages": [...], "tools": [...], ...config}
# Each DSPy message stays one OpenAI message, except assistant tool calls are
# split into OpenAI's top-level `tool_calls` field on the assistant message.
# ---------------------------------------------------------------------------


def to_openai_chat_request(request: LMRequest) -> dict[str, Any]:
    """Convert a normalized DSPy request into Chat Completions kwargs."""
    data = {"model": request.model, "messages": [message_to_openai_chat(message) for message in request.messages]}
    data.update(common_config_kwargs(request.config, model=request.model, endpoint="chat"))
    if request.config.tool_choice is not None:
        data.update(tool_choice_to_openai(request.config.tool_choice))
    if request.tools:
        data["tools"] = [tool_to_openai(tool) for tool in request.tools]
    return data


def message_to_openai_chat(message: LMMessage) -> dict[str, Any]:
    """Convert one DSPy message into one Chat Completions message."""
    output: dict[str, Any] = {"role": message.role}
    if message.name is not None:
        output["name"] = message.name

    if message.role == "assistant":
        tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
        content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
        output["content"] = None if tool_calls and not content_parts else parts_to_openai_content(content_parts)
        if tool_calls:
            output["tool_calls"] = [assistant_tool_call_to_openai(part) for part in tool_calls]
        return output

    if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
        result = message.parts[0]
        output.update(tool_result_to_openai(result))
        if result.call_id is not None:
            output["tool_call_id"] = result.call_id
        if result.name is not None:
            output["name"] = result.name
        return output

    output["content"] = parts_to_openai_content(message.parts)
    return output


# ---------------------------------------------------------------------------
# DSPy request -> OpenAI Responses
#
# Responses expects a flat `input` list, not a chat `messages` list. A single
# DSPy assistant message can therefore become two kinds of input items: one
# message item for content and one function_call item for each tool call.
# ---------------------------------------------------------------------------


def to_openai_responses_request(request: LMRequest) -> dict[str, Any]:
    """Convert a normalized DSPy request into Responses API kwargs."""
    config = request.config
    data: dict[str, Any] = {
        "model": request.model,
        "input": [item for message in request.messages for item in message_to_responses_input_items(message)],
    }
    data.update(responses_config_kwargs(config, model=request.model))
    if config.tool_choice is not None:
        data.update(tool_choice_to_openai(config.tool_choice))
    if request.tools:
        data["tools"] = [tool_to_openai(tool) for tool in request.tools]
    return data


def message_to_responses_input_items(message: LMMessage) -> list[dict[str, Any]]:
    """Convert one DSPy message into one or more Responses input items."""
    if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
        result = message.parts[0]
        item = {"type": "function_call_output", "output": responses_tool_output_text(tool_result_to_openai(result)["content"])}
        if result.call_id is not None:
            item["call_id"] = result.call_id
        return [item]

    tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
    content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
    content = parts_to_responses_content(content_parts)
    items: list[dict[str, Any]] = []

    if content or message.role != "assistant" or not tool_calls:
        item: dict[str, Any] = {"role": message.role, "content": content}
        if message.name is not None:
            item["name"] = message.name
        items.append(item)

    if message.role == "assistant":
        items.extend(tool_call_to_responses_input(tool_call) for tool_call in tool_calls)
    return items


def parts_to_responses_content(parts: list[Any]) -> list[dict[str, Any]]:
    blocks = parts_to_openai_content(parts)
    if isinstance(blocks, str):
        return [{"type": "input_text", "text": blocks}]
    return [content_block_to_responses(block) for block in blocks]


def tool_call_to_responses_input(tool_call_part: LMToolCallPart) -> dict[str, Any]:
    tool_call = assistant_tool_call_to_openai(tool_call_part)
    function = tool_call.get("function", {})
    item = {"type": "function_call", "name": function.get("name", ""), "arguments": function.get("arguments", "{}")}
    call_id = tool_call.get("id") or tool_call.get("call_id")
    if call_id is not None:
        item["call_id"] = call_id
    return item


def content_block_to_responses(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    if block_type == "text":
        return {"type": "input_text", "text": block.get("text", "")}
    if block_type == "image_url":
        image_url = block.get("image_url", {})
        out = {"type": "input_image", "image_url": image_url.get("url", "")}
        if image_url.get("detail") is not None:
            out["detail"] = image_url["detail"]
        return out
    if block_type == "input_audio":
        return block
    if block_type == "file":
        file = block.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return block


# ---------------------------------------------------------------------------
# DSPy request -> OpenAI text completions
#
# Text completions have no native message roles. We concatenate text-only
# messages with blank lines and append DSPy's historical response marker.
# ---------------------------------------------------------------------------


def to_openai_text_request(request: LMRequest) -> dict[str, Any]:
    """Convert a normalized DSPy request into text-completion kwargs."""
    data = {"model": request.model, "prompt": messages_to_text_prompt(request.messages)}
    data.update(text_config_kwargs(request.config))
    return data


def messages_to_text_prompt(messages: list[LMMessage]) -> str:
    """Flatten text-only messages into the prompt used by text completions."""
    chunks = []
    for message in messages:
        texts = []
        for part in message.parts:
            if not isinstance(part, LMTextPart):
                raise ValueError(f"OpenAI text completions only support text parts, but received {type(part).__name__}.")
            texts.append(part.text)
        chunks.append("".join(texts))
    return "\n\n".join(chunks + ["BEGIN RESPONSE:"])


# ---------------------------------------------------------------------------
# Shared request mappers
#
# The three OpenAI endpoint families use slightly different outer envelopes, but most
# leaf values are the same: text parts, media parts, tools, tool choices, and
# generation config. The helpers below keep those conversions in one place.
# ---------------------------------------------------------------------------


def parts_to_openai_content(parts: list[Any]) -> str | list[dict[str, Any]]:
    """Convert DSPy parts into OpenAI `content`.

    OpenAI accepts a bare string for the common single-text case. Mixed content
    becomes a list of content blocks.
    """
    if len(parts) == 1 and isinstance(parts[0], LMTextPart) and "legacy_content_block" not in parts[0].metadata:
        return parts[0].text
    blocks: list[dict[str, Any]] = []
    for part in parts:
        blocks.extend(part_to_openai_blocks(part))
    return blocks


def part_to_openai_blocks(part: Any) -> list[dict[str, Any]]:
    """Convert one DSPy part into one or more OpenAI content blocks."""
    legacy_block = getattr(part, "metadata", {}).get("legacy_content_block")
    if legacy_block is not None:
        return [dict(legacy_block)]
    if isinstance(part, LMTextPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, LMImagePart):
        return [image_to_openai(part)]
    if isinstance(part, LMDocumentPart):
        return document_to_openai_blocks(part)
    if isinstance(part, LMAudioPart):
        return [audio_to_openai(part)]
    if isinstance(part, LMVideoPart):
        return [video_to_openai(part)]
    if isinstance(part, LMBinaryPart):
        return [binary_to_openai(part)]
    if isinstance(part, LMThinkingPart):
        return [{"type": "text", "text": part.text}]
    if isinstance(part, LMCitationPart):
        citation = " ".join(value for value in (part.title, part.text, part.url) if value)
        return [{"type": "text", "text": citation}]
    if isinstance(part, LMToolResultPart):
        return part_to_openai_blocks(LMTextPart(text="".join(part_text(value) for value in part.content)))
    return [{"type": "text", "text": str(part)}]


def image_to_openai(image: LMImagePart) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": media_source(image)}
    if image.detail is not None:
        image_url["detail"] = image.detail
    return {"type": "image_url", "image_url": image_url}


def audio_to_openai(audio: LMAudioPart) -> dict[str, Any]:
    if audio.data is not None:
        data = audio.data
        media_type = audio.media_type
    elif audio.path is not None:
        data = read_path_base64(audio.path)
        media_type = media_type_for_path(audio.path, fallback=audio.media_type)
    else:
        raise ValueError("OpenAI-format audio input requires base64 `data` or local `path`.")
    return {"type": "input_audio", "input_audio": {"data": data, "format": media_format(media_type)}}


def document_to_openai_blocks(document: LMDocumentPart) -> list[dict[str, Any]]:
    block: dict[str, Any] = {"type": "document"}
    if document.source is not None:
        block["source"] = document.source
    else:
        block["source"] = media_source(document)
        block["media_type"] = document.media_type
    if document.citations:
        block["citations"] = document.citations
    if document.title is not None:
        block["title"] = document.title
    if document.context is not None:
        block["context"] = document.context
    return [block]


def video_to_openai(video: LMVideoPart) -> dict[str, Any]:
    filename = os.path.basename(video.path) if video.path is not None else None
    return binary_to_openai(
        LMBinaryPart(
            data=video.data,
            url=video.url,
            file_id=video.file_id,
            path=video.path,
            media_type=video.media_type,
            filename=filename,
        )
    )


def binary_to_openai(binary: LMBinaryPart) -> dict[str, Any]:
    file_data: dict[str, Any] = {}
    if binary.data is not None:
        file_data["file_data"] = data_uri(binary.media_type, binary.data)
    elif binary.path is not None:
        file_data["file_data"] = data_uri_from_path(binary.path, fallback_media_type=binary.media_type)
        file_data["filename"] = binary.filename or os.path.basename(binary.path)
    elif binary.url is not None:
        file_data["file_data"] = binary.url
    if binary.file_id is not None:
        file_data["file_id"] = binary.file_id
    if binary.filename is not None:
        file_data["filename"] = binary.filename
    return {"type": "file", "file": file_data}


def tool_to_openai(tool: LMToolSpec) -> dict[str, Any]:
    data = {"type": "function", "function": {"name": tool.name, "parameters": tool.parameters}}
    if tool.description is not None:
        data["function"]["description"] = tool.description
    data.update(tool.provider_data)
    return data


def tool_choice_to_openai(choice: LMToolChoice) -> dict[str, Any]:
    if choice.allowed:
        if len(choice.allowed) != 1 or choice.mode not in {"required", "auto"}:
            raise ValueError(
                "OpenAI-format tool_choice only supports constraining to a single allowed tool "
                "with mode 'required' or 'auto'."
            )
        data: dict[str, Any] = {"tool_choice": {"type": "function", "function": {"name": choice.allowed[0]}}}
    else:
        data = {"tool_choice": choice.mode}
    if choice.parallel is not None:
        data["parallel_tool_calls"] = choice.parallel
    return data


def assistant_tool_call_to_openai(call: LMToolCallPart) -> dict[str, Any]:
    data = {"type": "function", "function": {"name": call.name, "arguments": json.dumps(call.args)}}
    if call.id is not None:
        data["id"] = call.id
    data.update(call.provider_data)
    return data


def tool_result_to_openai(result: LMToolResultPart) -> dict[str, Any]:
    return {"content": parts_to_openai_content(result.content)}


def common_config_kwargs(config: LMConfig, *, model: str | None = None, endpoint: str = "chat") -> dict[str, Any]:
    """Convert shared DSPy config fields into Chat Completions kwargs."""
    data = dict(config.extensions)
    _validate_openai_reasoning_temperature(config, model=model, endpoint=endpoint)
    for key in ("temperature", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.max_tokens is not None:
        token_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
        data[token_key] = config.max_tokens
    if config.stop:
        data["stop"] = config.stop
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.n is not None:
        data["n"] = config.n
    if config.response_format is not None:
        data["response_format"] = config.response_format
    if config.reasoning is not None:
        data.update(reasoning_to_chat_kwargs(config.reasoning))
    if config.prompt_cache is not None:
        data.update(prompt_cache_to_kwargs(config.prompt_cache))
    return data


def responses_config_kwargs(config: LMConfig, *, model: str | None = None) -> dict[str, Any]:
    """Convert shared DSPy config fields into Responses API kwargs."""
    data = dict(config.extensions) if config.extensions else {}
    _validate_openai_reasoning_temperature(config, model=model, endpoint="responses")
    for key in ("temperature", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.max_tokens is not None:
        data["max_output_tokens"] = config.max_tokens
    if config.n is not None:
        data["n"] = config.n
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.stop:
        data["stop"] = config.stop
    if config.reasoning is not None:
        data.update(reasoning_to_responses_kwargs(config.reasoning))
    if config.prompt_cache is not None:
        data.update(prompt_cache_to_kwargs(config.prompt_cache))
    if config.response_format is not None:
        text = data.pop("text", {})
        data["text"] = {**text, "format": response_format_to_responses(config.response_format)}
    return data


def text_config_kwargs(config: LMConfig) -> dict[str, Any]:
    """Convert shared DSPy config fields into text-completion kwargs."""
    data = dict(config.extensions)
    for key in ("temperature", "max_tokens", "top_p"):
        value = getattr(config, key)
        if value is not None:
            data[key] = value
    if config.stop:
        data["stop"] = config.stop
    if config.logprobs is not None:
        data["logprobs"] = config.logprobs
    if config.n is not None:
        data["n"] = config.n
    return data


def reasoning_to_chat_kwargs(reasoning: Any) -> dict[str, Any]:
    data = {}
    if reasoning.effort is not None:
        data["reasoning_effort"] = reasoning.effort
    return data


def reasoning_to_responses_kwargs(reasoning: Any) -> dict[str, Any]:
    data = {}
    if reasoning.effort is not None:
        data["effort"] = reasoning.effort
    if reasoning.summary is not None:
        data["summary"] = reasoning.summary
    return {"reasoning": data} if data else {}


def _validate_openai_reasoning_temperature(config: LMConfig, *, model: str | None, endpoint: str) -> None:
    if not _is_openai_reasoning_model(model):
        return
    effort = getattr(config.reasoning, "effort", None) if config.reasoning is not None else None
    if effort in {None, "none"}:
        return
    if config.temperature in {None, 1}:
        return

    from dspy.utils.exceptions import LMUnsupportedFeatureError

    raise LMUnsupportedFeatureError(
        "OpenAI reasoning models only support the default temperature when reasoning effort is active. "
        "Use temperature=None or temperature=1, or set reasoning_effort='none'.",
        model=model,
        provider="openai",
        features=["temperature", "reasoning"],
        issues=[
            f"{endpoint} request used reasoning effort {effort!r} with temperature={config.temperature!r}.",
        ],
    )


def _uses_max_completion_tokens(model: str | None) -> bool:
    return _is_openai_reasoning_model(model)


def _is_openai_reasoning_model(model: str | None) -> bool:
    if not isinstance(model, str):
        return False
    model_name = model.removeprefix("openai/").lower()
    if "chat" in model_name:
        return False
    return model_name.startswith(("o1", "o3", "o4", "gpt-5"))


def prompt_cache_to_kwargs(cache: Any) -> dict[str, Any]:
    data = {}
    if cache.key is not None:
        data["prompt_cache_key"] = cache.key
    if cache.enabled is False:
        data["prompt_cache"] = False
    return data


def response_format_to_responses(value: Any) -> Any:
    if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
        return {"name": value.__name__, "type": "json_schema", "schema": value.model_json_schema()}
    return value


def responses_tool_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) and block.get("type") in {"text", "input_text"} else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# OpenAI response -> DSPy response
#
# Chat and text completions return `choices`; Responses returns one `output`
# list. Both are normalized to an `LMResponse` with one or more `LMOutput`s and
# typed DSPy parts.
# ---------------------------------------------------------------------------


def completion_to_lm_response(response: Any, request: LMRequest) -> LMResponse:
    """Convert an OpenAI Chat or text completion response into `LMResponse`."""
    choices = get_value(response, "choices", []) or []
    return LMResponse(
        model=get_value(response, "model") or request.model,
        outputs=[choice_to_lm_output(choice) for choice in choices],
        usage=usage_from_response(response),
        cache_hit=bool(get_value(response, "cache_hit", False)),
        response_id=get_value(response, "id"),
        provider_response=response,
    )


def choice_to_lm_output(choice: Any) -> LMOutput:
    """Convert one completion choice into one DSPy output candidate."""
    message = get_value(choice, "message")
    parts = []
    if message is not None:
        reasoning = get_value(message, "reasoning_content")
        if reasoning:
            parts.append(LMThinkingPart(text=str(reasoning)))
        content = get_value(message, "content")
        if content:
            parts.extend(message_content_to_parts(content))
        for tool_call in get_value(message, "tool_calls") or []:
            parts.append(provider_tool_call_to_part(tool_call))
        parts.extend(extract_citations_from_choice(choice))
    else:
        text = get_value(choice, "text")
        if text:
            parts.extend(message_content_to_parts(text))
    finish_reason = get_value(choice, "finish_reason")
    return LMOutput(
        parts=parts,
        finish_reason=finish_reason,
        truncated=finish_reason == "length",
        logprobs=get_value(choice, "logprobs"),
        provider_output=choice,
    )


def responses_to_lm_response(response: Any, request: LMRequest) -> LMResponse:
    """Convert an OpenAI Responses object into `LMResponse`.

    The Responses API represents one assistant answer as a sequence of output
    items: messages, function calls, reasoning, binary artifacts, images, and refusals.
    DSPy stores those as typed parts on one `LMOutput`.
    """
    parts = []
    for output_item in get_value(response, "output", []) or []:
        output_type = get_value(output_item, "type")
        if output_type == "message":
            for content_item in get_value(output_item, "content", []) or []:
                parts.extend(response_content_item_to_parts(content_item))
                parts.extend(responses_annotations_to_citations(content_item))
        elif output_type == "function_call":
            parts.append(responses_function_call_to_part(output_item))
        elif output_type in {"image", "output_image", "image_generation_call"}:
            parts.append(output_image_to_part(output_item))
        elif output_type in {"audio", "output_audio"}:
            parts.append(output_audio_to_part(output_item))
        elif output_type in {"file", "output_file"}:
            parts.append(output_file_to_part(output_item))
        elif output_type == "refusal":
            parts.append(refusal_to_part(output_item))
        elif output_type == "reasoning":
            for item in get_value(output_item, "content") or get_value(output_item, "summary") or []:
                text = get_value(item, "text")
                if text:
                    parts.append(LMThinkingPart(text=text))
    return LMResponse(
        model=get_value(response, "model") or request.model,
        outputs=[LMOutput(parts=parts, provider_output=response)],
        usage=usage_from_response(response),
        cache_hit=bool(get_value(response, "cache_hit", False)),
        response_id=get_value(response, "id"),
        provider_response=response,
    )


def message_content_to_parts(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [LMTextPart(text=content)]
    if not isinstance(content, list):
        return [LMTextPart(text=str(content))]
    parts = []
    for item in content:
        parts.extend(response_content_item_to_parts(item))
    return parts


def response_content_item_to_parts(item: Any) -> list[Any]:
    item_type = get_value(item, "type")
    text = get_value(item, "text")
    if item_type in {"text", "output_text", "input_text"} or (text is not None and item_type is None):
        return [LMTextPart(text=text)]
    if item_type in {"refusal", "output_refusal"}:
        return [refusal_to_part(item)]
    if item_type in {"image", "output_image", "image_url"}:
        return [output_image_to_part(item)]
    if item_type in {"audio", "output_audio", "input_audio"}:
        return [output_audio_to_part(item)]
    if item_type in {"file", "output_file", "input_file"}:
        return [output_file_to_part(item)]
    if item_type in {"tool_call", "function_call"}:
        return [provider_tool_call_to_part(item)]
    return []


def provider_tool_call_to_part(tool_call: Any) -> LMToolCallPart:
    """Convert an OpenAI-shaped tool call into a DSPy tool-call part."""
    function = get_value(tool_call, "function", {})
    name = get_value(function, "name") or get_value(tool_call, "name")
    arguments = get_value(function, "arguments", get_value(tool_call, "arguments", "{}"))
    provider_data = model_dump(tool_call)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except Exception as error:
        args = {}
        provider_data["raw_arguments"] = arguments
        provider_data["arguments_parse_error"] = str(error)
    call_id = get_value(tool_call, "call_id") or get_value(tool_call, "id")
    return LMToolCallPart(id=call_id, name=name or "", args=args, provider_data=provider_data)


def responses_function_call_to_part(output_item: Any) -> LMToolCallPart:
    """Convert one Responses function_call item into a DSPy tool-call part."""
    args = get_value(output_item, "arguments", {})
    provider_data = model_dump(output_item)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception as error:
            provider_data["raw_arguments"] = args
            provider_data["arguments_parse_error"] = str(error)
            args = {}
    return LMToolCallPart(id=get_value(output_item, "call_id"), name=get_value(output_item, "name", ""), args=args, provider_data=provider_data)


def citation_to_part(citation: Any) -> LMCitationPart:
    if hasattr(citation, "model_dump"):
        citation = model_dump(citation)
    if not isinstance(citation, dict):
        citation = {"text": str(citation)}
    citation_fields = {"cited_text", "text", "supported_text", "document_title", "title", "url"}
    return LMCitationPart(
        text=citation.get("cited_text") or citation.get("text") or citation.get("supported_text"),
        title=citation.get("document_title") or citation.get("title"),
        url=citation.get("url"),
        metadata={key: value for key, value in citation.items() if key not in citation_fields},
    )


def extract_citations_from_choice(choice: Any) -> list[LMCitationPart]:
    try:
        message = get_value(choice, "message")
        provider_specific_fields = get_value(message, "provider_specific_fields", {}) or {}
        citations_data = provider_specific_fields.get("citations")
        if isinstance(citations_data, list):
            citations = []
            for item in citations_data:
                citations.extend(item if isinstance(item, list) else [item])
            return [citation_to_part(citation) for citation in citations]
    except Exception:
        return []
    return []


def responses_annotations_to_citations(content_item: Any) -> list[LMCitationPart]:
    return [citation_to_part(annotation) for annotation in get_value(content_item, "annotations", []) or []]


def output_image_to_part(value: Any) -> LMImagePart:
    data = model_dump(value)
    image_url = data.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    source = image_url or data.get("url")
    b64_data = data.get("b64_json") or data.get("data")
    file_id = data.get("file_id")
    media_type = data.get("media_type") or data.get("mime_type") or "image/png"
    detail = data.get("detail")
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMImagePart(data=b64_data, media_type=media_type, detail=detail)
    if source is not None:
        return LMImagePart(url=source, media_type=media_type, detail=detail)
    if file_id is not None:
        return LMImagePart(file_id=file_id, media_type=media_type, detail=detail)
    raise ValueError("Provider image output did not include data, url, or file_id.")


def output_audio_to_part(value: Any) -> LMAudioPart:
    data = model_dump(value)
    audio = data.get("audio") if isinstance(data.get("audio"), dict) else data
    source = audio.get("url")
    b64_data = audio.get("data") or audio.get("b64_json")
    file_id = audio.get("file_id")
    media_type = audio.get("media_type") or audio.get("mime_type") or "audio/wav"
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMAudioPart(data=b64_data, media_type=media_type)
    if source is not None:
        return LMAudioPart(url=source, media_type=media_type)
    if file_id is not None:
        return LMAudioPart(file_id=file_id, media_type=media_type)
    raise ValueError("Provider audio output did not include data, url, or file_id.")


def output_file_to_part(value: Any) -> LMBinaryPart:
    data = model_dump(value)
    file = data.get("file") if isinstance(data.get("file"), dict) else data
    source = file.get("url")
    b64_data = file.get("file_data") or file.get("data")
    file_id = file.get("file_id") or file.get("id")
    filename = file.get("filename")
    media_type = file.get("media_type") or file.get("mime_type") or "application/octet-stream"
    if b64_data is not None:
        if isinstance(b64_data, str) and b64_data.startswith("data:"):
            media_type, b64_data = split_data_uri(b64_data)
        return LMBinaryPart(data=b64_data, media_type=media_type, filename=filename)
    if source is not None:
        return LMBinaryPart(url=source, media_type=media_type, filename=filename)
    if file_id is not None:
        return LMBinaryPart(file_id=file_id, media_type=media_type, filename=filename)
    raise ValueError("Provider file output did not include data, url, or file_id.")


def refusal_to_part(value: Any) -> LMRefusalPart:
    text = get_value(value, "refusal") or get_value(value, "text") or get_value(value, "content") or str(value)
    return LMRefusalPart(text=str(text))


def cost_from_response(response: Any) -> float | None:
    hidden = getattr(response, "_hidden_params", None) or {}
    return hidden.get("response_cost") if isinstance(hidden, dict) else None


def usage_from_response(response: Any) -> LMUsage | None:
    """Convert provider usage objects or dictionaries into `LMUsage`."""
    usage = get_value(response, "usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = model_dump(usage)
    elif not isinstance(usage, dict):
        data = {}
        for key in dir(usage):
            if key.startswith("_"):
                continue
            try:
                value = getattr(usage, key)
            except Exception:
                continue
            if value is not None and not callable(value):
                data[key] = value
        usage = data
    return LMUsage(**dict(usage))


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def media_source(part: LMImagePart | LMAudioPart | LMDocumentPart | LMBinaryPart) -> str:
    if part.data is not None:
        return data_uri(part.media_type, part.data)
    if part.url is not None:
        return part.url
    if part.file_id is not None:
        return part.file_id
    if part.path is not None:
        return data_uri_from_path(part.path, fallback_media_type=part.media_type)
    raise ValueError(f"{type(part).__name__} has no media source.")


def read_path_base64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def media_type_for_path(path: str, *, fallback: str) -> str:
    return mimetypes.guess_type(path)[0] or fallback


def data_uri_from_path(path: str, *, fallback_media_type: str) -> str:
    return data_uri(media_type_for_path(path, fallback=fallback_media_type), read_path_base64(path))


def data_uri(media_type: str, data: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{media_type};base64,{data}"


def split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def media_format(media_type: str) -> str:
    format_ = media_type.split("/", 1)[1] if "/" in media_type else media_type
    return {"x-wav": "wav", "mpeg": "mp3"}.get(format_, format_)


def part_text(value: Any) -> str:
    return value.text if isinstance(value, LMTextPart) else str(value)


def get_value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _rebuild_pydantic_serializers(value: Any) -> None:
    stack = [value]
    seen = set()
    for item in stack:
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, pydantic.BaseModel):
            type(item).model_rebuild(force=True, raise_errors=False)
            stack.extend([*item.__dict__.values(), *((item.__pydantic_extra__ or {}).values())])
        elif isinstance(item, (dict, list, tuple)):
            stack.extend(item.values() if isinstance(item, dict) else item)


def model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except TypeError as error:
            if "MockValSer" not in str(error):
                raise
            _rebuild_pydantic_serializers(value)
            return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    data = {}
    for key in ("id", "call_id", "type", "name", "arguments", "status", "text", "refusal", "url", "data", "file_id", "filename", "media_type", "mime_type"):
        item = getattr(value, key, None)
        if item is not None:
            data[key] = item
    return data


# ---------------------------------------------------------------------------
# Current BaseLM adapter compatibility helpers
#
# These helpers keep today's adapter->BaseLM boundary normalized internally while
# BaseLM still returns legacy `list[str | dict | None]` outputs. Future native
# LanguageModel implementations should use completion_to_lm_response() /
# responses_to_lm_response() directly.
# ---------------------------------------------------------------------------


def lm_response_from_legacy_outputs(outputs: list[dict[str, Any] | str | None], request: LMRequest) -> LMResponse:
    """Normalize current legacy `BaseLM` outputs into an `LMResponse`."""
    if not outputs:
        return LMResponse(model=request.model, outputs=[LMOutput(parts=[], metadata={"empty_legacy_outputs": True})])
    return LMResponse(model=request.model, outputs=[lm_output_from_legacy_output(output) for output in outputs])


def legacy_outputs_from_lm_response(response: LMResponse) -> list[dict[str, Any] | str | None]:
    """Return legacy adapter postprocess values from a normalized response.

    This is a temporary compatibility helper for current adapter postprocessing.
    Prefer the exact original legacy output when it is available so compatibility
    hooks that distinguish `str` from provider dictionaries keep working.
    """
    outputs = []
    for output in response.outputs:
        if output.metadata.get("empty_legacy_outputs"):
            continue
        if output.provider_output is not None:
            outputs.append(output.provider_output)
        elif output.text is not None and not output.reasoning_content and not output.tool_calls and not output.citations and output.logprobs is None:
            outputs.append(output.text)
        else:
            outputs.append(output.to_output_dict())
    return outputs


def lm_output_from_legacy_output(output: dict[str, Any] | str | None) -> LMOutput:
    """Normalize one current legacy `BaseLM` output item into an `LMOutput`."""
    if isinstance(output, str):
        return LMOutput(parts=[LMTextPart(text=output)], provider_output=output)
    if output is None:
        return LMOutput(parts=[])

    parts = []
    text = output.get("text")
    if text:
        parts.append(LMTextPart(text=text))
    reasoning = output.get("reasoning_content")
    if reasoning:
        parts.append(LMThinkingPart(text=str(reasoning)))
    for tool_call in output.get("tool_calls") or []:
        parts.append(provider_tool_call_to_part(tool_call))
    for citation in output.get("citations") or []:
        parts.append(citation_to_part(citation))
    return LMOutput(parts=parts, logprobs=output.get("logprobs"), provider_output=output)
