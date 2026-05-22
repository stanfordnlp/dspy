"""Private compatibility helpers for legacy `dspy.Type` marker serialization."""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair

from dspy.adapters.types.base_type import CUSTOM_TYPE_END_IDENTIFIER, CUSTOM_TYPE_START_IDENTIFIER
from dspy.core.types import LMAudioPart, LMBinaryPart, LMDocumentPart, LMImagePart, LMMessage, LMPart, LMTextPart


def _expand_legacy_custom_type_markers_in_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    """Expand legacy marker payloads in an OpenAI-chat-shaped user message."""
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return message
    content = message["content"]
    if CUSTOM_TYPE_START_IDENTIFIER not in content:
        return message
    return {**message, "content": _split_legacy_custom_type_text_to_blocks(content)}


def _expand_legacy_custom_type_markers_in_lm_message(message: LMMessage) -> LMMessage:
    """Expand legacy marker payloads in user `LMMessage`s into normalized parts."""
    if message.role != "user":
        return message
    expanded_parts: list[LMPart] = []
    changed = False
    for part in message.parts:
        if not isinstance(part, LMTextPart) or CUSTOM_TYPE_START_IDENTIFIER not in part.text:
            expanded_parts.append(part)
            continue
        changed = True
        expanded_parts.extend(_split_legacy_custom_type_text_to_parts(part.text))
    return message.model_copy(update={"parts": expanded_parts}) if changed else message


def _split_legacy_custom_type_text_to_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    last_end = 0
    for match in re.finditer(_marker_pattern(), text, re.DOTALL):
        start, end = match.span()
        if start > last_end:
            blocks.append({"type": "text", "text": text[last_end:start]})
        blocks.extend(_legacy_custom_type_payload_to_blocks(match.group(1).strip()))
        last_end = end
    if last_end < len(text):
        blocks.append({"type": "text", "text": text[last_end:]})
    return blocks


def _split_legacy_custom_type_text_to_parts(text: str) -> list[LMPart]:
    parts: list[LMPart] = []
    last_end = 0
    for match in re.finditer(_marker_pattern(), text, re.DOTALL):
        start, end = match.span()
        if start > last_end:
            parts.append(LMTextPart(text=text[last_end:start]))
        parts.extend(_legacy_custom_type_payload_to_parts(match.group(1).strip()))
        last_end = end
    if last_end < len(text):
        parts.append(LMTextPart(text=text[last_end:]))
    return parts


def _legacy_custom_type_payload_to_blocks(payload: str) -> list[dict[str, Any]]:
    parsed = _parse_legacy_payload(payload)
    if isinstance(parsed, list):
        return [block if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in parsed]
    return [{"type": "text", "text": payload}]


def _legacy_custom_type_payload_to_parts(payload: str) -> list[LMPart]:
    parsed = _parse_legacy_payload(payload)
    if not isinstance(parsed, list):
        return [LMTextPart(text=payload)]
    return [_legacy_content_block_to_lm_part(block) for block in parsed]


def _parse_legacy_payload(payload: str) -> Any:
    for parse_fn in (json.loads, _parse_doubly_quoted_json, json_repair.loads):
        try:
            return parse_fn(payload)
        except Exception:
            continue
    return None


def _parse_doubly_quoted_json(value: str) -> Any:
    # Legacy `Type` payloads can be JSON-encoded twice when the serialized
    # marker string is nested inside a larger JSON value, e.g. list[Image].
    return json.loads(json.loads(f'"{value}"'))


def _legacy_content_block_to_lm_part(block: Any) -> LMPart:
    if not isinstance(block, dict):
        return LMTextPart(text=str(block))

    block_type = block.get("type")
    if block_type == "text":
        return LMTextPart(text=block.get("text", ""))
    if block_type == "image_url":
        image_url = block.get("image_url", {})
        source = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(source, str) and source.startswith("data:") and "," in source:
            media_type, data = _split_data_uri(source)
            return LMImagePart(data=data, media_type=media_type)
        return LMImagePart(url=source or "")
    if block_type == "input_audio":
        audio = block.get("input_audio", {})
        return LMAudioPart(data=audio.get("data", ""), media_type=f"audio/{audio.get('format', 'wav')}")
    if block_type == "file":
        file = block.get("file", {})
        if file.get("file_data") is not None:
            media_type, data = _split_data_uri(file["file_data"])
            return LMBinaryPart(data=data, media_type=media_type, filename=file.get("filename"), metadata={"legacy_content_block": block})
        return LMBinaryPart(file_id=file.get("file_id", ""), filename=file.get("filename"), metadata={"legacy_content_block": block})
    if block_type == "document":
        source = block.get("source", {})
        return LMDocumentPart(
            source=source if isinstance(source, dict) else {"type": "text", "data": str(source)},
            citations=block.get("citations") or {"enabled": True},
            title=block.get("title"),
            context=block.get("context"),
        )

    # Keep unknown provider-shaped blocks losslessly for the current BaseLM
    # compatibility path. `openai_format.part_to_openai_blocks()` rehydrates
    # this metadata before the legacy provider-shaped call. A later normalized
    # type pass should replace this with an explicit opaque part or remove it
    # when marker-based custom type rendering is retired.
    return LMTextPart(text="", metadata={"legacy_content_block": block})


def _marker_pattern() -> str:
    return rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"


def _split_data_uri(value: str) -> tuple[str, str]:
    if value.startswith("data:") and "," in value:
        header, data = value.split(",", 1)
        return header.removeprefix("data:").split(";", 1)[0], data
    return "application/octet-stream", value
