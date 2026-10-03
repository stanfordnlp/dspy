"""Private compatibility helpers for legacy `dspy.Type` marker serialization."""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair

from dspy.adapters.types.base_type import CUSTOM_TYPE_END_IDENTIFIER, CUSTOM_TYPE_START_IDENTIFIER


def _expand_legacy_custom_type_markers_in_chat_message(message: dict[str, Any]) -> dict[str, Any]:
    """Expand legacy marker payloads in an OpenAI-chat-shaped user message."""
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return message
    content = message["content"]
    if CUSTOM_TYPE_START_IDENTIFIER not in content:
        return message
    return {**message, "content": _split_legacy_custom_type_text_to_blocks(content)}


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


def _legacy_custom_type_payload_to_blocks(payload: str) -> list[dict[str, Any]]:
    parsed = _parse_legacy_payload(payload)
    if isinstance(parsed, list):
        return [block if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in parsed]
    return [{"type": "text", "text": payload}]


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


def _marker_pattern() -> str:
    return rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"


def _split_data_uri(value: str) -> tuple[str, str]:
    if value.startswith("data:") and "," in value:
        header, data = value.split(",", 1)
        return header.removeprefix("data:").split(";", 1)[0], data
    return "application/octet-stream", value
