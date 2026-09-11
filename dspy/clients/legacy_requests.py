"""Legacy chat-to-Responses translation, independent of the typed LM API.

This boundary preserves provider-native fields accepted by ordinary DSPy calls.
It deliberately does not force those fields through lm15's narrower vocabulary.
"""

import copy
from typing import Any

import pydantic


def _close_object_schemas(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    closed = dict(schema)
    for key in ("items", "additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"):
        if isinstance(closed.get(key), dict):
            closed[key] = _close_object_schemas(closed[key])
    for key in ("items", "prefixItems", "anyOf", "oneOf", "allOf"):
        if isinstance(closed.get(key), list):
            closed[key] = [_close_object_schemas(item) for item in closed[key]]
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        if isinstance(closed.get(key), dict):
            closed[key] = {name: _close_object_schemas(sub) for name, sub in closed[key].items()}
    if closed.get("type") == "object" and "additionalProperties" not in closed:
        closed["additionalProperties"] = False
    return closed


def _content(content, role):
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    blocks = []
    for block in content or []:
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            blocks.append({"type": text_type, "text": block.get("text", "")})
        elif kind == "image_url":
            image = block["image_url"]
            image = image if isinstance(image, dict) else {"url": image}
            converted = {"type": "input_image", "image_url": image.get("url", "")}
            if image.get("detail") is not None:
                converted["detail"] = image["detail"]
            blocks.append(converted)
        elif kind == "file":
            file = block.get("file", {})
            blocks.append({"type": "input_file", **{key: file.get(key) for key in ("file_data", "filename", "file_id")}})
        else:
            blocks.append(copy.deepcopy(block))
    return blocks


def chat_to_responses(request: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(request)
    messages = data.pop("messages", [])
    inputs = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "") if isinstance(block, dict) and block.get("type") in {"text", "input_text"}
                    else str(block) for block in content
                )
            item = {"type": "function_call_output", "output": content or ""}
            if message.get("tool_call_id") is not None:
                item["call_id"] = message["tool_call_id"]
            inputs.append(item)
            continue
        calls = message.get("tool_calls") or []
        blocks = _content(content, role)
        if blocks or role != "assistant" or not calls:
            item = {"role": role, "content": blocks}
            if message.get("name") is not None:
                item["name"] = message["name"]
            inputs.append(item)
        if role == "assistant":
            for call in calls:
                function = call.get("function") or call
                item = {"type": "function_call", "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}")}
                call_id = call.get("call_id") or call.get("id")
                if call_id is not None:
                    item["call_id"] = call_id
                inputs.append(item)
    data["input"] = inputs
    if "max_completion_tokens" in data and "max_tokens" not in data:
        data["max_tokens"] = data.pop("max_completion_tokens")
    if data.get("max_tokens") is not None:
        data["max_output_tokens"] = data.pop("max_tokens")
    else:
        data.pop("max_tokens", None)
    if "reasoning_effort" in data:
        effort = data.pop("reasoning_effort")
        if data.get("reasoning") is None:
            data["reasoning"] = {"effort": effort, "summary": "auto"}
    if data.get("tools"):
        data["tools"] = [
            {"type": "function", **{key: val for key, val in tool.items() if key not in {"type", "function"}}, **tool["function"]}
            if "function" in tool else tool for tool in data["tools"]
        ]
        for tool in data["tools"]:
            if tool.get("type") == "function" and tool.get("description") is None:
                tool.pop("description", None)
    choice = data.get("tool_choice")
    if isinstance(choice, dict) and "function" in choice:
        data["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
    format_ = data.pop("response_format", None)
    if format_ is not None:
        if isinstance(format_, type) and issubclass(format_, pydantic.BaseModel):
            format_ = {"name": format_.__name__, "type": "json_schema", "schema": _close_object_schemas(format_.model_json_schema())}
        data["text"] = {**data.get("text", {}), "format": format_}
    # The former config serializer omitted absent common generation parameters.
    for key in ("temperature", "top_p", "n", "logprobs", "reasoning", "tool_choice"):
        if data.get(key) is None:
            data.pop(key, None)
    if not data.get("stop"):
        data.pop("stop", None)
    return data
