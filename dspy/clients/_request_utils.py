"""Shared request transformation helpers.

These convert DSPy's chat-style request format to other OpenAI API formats
(Responses API, text completions). Used by multiple backends.
"""

from typing import Any


def convert_chat_to_responses_request(request: dict[str, Any]) -> dict[str, Any]:
    """Convert a chat-format request to the OpenAI Responses API format.

    See https://platform.openai.com/docs/api-reference/responses/create
    """
    import pydantic

    request = dict(request)
    if "messages" in request:
        content_blocks = []
        for msg in request.pop("messages"):
            c = msg.get("content")
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                for item in c:
                    content_blocks.append(_convert_content_item(item))
        request["input"] = [{"role": msg.get("role", "user"), "content": content_blocks}]

    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            response_format = {
                "name": response_format.__name__,
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}

    return request


def _convert_content_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat API content item to Responses API format."""
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {"type": "input_image", "image_url": image_url}
    elif item.get("type") == "text":
        return {"type": "input_text", "text": item.get("text", "")}
    elif item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }
    return item
