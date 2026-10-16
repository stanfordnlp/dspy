"""Preserve DSPy's public output shapes without the retired experimental types."""

import json
from typing import Any


def value(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def plain(obj: Any):
    """Read SDK fields without invoking deferred Pydantic serializers."""
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {key: plain(item) for key, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(item) for item in obj]
    return obj


def responses_outputs(response: Any) -> list[dict[str, Any]]:
    """Render a Responses API answer using DSPy's existing list-of-dicts API."""
    texts, reasoning, calls, citations = [], [], [], []
    for item in value(response, "output", []) or []:
        kind = value(item, "type")
        if kind == "message":
            for part in value(item, "content", []) or []:
                if value(part, "type") in {"text", "output_text", "input_text"}:
                    text = value(part, "text")
                    if text is not None:
                        texts.append(text)
                for annotation in value(part, "annotations", []) or []:
                    raw = plain(annotation)
                    fields = {"cited_text", "text", "supported_text", "document_title", "title", "url"}
                    citation = {
                        "type": "citation",
                        "metadata": {key: val for key, val in raw.items() if key not in fields and val is not None},
                        "text": raw.get("cited_text") or raw.get("text") or raw.get("supported_text"),
                        "title": raw.get("document_title") or raw.get("title"),
                        "url": raw.get("url"),
                    }
                    citations.append({key: val for key, val in citation.items() if val is not None})
        elif kind == "function_call":
            arguments = value(item, "arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            call = {"type": "function", "function": {"name": value(item, "name", ""), "arguments": json.dumps(arguments)}}
            if value(item, "call_id") is not None:
                call["id"] = value(item, "call_id")
            calls.append(call)
        elif kind == "reasoning":
            for part in value(item, "content") or value(item, "summary") or []:
                if text := value(part, "text"):
                    reasoning.append(text)
    output = {"text": "".join(texts) if texts else None}
    if reasoning:
        output["reasoning_content"] = "".join(reasoning)
    if calls:
        output["tool_calls"] = calls
    if citations:
        output["citations"] = citations
    return [output]
