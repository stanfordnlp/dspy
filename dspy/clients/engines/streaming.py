"""Bridge canonical events to DSPy's existing streaming listeners."""

import json

from dspy.clients.call_result import AttributeDict, attributes


class EngineChunk(AttributeDict):
    """Listener-facing chunk. Canonical assembly uses the original event."""

    def json(self):
        return json.dumps(self)


class ListenerBridge:
    def __init__(self, model, predict_id=None):
        self.model = model
        self.predict_id = predict_id
        self.id = None
        self.tools = {}

    def chunk(self, event):
        if event.type == "start":
            self.id = event.id
            self.model = event.model or self.model
            return None
        delta = {}
        finish = None
        if event.type == "end":
            finish = "tool_calls" if event.finish_reason == "tool_call" else event.finish_reason
        elif event.type == "delta":
            part = event.delta
            if part.type == "text":
                delta["content"] = part.text
            elif part.type == "thinking":
                delta["reasoning_content"] = part.text
            elif part.type == "tool_call":
                index = self.tools.setdefault(part.part_index, len(self.tools))
                function = {"arguments": part.input}
                if part.name is not None:
                    function["name"] = part.name
                call = {"index": index, "type": "function", "function": function}
                if part.id is not None:
                    call["id"] = part.id
                delta["tool_calls"] = [call]
            elif part.type == "citation":
                delta["provider_specific_fields"] = {"citation": {
                    key: val for key, val in {"text": part.text, "title": part.title, "url": part.url}.items()
                    if val is not None
                }}
            elif part.type == "continuation":
                # Opaque replay state is retained by canonical assembly, not
                # exposed as visible answer text or mistaken for a tool call.
                return None
            else:
                delta[part.type] = {"data": part.data, "url": part.url, "file_id": part.file_id,
                                    "media_type": part.media_type}
        else:
            return None
        delta.setdefault("content", None)
        return EngineChunk(id=self.id, model=self.model, predict_id=self.predict_id,
                           choices=[attributes({"index": 0, "delta": delta, "finish_reason": finish})])
