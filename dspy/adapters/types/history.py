from typing import Any, Callable

import pydantic
from pydantic import Field, model_validator

from dspy.adapters.types.tool import ToolCallResults, ToolCalls


class History(pydantic.BaseModel):
    """Reusable ordered field history."""

    messages: list[dict[str, Any]] = Field(default_factory=list)

    model_config = pydantic.ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def _normalize_messages(self) -> "History":
        self.messages = [
            self._normalize_message(message) if isinstance(message, dict) else message for message in self.messages
        ]
        return self

    def __init__(self, *args: Any, compact_fn: Callable[["History"], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_compact_fn", compact_fn)

    @pydantic.model_serializer()
    def serialize_model(self) -> dict[str, Any]:
        return {"messages": [self._serialize_message(message) for message in self.messages]}

    def compact_if_needed(self) -> None:
        fn = getattr(self, "_compact_fn", None)
        if fn is not None:
            fn(self)

    def append(self, message: dict[str, Any]) -> dict[str, Any]:
        message = self._normalize_message(dict(message))
        self.messages.append(message)
        return message

    def to_lm_messages(self, adapter: Any, signature: type[Any], *, use_native_tool_calls: bool = False) -> list[Any]:
        return adapter.format_history(self, signature, use_native_tool_calls=use_native_tool_calls)

    @staticmethod
    def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
        normalized = {}
        for key, value in message.items():
            if key == "tool_calls" and isinstance(value, list):
                normalized[key] = ToolCalls.model_validate(value)
            elif key == "tool_call_results" and isinstance(value, list):
                normalized[key] = ToolCallResults.model_validate(value)
            elif isinstance(value, dict) and set(value.keys()) == {"tool_calls"} and isinstance(value["tool_calls"], list):
                normalized[key] = ToolCalls.model_validate(value)
            elif (
                isinstance(value, dict)
                and set(value.keys()) == {"tool_call_results"}
                and isinstance(value["tool_call_results"], list)
            ):
                normalized[key] = ToolCallResults.model_validate(value)
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _serialize_message(message: dict[str, Any]) -> dict[str, Any]:
        serialized = {}
        for key, value in message.items():
            if isinstance(value, (ToolCalls, ToolCallResults)):
                serialized[key] = value.format()
            elif hasattr(value, "model_dump"):
                serialized[key] = value.model_dump()
            else:
                serialized[key] = value
        return serialized


def truncate_oldest_actions(history: History, *, max_tokens: int = 200_000, keep_n: int = 3) -> None:
    if len(str(history.messages)) // 4 <= max_tokens:
        return

    action_indices = [
        idx
        for idx, message in enumerate(history.messages)
        if isinstance(message, dict) and any(isinstance(value, ToolCalls) for value in message.values())
    ]
    drop_count = len(action_indices) - keep_n
    if drop_count <= 0:
        return

    drop_indices = set(action_indices[:drop_count])

    history.messages[:] = [message for idx, message in enumerate(history.messages) if idx not in drop_indices]


def make_truncate_oldest_actions(max_tokens: int = 200_000, keep_n: int = 3) -> Callable[[History], None]:
    def _compact(history: History) -> None:
        truncate_oldest_actions(history, max_tokens=max_tokens, keep_n=keep_n)

    return _compact
