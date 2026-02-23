from typing import Any, Literal

import json
import pydantic


class HistoryMessage(pydantic.BaseModel):
    """A role-tagged history event.

    Each message represents one event with arbitrary structured fields.
    """

    role: Literal["user", "assistant", "tool"]
    fields: dict[str, Any]

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class HistoryResumeState(pydantic.BaseModel):
    completed_turns: int
    last_completed_tool_name: str | None
    pending_tool_call: tuple[str, dict[str, Any]] | None

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )


class History(pydantic.BaseModel):
    """Role/event-based conversation history.

    `History` stores explicit events, each with a `role` (`user`, `assistant`, or
    `tool`) and arbitrary structured `fields`.
    """

    messages: list[HistoryMessage]
    max_chars: int = 200_000
    overflow_drop_strategy: Literal["oldest_pair", "oldest_message"] = "oldest_pair"

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def estimated_size(self) -> int:
        return sum(len(json.dumps(message.model_dump(), ensure_ascii=False)) for message in self.messages)

    def trimmed(self) -> "History":
        if self.estimated_size() <= self.max_chars:
            return self

        history = self
        while history.messages and history.estimated_size() > history.max_chars:
            history = history.drop_for_overflow()

        return history

    def append(self, message: HistoryMessage) -> "History":
        return self.model_copy(update={"messages": [*self.messages, message]}).trimmed()

    def append_many(self, messages: list[HistoryMessage]) -> "History":
        history = self
        for message in messages:
            history = history.append(message)
        return history

    def drop_oldest_pair(self) -> "History":
        if len(self.messages) < 2:
            raise ValueError("Cannot truncate history further because it does not contain a full assistant/tool pair.")
        return self.model_copy(update={"messages": self.messages[2:]})

    def drop_oldest_message(self) -> "History":
        if not self.messages:
            raise ValueError("Cannot truncate history further because it has no messages.")
        return self.model_copy(update={"messages": self.messages[1:]})

    def drop_for_overflow(self) -> "History":
        if self.overflow_drop_strategy == "oldest_pair":
            return self.drop_oldest_pair()
        if self.overflow_drop_strategy == "oldest_message":
            return self.drop_oldest_message()
        raise ValueError(f"Unknown history overflow_drop_strategy: {self.overflow_drop_strategy}")

    def parse_resume_state(
        self,
        *,
        input_args: dict[str, Any],
        input_keys: set[str],
        available_tools: set[str],
        resume_mode: Literal["off", "auto", "strict"],
        assistant_required_fields: set[str] | None = None,
        assistant_tool_name_field: str = "next_tool_name",
        assistant_tool_args_field: str = "next_tool_args",
        tool_message_tool_name_field: str = "tool_name",
        tool_message_observation_field: str = "observation",
    ) -> HistoryResumeState:
        if resume_mode == "off":
            return HistoryResumeState(completed_turns=0, last_completed_tool_name=None, pending_tool_call=None)

        if resume_mode not in {"auto", "strict"}:
            raise ValueError("History `resume_mode` must be one of: 'off', 'auto', 'strict'.")

        required_assistant_fields = assistant_required_fields or {
            "next_thought",
            assistant_tool_name_field,
            assistant_tool_args_field,
        }

        completed_turns = 0
        last_completed_tool_name = None
        i = 0
        while i < len(self.messages):
            message = self.messages[i]

            if message.role == "user":
                if resume_mode == "strict":
                    overlapping_keys = input_keys.intersection(message.fields.keys())
                    for key in overlapping_keys:
                        if key in input_args and input_args[key] != message.fields[key]:
                            raise ValueError(
                                f"Cannot resume trajectory: history input field `{key}` does not match current input."
                            )
                i += 1
                continue

            if message.role == "tool":
                if resume_mode == "strict":
                    raise ValueError("Cannot resume trajectory: found tool observation without a preceding assistant step.")
                i += 1
                continue

            fields = message.fields
            if not required_assistant_fields.issubset(fields):
                if resume_mode == "strict":
                    raise ValueError("Cannot resume trajectory: assistant history message is missing required fields.")
                i += 1
                continue

            next_tool_name = fields[assistant_tool_name_field]
            if next_tool_name not in available_tools:
                if resume_mode == "strict":
                    raise ValueError(f"Cannot resume trajectory: tool `{next_tool_name}` is not available.")
                i += 1
                continue

            next_message_index = i + 1
            has_following_tool_observation = (
                next_message_index < len(self.messages) and self.messages[next_message_index].role == "tool"
            )
            if has_following_tool_observation:
                tool_fields = self.messages[next_message_index].fields
                observed_tool_name = tool_fields.get(tool_message_tool_name_field, next_tool_name)
                if observed_tool_name != next_tool_name and resume_mode == "strict":
                    raise ValueError(
                        f"Cannot resume trajectory: tool observation `{observed_tool_name}` does not match preceding tool call `{next_tool_name}`."
                    )
                if tool_message_observation_field in tool_fields:
                    completed_turns += 1
                    last_completed_tool_name = next_tool_name
                    i += 2
                    continue

            return HistoryResumeState(
                completed_turns=completed_turns,
                last_completed_tool_name=last_completed_tool_name,
                pending_tool_call=(next_tool_name, fields[assistant_tool_args_field]),
            )

        return HistoryResumeState(
            completed_turns=completed_turns,
            last_completed_tool_name=last_completed_tool_name,
            pending_tool_call=None,
        )


def _estimated_size(messages: list[HistoryMessage]) -> int:
    return sum(len(json.dumps(message.model_dump(), ensure_ascii=False)) for message in messages)
