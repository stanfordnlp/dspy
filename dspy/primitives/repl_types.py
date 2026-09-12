"""
REPL data types for RLM and interpreter interactions.

These types represent the state and history of REPL-based execution:
- REPLVariable: Metadata about variables available in the REPL
- REPLEntry: A single interaction (reasoning, code, output)
- REPLHistory: Container for the full interaction history
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

import pydantic
from pydantic import Field

from dspy.adapters.utils import serialize_for_json

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

__all__ = ["REPLVariable", "REPLEntry", "REPLHistory"]


class REPLVariable(pydantic.BaseModel):
    """Metadata about a variable available in the REPL environment."""

    name: str
    type_name: str
    desc: str = ""
    constraints: str = ""
    total_length: int
    preview: str

    model_config = pydantic.ConfigDict(frozen=True)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        field_info: FieldInfo | None = None,
        preview_chars: int = 1000,
    ) -> REPLVariable:
        """Create REPLVariable from an actual value and optional field info.

        Args:
            name: Variable name
            value: The actual value
            field_info: Optional pydantic FieldInfo with desc/constraints metadata
            preview_chars: Max characters for preview
        """
        jsonable = serialize_for_json(value)
        if isinstance(jsonable, (dict, list)):
            value_str = json.dumps(jsonable, indent=2)
        else:
            value_str = str(jsonable)
        is_truncated = len(value_str) > preview_chars
        if is_truncated:
            head_chars = preview_chars // 2
            tail_chars = preview_chars - head_chars if preview_chars > 0 else head_chars
            preview = value_str[:head_chars] + "..." + value_str[-tail_chars:]
        else:
            preview = value_str

        # Extract desc and constraints from field_info if provided
        desc = ""
        constraints = ""
        if field_info and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            raw_desc = field_info.json_schema_extra.get("desc", "")
            # Skip placeholder descs like "${name}"
            if raw_desc and not raw_desc.startswith("${"):
                desc = raw_desc
            constraints = field_info.json_schema_extra.get("constraints", "")

        return cls(
            name=name,
            type_name=type(value).__name__,
            desc=desc,
            constraints=constraints,
            total_length=len(value_str),
            preview=preview,
        )

    def format(self) -> str:
        """Format variable metadata for prompt inclusion."""
        lines = [f"Variable: `{self.name}` (access it in your code)"]
        lines.append(f"Type: {self.type_name}")
        if self.desc:
            lines.append(f"Description: {self.desc}")
        if self.constraints:
            lines.append(f"Constraints: {self.constraints}")
        lines.append(f"Total length: {self.total_length:,} characters")
        lines.append(f"Preview:\n```\n{self.preview}\n```")
        return "\n".join(lines)

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()


class REPLEntry(pydantic.BaseModel):
    """A single REPL interaction entry containing reasoning, code, and output."""

    reasoning: str = ""
    code: str
    output: str

    max_output_chars: int

    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, value: Any) -> str:
        # The RLM's internal action signature types `reasoning` as `dspy.Reasoning`, so predictions carry a Reasoning
        # object rather than a plain str. Store its text so history entries stay plain strings.
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def format_output(output: str, max_output_chars: int = 10_000) -> str:
        """Format output with head+tail truncation, preserving true length in header."""
        raw_len = len(output)
        if raw_len > max_output_chars:
            head_chars = max_output_chars // 2
            tail_chars = max_output_chars - head_chars if max_output_chars > 0 else head_chars
            omitted = raw_len - max_output_chars
            output = output[:head_chars] + f"\n\n... ({omitted:,} characters omitted) ...\n\n" + output[-tail_chars:]
        return output

    def format(self, index: int) -> str:
        """Format this entry for inclusion in prompts."""
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        code_block = f"```python\n{self.code}\n```"
        return f"=== Step {index + 1} ===\n{reasoning_line}Code:\n{code_block}\n{self.format_output(self.output, self.max_output_chars)}"


REPL_ENTRY_KEY = "repl_entry"


def build_repl_event(
    inputs: dict[str, Any] | None,
    repl_entry: REPLEntry | None,
    outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one RLM conversation-history event.

    This is the single definition of the event layout that `split_repl_event` reads back: the RLM's own input fields
    (only on the first iteration of a turn) come first, then the REPL entry under `REPL_ENTRY_KEY`, then the RLM's
    output fields (only on the final iteration). Key order is what lets a reader that does not know the RLM's outer
    signature tell inputs from outputs.

    The `REPL_ENTRY_KEY` key is always present so an event can be recognized as an RLM event by key alone; it is
    `None` for the extract-fallback event, which records only the final output fields produced by the extract call.
    """
    event: dict[str, Any] = dict(inputs or {})
    event[REPL_ENTRY_KEY] = repl_entry
    event.update(outputs or {})
    return event


def is_repl_event(message: dict[str, Any]) -> bool:
    """Whether `message` was built by `build_repl_event`."""
    return REPL_ENTRY_KEY in message


def split_repl_event(message: dict[str, Any]) -> tuple[dict[str, Any], REPLEntry | None, dict[str, Any]]:
    """Split a conversation-history event built by `build_repl_event` into (inputs, repl_entry, outputs).

    Keys before `REPL_ENTRY_KEY` are inputs and keys after it are outputs. `repl_entry` is `None` for the
    extract-fallback event.

    Raises:
        ValueError: If `message` does not carry `REPL_ENTRY_KEY` (check with `is_repl_event` first).
    """
    names = list(message)
    if REPL_ENTRY_KEY not in message:
        raise ValueError(f"Not an RLM history event: missing `{REPL_ENTRY_KEY}` key. Keys: {names}")
    entry_index = names.index(REPL_ENTRY_KEY)

    repl_entry = message[REPL_ENTRY_KEY]
    inputs = {name: message[name] for name in names[:entry_index]}
    outputs = {name: message[name] for name in names[entry_index + 1 :]}
    return inputs, repl_entry, outputs


class REPLHistory(pydantic.BaseModel):
    """Container for REPL interaction history.

    Immutable: append() returns a new instance with the entry added.
    """

    entries: list[REPLEntry] = Field(default_factory=list)
    max_output_chars: int = 10_000

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(entry.format(index=i) for i, entry in enumerate(self.entries))

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append(self, *, reasoning: str = "", code: str, output: str) -> REPLHistory:
        """Return a new REPLHistory with the entry appended."""
        new_entry = REPLEntry(reasoning=reasoning, code=code, output=output, max_output_chars=self.max_output_chars)
        return REPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[REPLEntry]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
