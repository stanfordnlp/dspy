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


def _longest_backtick_run(text: str) -> int:
    longest = 0
    current = 0
    for char in text:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _markdown_block(text: str, language: str = "") -> str:
    fence = "`" * max(3, _longest_backtick_run(text) + 1)
    return f"{fence}{language}\n{text}\n{fence}"


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
        if preview_chars <= 0:
            raise ValueError("preview_chars must be greater than 0")

        jsonable = serialize_for_json(value)
        if isinstance(jsonable, (dict, list)):
            value_str = json.dumps(jsonable, indent=2)
        else:
            value_str = str(jsonable)
        is_truncated = len(value_str) > preview_chars
        if is_truncated:
            head_chars = preview_chars // 2
            tail_chars = preview_chars - head_chars
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
        lines.append(f"Preview:\n{_markdown_block(self.preview)}")
        return "\n".join(lines)

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()


class REPLEntry(pydantic.BaseModel):
    """A single REPL interaction entry containing reasoning, code, and output."""

    reasoning: str = ""
    code: str
    output: str

    model_config = pydantic.ConfigDict(frozen=True)

    @staticmethod
    def format_output(output: str, max_output_chars: int = 10_000) -> str:
        """Format output with head+tail truncation, preserving true length in header."""
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than 0")

        raw_len = len(output)
        if raw_len > max_output_chars:
            head_chars = max_output_chars // 2
            tail_chars = max_output_chars - head_chars
            omitted = raw_len - max_output_chars
            output = output[:head_chars] + f"\n\n... ({omitted:,} characters omitted) ...\n\n" + output[-tail_chars:]
        return f"Output ({raw_len:,} chars):\n{_markdown_block(output)}"

    def format(self, index: int, max_output_chars: int = 10_000) -> str:
        """Format this entry for inclusion in prompts."""
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        code_block = _markdown_block(self.code, "python")
        return f"=== Step {index + 1} ===\n{reasoning_line}Code:\n{code_block}\n{self.format_output(self.output, max_output_chars)}"


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
        return "\n".join(entry.format(index=i, max_output_chars=self.max_output_chars) for i, entry in enumerate(self.entries))

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append(self, *, reasoning: str = "", code: str, output: str) -> REPLHistory:
        """Return a new REPLHistory with the entry appended."""
        new_entry = REPLEntry(reasoning=reasoning, code=code, output=output)
        return REPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[REPLEntry]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
