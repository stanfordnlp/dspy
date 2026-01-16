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
        preview_chars: int = 500,
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
        preview = value_str[:preview_chars] + ("..." if is_truncated else "")

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

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self, index: int, max_output_chars: int = 5000) -> str:
        """Format this entry for inclusion in prompts."""
        output = self.output
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + f"\n... (truncated to {max_output_chars}/{len(self.output):,} chars)"
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return f"=== Step {index + 1} ===\n{reasoning_line}Code:\n```python\n{self.code}\n```\nOutput ({len(self.output):,} chars):\n{output}"


class REPLHistory(pydantic.BaseModel):
    """Container for REPL interaction history.

    Immutable: append() returns a new instance with the entry added.
    """

    entries: list[REPLEntry] = Field(default_factory=list)

    model_config = pydantic.ConfigDict(frozen=True)

    def format(self, max_output_chars: int = 5000) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(entry.format(index=i, max_output_chars=max_output_chars) for i, entry in enumerate(self.entries))

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    def append(self, *, reasoning: str = "", code: str, output: str) -> REPLHistory:
        """Return a new REPLHistory with the entry appended."""
        new_entry = REPLEntry(reasoning=reasoning, code=code, output=output)
        return REPLHistory(entries=list(self.entries) + [new_entry])

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[REPLEntry]:
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
