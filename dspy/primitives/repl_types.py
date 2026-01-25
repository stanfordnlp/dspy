"""
REPL data types for RLM and interpreter interactions.

These types represent the state and history of REPL-based execution:
- REPLVariable: Metadata about variables available in the REPL
- REPLEntry: A single interaction (reasoning, code, output)
- REPLHistory: Container for the full interaction history
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Iterator

import pydantic
from pydantic import Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from dspy.adapters.utils import serialize_for_json

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

__all__ = ["REPLVariable", "REPLEntry", "REPLHistory", "DataFrame"]


class _DataFrameAnnotation:
    """Pydantic-compatible annotation for pandas DataFrames.

    This class provides the __get_pydantic_core_schema__ method that tells
    Pydantic how to validate and serialize pandas DataFrames, without
    requiring arbitrary_types_allowed=True globally.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Return a Pydantic core schema that accepts any DataFrame instance."""

        def validate_dataframe(value: Any) -> Any:
            if not _is_dataframe(value):
                raise ValueError(f"Expected a pandas DataFrame, got {type(value).__name__}")
            return value

        return core_schema.no_info_plain_validator_function(
            validate_dataframe,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x.to_dict(orient="records"),
                info_arg=False,
            ),
        )


# Type alias for use in DSPy Signatures: `dataframe: dspy.DataFrame = dspy.InputField()`
# This uses Annotated to attach the Pydantic schema handler to the pandas DataFrame type
try:
    import pandas as pd
    DataFrame = Annotated[pd.DataFrame, _DataFrameAnnotation()]
except ImportError:
    # If pandas is not installed, create a placeholder that will fail gracefully
    DataFrame = Any


def _is_dataframe(value: Any) -> bool:
    """Check if value is a pandas DataFrame without requiring pandas import."""
    type_module = getattr(type(value), "__module__", "")
    type_name = type(value).__name__
    return type_module.startswith("pandas") and type_name == "DataFrame"


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
        # Handle DataFrames specially with rich metadata
        if _is_dataframe(value):
            return cls._from_dataframe(name, value, field_info)

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

    @classmethod
    def _from_dataframe(
        cls,
        name: str,
        df: Any,
        field_info: FieldInfo | None = None,
    ) -> REPLVariable:
        """Create REPLVariable with rich DataFrame metadata.

        Args:
            name: Variable name
            df: A pandas DataFrame
            field_info: Optional pydantic FieldInfo with desc/constraints metadata
        """
        shape = df.shape

        # Build column info with dtypes and null counts
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            null_info = f" ({null_count:,} nulls)" if null_count > 0 else ""
            columns_info.append(f"  - {col}: {dtype}{null_info}")

        # Calculate memory usage
        memory_bytes = df.memory_usage(deep=True).sum()
        if memory_bytes >= 1024 * 1024:
            memory_str = f"{memory_bytes / (1024 * 1024):.1f} MB"
        else:
            memory_str = f"{memory_bytes / 1024:.1f} KB"

        # Build informative preview
        preview_lines = [
            f"Shape: {shape[0]:,} rows x {shape[1]} columns",
            f"Memory: {memory_str}",
            "",
            "Columns:",
            *columns_info,
            "",
            "First 3 rows:",
            df.head(3).to_string(),
        ]

        if shape[0] > 6:
            preview_lines.extend([
                "",
                "Last 3 rows:",
                df.tail(3).to_string(),
            ])

        preview = "\n".join(preview_lines)

        # Extract desc and constraints from field_info if provided
        desc = ""
        constraints = ""
        if field_info and hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            raw_desc = field_info.json_schema_extra.get("desc", "")
            if raw_desc and not raw_desc.startswith("${"):
                desc = raw_desc
            constraints = field_info.json_schema_extra.get("constraints", "")

        return cls(
            name=name,
            type_name="DataFrame",
            desc=desc,
            constraints=constraints,
            total_length=shape[0],  # Use row count as "length" for DataFrames
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

        # Use appropriate label for total_length based on type
        if self.type_name == "DataFrame":
            lines.append(f"Total rows: {self.total_length:,}")
        else:
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
