"""Protocol for types that can be serialized into an RLM sandbox.

Types implementing this protocol can be injected into the REPL environment
used by dspy.RLM. The protocol defines how to serialize, reconstruct, and
preview the value for LLM consumption.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import Protocol

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from dspy.primitives.repl_types import REPLVariable

__all__ = ["SandboxSerializable"]


@runtime_checkable
class SandboxSerializable(Protocol):
    """Protocol for types that support RLM sandbox injection.

    Implementors define how to serialize their data into the sandbox,
    what setup code (imports) is needed, how to reconstruct the value
    from the serialized payload, and how to produce an LLM-friendly preview.

    Example implementation::

        class DataFrame(SandboxSerializable):
            def sandbox_setup(self) -> str:
                return "import pandas as pd\\nimport base64\\nimport io"

            def to_sandbox(self) -> bytes:
                return base64.b64encode(self.data.to_parquet(index=False))

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"DataFrame: {self.data.shape[0]} rows x {self.data.shape[1]} columns"
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Allow SandboxSerializable subclasses to be used as Pydantic type annotations."""
        return core_schema.no_info_plain_validator_function(
            lambda v: v,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: str(v)),
        )

    def sandbox_setup(self) -> str:
        """Return Python import statements needed in the sandbox."""
        ...

    def to_sandbox(self) -> bytes:
        """Serialize this value for injection into the sandbox.

        Implementations may return either UTF-8 text bytes (e.g., JSON/base64 text)
        or arbitrary binary bytes. Binary payloads are transparently transported
        via base64 before sandbox reconstruction.
        """
        ...

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        """Return Python code that reconstructs this value from a data expression.

        Args:
            var_name: Variable name to assign in the sandbox.
            data_expr: Expression that evaluates to the raw serialized data as
                       ``str`` or ``bytes`` (e.g. ``open('/tmp/dspy_vars/x.json').read()``).
        """
        ...

    def rlm_preview(self, max_chars: int = 500) -> str:
        """Return an LLM-friendly preview of this value."""
        ...

    def to_repl_variable(self, name: str, field_info: FieldInfo | None = None) -> REPLVariable:
        """Build a REPLVariable using rlm_preview().

        Concrete implementations that inherit from SandboxSerializable
        (rather than just structurally matching the Protocol) get this
        method for free.
        """
        from dspy.primitives.repl_types import REPLVariable

        preview = self.rlm_preview()
        var = REPLVariable.from_value(name, self, field_info=field_info)
        return var.model_copy(update={"preview": preview, "total_length": len(preview)})
