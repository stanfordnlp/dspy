"""Protocol and ergonomic base class for RLM sandbox-serializable types.

Types implementing :class:`SandboxSerializable` can be injected into the REPL
environment used by :class:`dspy.RLM`. The protocol is intentionally minimal:
four methods describing how a value enters and appears inside the sandbox.
No inheritance is required — because the protocol is ``@runtime_checkable``,
``isinstance(obj, SandboxSerializable)`` returns True as soon as the four
methods are present.

For users who want the ergonomic path of "subclass one thing and it works",
:class:`SandboxSerializableBase` provides the Pydantic hook needed for direct
``dspy.Signature`` annotations plus a default ``to_repl_variable()`` helper.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast, runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing_extensions import Protocol

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from dspy.primitives.repl_types import REPLVariable

__all__ = ["SandboxSerializable", "SandboxSerializableBase", "build_repl_variable"]


@runtime_checkable
class SandboxSerializable(Protocol):
    """Protocol for types that support RLM sandbox injection.

    Implementors define four things:

    - ``sandbox_setup``: Python statements (usually imports) executed once
      in the sandbox. The returned text is also surfaced to the LLM in the
      variable description, so the model knows which names are in scope
      (e.g. ``pd`` when pandas is imported).
    - ``to_sandbox``: serialize the value to text bytes or binary bytes.
    - ``sandbox_assignment``: Python code that reconstructs the value from
      a data expression.
    - ``rlm_preview``: a short, LLM-friendly description of the value.

    Example::

        class DataFrame:
            def sandbox_setup(self) -> str:
                return "import pandas as pd\\nimport base64\\nimport io"

            def to_sandbox(self) -> bytes:
                return base64.b64encode(self.data.to_parquet(index=False))

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"DataFrame: {self.data.shape[0]} rows x {self.data.shape[1]} columns"

        assert isinstance(DataFrame(...), SandboxSerializable)  # structural

    To use your type directly as a :class:`dspy.Signature` input annotation
    (e.g. ``df: MyType = dspy.InputField(...)``), also define a
    ``__get_pydantic_core_schema__`` classmethod on your type, or subclass
    :class:`SandboxSerializableBase` for the default implementation.
    """

    def sandbox_setup(self) -> str: ...
    def to_sandbox(self) -> bytes: ...
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str: ...
    def rlm_preview(self, max_chars: int = 500) -> str: ...


class SandboxSerializableBase:
    """Optional base class for ergonomic SandboxSerializable implementations.

    Subclass this when you want three things together:
    - structural conformance to :class:`SandboxSerializable`
    - direct use as a :class:`dspy.Signature` field annotation
    - a default ``to_repl_variable()`` implementation

    Users who prefer duck typing can skip this base class entirely and just
    implement the four protocol methods.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Allow subclasses to be used as Pydantic type annotations."""
        return core_schema.no_info_plain_validator_function(
            lambda v: v,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: str(v)),
        )

    def to_repl_variable(self, name: str, field_info: "FieldInfo | None" = None) -> "REPLVariable":
        """Build a REPLVariable using the default RLM helper."""
        return build_repl_variable(cast(SandboxSerializable, self), name, field_info=field_info)


def build_repl_variable(
    obj: SandboxSerializable,
    name: str,
    field_info: "FieldInfo | None" = None,
) -> "REPLVariable":
    """Build a :class:`REPLVariable` for a SandboxSerializable value.

    This is a free function, not a protocol method, so users never have to
    subclass just to get default behavior. The resulting variable's preview
    comes from ``rlm_preview()``; ``sandbox_setup()`` imports are appended
    to the description so the model learns which names are bound in the
    sandbox before it writes code.
    """
    from dspy.primitives.repl_types import REPLVariable

    preview = obj.rlm_preview()
    var = REPLVariable.from_value(name, obj, field_info=field_info)
    setup = obj.sandbox_setup().strip()
    desc = var.desc
    if setup:
        setup_note = f"Sandbox imports available:\n{setup}"
        desc = f"{desc}\n{setup_note}" if desc else setup_note
    return var.model_copy(update={"preview": preview, "total_length": len(preview), "desc": desc})
