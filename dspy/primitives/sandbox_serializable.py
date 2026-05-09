"""Abstract base class for RLM sandbox-serializable types.

Types that subclass :class:`SandboxSerializable` can be injected into the
REPL environment used by :class:`dspy.RLM`. Subclasses implement four
abstract methods describing how a value enters and appears inside the
sandbox, and inherit:

- ``__get_pydantic_core_schema__`` so the type can be used directly as a
  :class:`dspy.Signature` field annotation (see "The pydantic hook" below).
- ``to_repl_variable()`` as a default helper that delegates to the free
  :func:`build_repl_variable` function.

The free function :func:`build_repl_variable` is also exported for the rare
case where you need to wrap a value without subclassing.

The pydantic hook
-----------------

``__get_pydantic_core_schema__`` lets subclasses be used as
``dspy.Signature`` field annotations. It is a pass-through (no validation,
``str()`` serialization) — RLM owns real serialization via ``to_sandbox()``
and ``sandbox_assignment()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from dspy.primitives.repl_types import REPLVariable

__all__ = ["SandboxSerializable", "build_repl_variable"]


class SandboxSerializable(ABC):
    """Abstract base for types that support RLM sandbox injection.

    Subclasses implement four methods:

    - ``sandbox_setup``: Python statements (usually imports) executed once
      in the sandbox. The returned text is also surfaced to the LLM in the
      variable description, so the model knows which names are in scope
      (e.g. ``pd`` when pandas is imported).
    - ``to_sandbox``: serialize the value to text bytes or binary bytes.
    - ``sandbox_assignment``: Python code that reconstructs the value from
      a data expression.
    - ``rlm_preview``: a short, LLM-friendly description of the value.

    Example::

        class DataFrame(SandboxSerializable):
            def __init__(self, df):
                self.data = df

            def sandbox_setup(self) -> str:
                return "import pandas as pd\\nimport base64\\nimport io"

            def to_sandbox(self) -> bytes:
                return base64.b64encode(self.data.to_parquet(index=False))

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"DataFrame: {self.data.shape[0]} rows x {self.data.shape[1]} columns"

    Subclasses can be used directly as :class:`dspy.Signature` field
    annotations because of the inherited ``__get_pydantic_core_schema__``
    hook (see the module docstring for what that hook does and why it is
    needed).
    """

    @abstractmethod
    def sandbox_setup(self) -> str: ...

    @abstractmethod
    def to_sandbox(self) -> bytes: ...

    @abstractmethod
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str: ...

    @abstractmethod
    def rlm_preview(self, max_chars: int = 500) -> str: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Pass-through schema so subclasses work as ``dspy.Signature`` annotations."""
        return core_schema.no_info_plain_validator_function(
            lambda v: v,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: str(v)),
        )

    def to_repl_variable(self, name: str, field_info: FieldInfo | None = None) -> REPLVariable:
        """Build a REPLVariable using the default RLM helper."""
        return build_repl_variable(self, name, field_info=field_info)


def build_repl_variable(
    obj: SandboxSerializable,
    name: str,
    field_info: FieldInfo | None = None,
) -> REPLVariable:
    """Build a :class:`REPLVariable` for a SandboxSerializable value.

    Free function form of :meth:`SandboxSerializable.to_repl_variable`. Use
    it when you need to wrap a value imperatively without going through the
    method on the instance. The resulting variable's preview comes from
    ``rlm_preview()``; ``sandbox_setup()`` imports are appended to the
    description so the model learns which names are bound in the sandbox
    before it writes code.
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
