"""Abstract base class for RLM sandbox-serializable types.

Types that subclass :class:`SandboxSerializable` can flow between the host
and the REPL environment used by :class:`dspy.RLM` in **both** directions:

- **As an input** (``dspy.InputField``): the host wraps a value in a subclass
  instance, the framework serializes it via ``to_sandbox()``, ships the
  payload into the sandbox, and reconstructs it as a native Python object
  via ``sandbox_assignment()``. The agent's code sees the underlying value
  bound to the field name.

- **As an output** (``dspy.OutputField``): the agent's code passes the
  underlying value to ``SUBMIT(...)``. The framework injects a wrapper
  around ``SUBMIT`` that converts the bare value to a wire string via
  ``sandbox_serialize_code()`` before raising ``FinalOutput``. On the host
  side, ``from_sandbox()`` reconstructs the wrapper instance from the wire
  payload.

Subclasses implement one shared setup classmethod, three required methods
for the host -> sandbox direction, and two optional methods for the
sandbox -> host direction. The two output methods default to
``NotImplementedError`` so input-only subclasses keep working — the error
only fires when the type is actually used as an ``OutputField``
annotation. Implement them to opt in to OutputField support. Subclasses
also inherit:

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
and ``sandbox_serialize_code()``.
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
    """Abstract base for types that round-trip through the RLM sandbox.

    Shared sandbox setup — classmethod:

    - ``sandbox_setup``: Python statements (usually imports) executed once
      in the sandbox for both inputs and output annotations. The returned
      text is also surfaced to the LLM in the variable description for
      inputs, so the model knows which names are in scope (e.g. ``pd``
      when pandas is imported).

    Host -> sandbox (input) — instance methods:

    - ``to_sandbox``: serialize the wrapped value to text bytes or binary
      bytes for transport.
    - ``sandbox_assignment``: Python code that reconstructs the value from
      a data expression on the sandbox side.
    - ``rlm_preview``: a short, LLM-friendly description of the value.

    Sandbox -> host (output) — classmethods, optional:

    - ``sandbox_serialize_code``: Python expression (a *string*) that, when
      evaluated inside the sandbox, converts the bare value (bound to
      ``var_name``) into a JSON-serializable wire payload — typically a
      base64-encoded string. The expression runs after ``sandbox_setup()``
      has executed, so it can use names imported by setup.
    - ``from_sandbox``: reconstruct a wrapper instance on the host from
      the wire payload.

    Both default to raising ``NotImplementedError`` so subclasses written
    against the input-only API keep working as ``InputField`` annotations.
    Implement them to make a subclass usable as an ``OutputField``.

    Example::

        class DataFrame(SandboxSerializable):
            def __init__(self, df):
                self.data = df

            @classmethod
            def sandbox_setup(cls) -> str:
                return "import pandas as pd\\nimport pyarrow\\nimport base64\\nimport io"

            # host -> sandbox
            def to_sandbox(self) -> bytes:
                return base64.b64encode(self.data.to_parquet(index=False))

            def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
                return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

            def rlm_preview(self, max_chars: int = 500) -> str:
                return f"DataFrame: {self.data.shape[0]} rows x {self.data.shape[1]} columns"

            # sandbox -> host
            @classmethod
            def sandbox_serialize_code(cls, var_name: str) -> str:
                return f"base64.b64encode({var_name}.to_parquet(index=False)).decode('ascii')"

            @classmethod
            def from_sandbox(cls, payload: str) -> "DataFrame":
                import base64, io
                import pandas as pd
                return cls(pd.read_parquet(io.BytesIO(base64.b64decode(payload))))

    Subclasses can be used directly as :class:`dspy.Signature` field
    annotations because of the inherited ``__get_pydantic_core_schema__``
    hook (see the module docstring for what that hook does and why it is
    needed).
    """

    @classmethod
    @abstractmethod
    def sandbox_setup(cls) -> str: ...

    @abstractmethod
    def to_sandbox(self) -> bytes: ...

    @abstractmethod
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str: ...

    @abstractmethod
    def rlm_preview(self, max_chars: int = 500) -> str: ...

    @classmethod
    def sandbox_serialize_code(cls, var_name: str) -> str:
        """Python expression that converts a bare value at ``var_name`` to a wire payload.

        Optional: implement to support use as an ``OutputField`` annotation.
        The expression is evaluated inside the sandbox by the generated
        SUBMIT wrapper. It must produce a JSON-serializable value
        (typically a base64-encoded string). Shared ``sandbox_setup()`` runs
        before the REPL loop, so the expression can use names imported by
        setup.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support OutputField usage. "
            f"Implement sandbox_serialize_code and from_sandbox to opt in."
        )

    @classmethod
    def from_sandbox(cls, payload: Any) -> SandboxSerializable:
        """Reconstruct a wrapper instance on the host from the wire payload.

        Optional: implement to support use as an ``OutputField`` annotation.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support OutputField usage. "
            f"Implement sandbox_serialize_code and from_sandbox to opt in."
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Pass-through schema so subclasses work as ``dspy.Signature`` annotations.

        ``json_schema_input_schema`` keeps JSON-schema generation happy for
        OutputField usage (RLM materialises field schemas while building
        action signatures) — RLM only ever needs the type *name* anyway, so
        ``any_schema`` is the right semantic.
        """
        return core_schema.no_info_plain_validator_function(
            lambda v: v,
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: str(v)),
            json_schema_input_schema=core_schema.any_schema(),
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
