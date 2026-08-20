"""Tests for the SandboxSerializable ABC and build_repl_variable helper."""

import pytest

import dspy
from dspy.primitives.repl_types import REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable

# -- Stub implementations --


class ExampleSerializable(SandboxSerializable):
    """A complete subclass of the SandboxSerializable ABC."""

    def __init__(self, data: str = "example_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        preview = f"ExampleData: {self.data}"
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview


class IncompleteSerializable(SandboxSerializable):
    """Missing the other three abstract methods — should not be instantiable."""

    def sandbox_setup(self) -> str:
        return ""


class NotASubclass:
    """Implements all four methods but does not subclass — should NOT pass isinstance."""

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return b""

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return "NotASubclass"


# -- ABC enforcement and isinstance() --


class TestABCConformance:
    def test_subclass_conformance(self):
        assert isinstance(ExampleSerializable(), SandboxSerializable)

    def test_incomplete_subclass_cannot_instantiate(self):
        with pytest.raises(TypeError, match="abstract"):
            IncompleteSerializable()  # type: ignore[abstract]

    def test_structural_conformance_no_longer_accepted(self):
        """Nominal typing only — duck-typed classes must explicitly subclass."""
        assert not isinstance(NotASubclass(), SandboxSerializable)
        assert not isinstance("hello", SandboxSerializable)


# -- Core methods on a conforming implementation --


class TestCoreMethods:
    """Smoke tests that a conforming implementation behaves as expected."""

    def test_sandbox_setup(self):
        assert ExampleSerializable().sandbox_setup() == "import json"

    def test_to_sandbox_returns_bytes(self):
        payload = ExampleSerializable("hello").to_sandbox()
        assert isinstance(payload, bytes)
        assert payload == b"hello"

    def test_sandbox_assignment(self):
        code = ExampleSerializable().sandbox_assignment("my_var", "raw_data")
        assert "my_var" in code
        assert "raw_data" in code
        assert "json.loads" in code

    def test_rlm_preview(self):
        preview = ExampleSerializable("test_value").rlm_preview()
        assert "ExampleData" in preview
        assert "test_value" in preview

    def test_rlm_preview_truncation(self):
        preview = ExampleSerializable("x" * 1000).rlm_preview(max_chars=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")


# -- build_repl_variable helper --


class TestBuildReplVariable:
    """Tests for the module-level helper function."""

    def test_builds_variable_with_preview_from_rlm_preview(self):
        obj = ExampleSerializable("my_data")
        var = build_repl_variable(obj, "my_var")
        assert isinstance(var, REPLVariable)
        assert var.name == "my_var"
        assert var.preview == obj.rlm_preview()
        assert var.total_length == len(obj.rlm_preview())

    def test_surfaces_sandbox_setup_in_description(self):
        """sandbox_setup imports should appear in the variable description
        so the model knows which names are bound in the REPL."""
        var = build_repl_variable(ExampleSerializable(), "x")
        assert "import json" in var.desc

    def test_passes_field_info_through(self):
        field = dspy.InputField(desc="A data column")
        var = build_repl_variable(ExampleSerializable(), "data", field_info=field)
        assert "A data column" in var.desc
        # Setup note still appended after user-provided desc
        assert "import json" in var.desc


class TestToReplVariableMethod:
    """Tests for the inherited default to_repl_variable() method."""

    def test_default_to_repl_variable(self):
        obj = ExampleSerializable("payload")
        var = obj.to_repl_variable("data")
        assert isinstance(var, REPLVariable)
        assert "ExampleData: payload" in var.preview


class TestSignatureAnnotation:
    """Subclasses should be usable as dspy.Signature field annotations."""

    def test_subclass_supports_signature_annotation(self):
        class ExampleSignature(dspy.Signature):
            data: ExampleSerializable = dspy.InputField()
            answer: str = dspy.OutputField()

        assert ExampleSignature.input_fields["data"].annotation is ExampleSerializable
