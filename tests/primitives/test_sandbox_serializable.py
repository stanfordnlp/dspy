"""Tests for the SandboxSerializable protocol and build_repl_variable helper."""

import dspy

from dspy.primitives.repl_types import REPLVariable
from dspy.primitives.sandbox_serializable import SandboxSerializable, SandboxSerializableBase, build_repl_variable


# -- Stub implementations --


class StructuralSerializable:
    """Structurally conforms to SandboxSerializable without subclassing."""

    def __init__(self, data: str = "struct_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        preview = f"StructData: {self.data}"
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview


class BaseSerializable(SandboxSerializableBase):
    """Uses the ergonomic base class for annotation support and defaults."""

    def __init__(self, data: str = "inherited_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return f"InheritedData: {self.data}"


class NotConforming:
    def sandbox_setup(self) -> str:
        return ""


# -- runtime_checkable isinstance() --


class TestProtocolConformance:
    """Structural and subclass isinstance() checks both work."""

    def test_structural_conformance(self):
        assert isinstance(StructuralSerializable(), SandboxSerializable)

    def test_subclass_conformance(self):
        assert isinstance(BaseSerializable(), SandboxSerializable)

    def test_missing_methods_rejected(self):
        assert not isinstance(NotConforming(), SandboxSerializable)
        assert not isinstance("hello", SandboxSerializable)


# -- Core methods on a conforming implementation --


class TestCoreMethods:
    """Smoke tests that a conforming implementation behaves as expected."""

    def test_sandbox_setup(self):
        assert StructuralSerializable().sandbox_setup() == "import json"

    def test_to_sandbox_returns_bytes(self):
        payload = StructuralSerializable("hello").to_sandbox()
        assert isinstance(payload, bytes)
        assert payload == b"hello"

    def test_sandbox_assignment(self):
        code = StructuralSerializable().sandbox_assignment("my_var", "raw_data")
        assert "my_var" in code
        assert "raw_data" in code
        assert "json.loads" in code

    def test_rlm_preview(self):
        preview = StructuralSerializable("test_value").rlm_preview()
        assert "StructData" in preview
        assert "test_value" in preview

    def test_rlm_preview_truncation(self):
        preview = StructuralSerializable("x" * 1000).rlm_preview(max_chars=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")


# -- build_repl_variable helper --


class TestBuildReplVariable:
    """Tests for the module-level helper function."""

    def test_builds_variable_with_preview_from_rlm_preview(self):
        obj = StructuralSerializable("my_data")
        var = build_repl_variable(obj, "my_var")
        assert isinstance(var, REPLVariable)
        assert var.name == "my_var"
        assert var.preview == obj.rlm_preview()
        assert var.total_length == len(obj.rlm_preview())

    def test_works_for_subclass_implementation(self):
        obj = BaseSerializable("other")
        var = build_repl_variable(obj, "x")
        assert "InheritedData: other" in var.preview

    def test_surfaces_sandbox_setup_in_description(self):
        """sandbox_setup imports should appear in the variable description
        so the model knows which names are bound in the REPL."""
        var = build_repl_variable(StructuralSerializable(), "x")
        assert "import json" in var.desc

    def test_passes_field_info_through(self):
        field = dspy.InputField(desc="A data column")
        var = build_repl_variable(StructuralSerializable(), "data", field_info=field)
        assert "A data column" in var.desc
        # Setup note still appended after user-provided desc
        assert "import json" in var.desc


class TestSandboxSerializableBase:
    """Tests for the ergonomic base class."""

    def test_base_exports_default_to_repl_variable(self):
        obj = BaseSerializable("payload")
        var = obj.to_repl_variable("data")
        assert isinstance(var, REPLVariable)
        assert "InheritedData: payload" in var.preview

    def test_base_supports_signature_annotations_without_extra_schema_hook(self):
        class ExampleSignature(dspy.Signature):
            data: BaseSerializable = dspy.InputField()
            answer: str = dspy.OutputField()

        assert ExampleSignature.input_fields["data"].annotation is BaseSerializable
