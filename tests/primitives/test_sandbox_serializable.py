"""Tests for the SandboxSerializable protocol."""

import pytest

from dspy.primitives.sandbox_serializable import SandboxSerializable
from dspy.primitives.repl_types import REPLVariable


# -- Stub implementation for testing --


class StubSerializable:
    """Minimal implementation that structurally matches SandboxSerializable."""

    def __init__(self, data: str = "test_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        preview = f"StubData: {self.data}"
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview


class InheritingSerializable(SandboxSerializable):
    """Implementation that inherits from SandboxSerializable to get to_repl_variable()."""

    def __init__(self, data: str = "inherited_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        preview = f"InheritedData: {self.data}"
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview


class NotSerializable:
    """Type that doesn't implement the protocol."""

    def sandbox_setup(self) -> str:
        return ""


# -- Protocol isinstance() --


class TestProtocolConformance:
    """Test that isinstance() checks work correctly."""

    def test_stub_is_not_instance_without_to_repl_variable(self):
        """Structural conformance requires all methods including to_repl_variable."""
        obj = StubSerializable()
        # StubSerializable lacks to_repl_variable, so it doesn't match.
        # This is by design: use InheritingSerializable to get the default impl.
        assert not isinstance(obj, SandboxSerializable)

    def test_inheriting_is_instance(self):
        """Direct inheritance: InheritingSerializable is an instance."""
        obj = InheritingSerializable()
        assert isinstance(obj, SandboxSerializable)

    def test_non_conforming_is_not_instance(self):
        """NotSerializable doesn't implement all methods."""
        obj = NotSerializable()
        assert not isinstance(obj, SandboxSerializable)

    def test_plain_string_is_not_instance(self):
        """Built-in types don't match the protocol."""
        assert not isinstance("hello", SandboxSerializable)


# -- Protocol methods --


class TestProtocolMethods:
    """Test calling protocol methods on conforming types."""

    def test_sandbox_setup(self):
        obj = StubSerializable()
        assert obj.sandbox_setup() == "import json"

    def test_to_sandbox_returns_bytes(self):
        obj = StubSerializable("hello")
        payload = obj.to_sandbox()
        assert isinstance(payload, bytes)
        assert payload == b"hello"

    def test_sandbox_assignment(self):
        obj = StubSerializable()
        code = obj.sandbox_assignment("my_var", "raw_data")
        assert "my_var" in code
        assert "raw_data" in code
        assert "json.loads" in code

    def test_rlm_preview(self):
        obj = StubSerializable("test_value")
        preview = obj.rlm_preview()
        assert "StubData" in preview
        assert "test_value" in preview

    def test_rlm_preview_truncation(self):
        obj = StubSerializable("x" * 1000)
        preview = obj.rlm_preview(max_chars=50)
        assert len(preview) <= 53  # 50 + "..."
        assert preview.endswith("...")


# -- to_repl_variable() --


class TestToReplVariable:
    """Test the derived to_repl_variable() method."""

    def test_inheriting_has_to_repl_variable(self):
        """InheritingSerializable gets to_repl_variable from SandboxSerializable."""
        obj = InheritingSerializable("my_data")
        var = obj.to_repl_variable("my_var")
        assert isinstance(var, REPLVariable)
        assert var.name == "my_var"
        assert "InheritedData" in var.preview
        assert var.total_length == len(obj.rlm_preview())

    def test_to_repl_variable_with_field_info(self):
        """to_repl_variable passes field_info through."""
        import dspy

        field = dspy.InputField(desc="A data column")
        obj = InheritingSerializable()
        var = obj.to_repl_variable("data", field_info=field)
        assert var.desc == "A data column"

    def test_to_repl_variable_uses_rlm_preview(self):
        """The preview in REPLVariable should come from rlm_preview(), not default."""
        obj = InheritingSerializable("custom_preview_data")
        var = obj.to_repl_variable("x")
        expected_preview = obj.rlm_preview()
        assert var.preview == expected_preview
