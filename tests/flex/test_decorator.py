"""Tests for the ``@dspy.flex`` decorator and functional ``dspy.Flex`` form."""

from __future__ import annotations

import dspy
from dspy.flex import Flex, flex


def test_bare_decorator_returns_factory_preserving_signature() -> None:
    @flex
    class MySig(dspy.Signature):
        """Do a thing."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    assert MySig.signature is not None
    # The factory exposes the original Signature class via .signature.
    assert MySig.signature.__name__ == "MySig"
    assert "q" in MySig.signature.input_fields
    assert "a" in MySig.signature.output_fields


def test_decorator_with_args_attaches_persist_to() -> None:
    @flex(persist_to="/tmp/never_written.py", flex_id="custom-id")
    class MySig(dspy.Signature):
        """Trivial."""

        x: str = dspy.InputField()
        y: str = dspy.OutputField()

    assert MySig._flex_persist_to == "/tmp/never_written.py"
    assert MySig._flex_id == "custom-id"


def test_flex_class_is_a_dspy_module() -> None:
    # Constructing a Flex with persist_to=None and no LM raises during codegen,
    # but the class itself must inherit from dspy.Module.
    assert issubclass(Flex, dspy.Module)
