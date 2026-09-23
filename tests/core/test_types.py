"""The retired experimental vocabulary must not remain as misleading aliases."""

import importlib

import pytest

import dspy
from dspy.lm15 import Message, Request, Response


def test_old_types_import_explains_migration():
    with pytest.raises(ImportError, match="dspy.lm15"):
        importlib.import_module("dspy.core.types")


def test_old_public_type_names_are_removed():
    for name in ("LMRequest", "LMResponse", "LMMessage", "LMConfig", "System", "User", "Assistant", "Developer", "ToolCall", "ToolResult"):
        assert not hasattr(dspy, name)


def test_bundled_types_are_not_dspy_reimplementations():
    from dspy._vendor import lm15

    assert Request is lm15.Request
    assert Response is lm15.Response
    request = Request(model="example", system="Instructions", messages=(Message.user("Hello"),))
    assert request.messages[0].text == "Hello"
    assert request.system == "Instructions"
