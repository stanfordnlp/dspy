import pytest
from pydantic import BaseModel
from dspy.primitives.tool import Tool
from typing import Any, Optional


# Test fixtures
def dummy_function(x: int, y: str) -> str:
    """A dummy function for testing.

    Args:
        x: An integer parameter
        y: A string parameter
    """
    return f"{y} {x}"


class DummyModel(BaseModel):
    field1: str
    field2: int


def dummy_with_pydantic(model: DummyModel) -> str:
    """A dummy function that accepts a Pydantic model."""
    return f"{model.field1} {model.field2}"


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    is_primary: bool = False


class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    addresses: list[Address]


class UserProfile(BaseModel):
    user_id: int
    name: str
    age: Optional[int] = None
    contact: ContactInfo
    tags: list[str] = []


def complex_dummy_function(profile: UserProfile, priority: int, notes: Optional[str] = None) -> dict[str, Any]:
    """Process user profile with complex nested structure.

    Args:
        profile: User profile containing nested contact and address information
        priority: Priority level of the processing
        notes: Optional processing notes
    """
    primary_address = next(
        (addr for addr in profile.contact.addresses if addr.is_primary), profile.contact.addresses[0]
    )

    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "priority": priority,
        "primary_address": primary_address.model_dump(),
        "notes": notes,
    }


def test_basic_initialization():
    tool = Tool(name="test_tool", desc="A test tool", args={"param1": {"type": "string"}}, func=lambda x: x)
    assert tool.name == "test_tool"
    assert tool.desc == "A test tool"
    assert tool.args == {"param1": {"type": "string"}}
    assert callable(tool.func)


def test_tool_from_function():
    tool = Tool(dummy_function)

    assert tool.name == "dummy_function"
    assert "A dummy function for testing" in tool.desc
    assert "x" in tool.args
    assert "y" in tool.args
    assert tool.args["x"]["type"] == "integer"
    assert tool.args["y"]["type"] == "string"


def test_tool_from_class():
    class Foo:
        def __init__(self, user_id: str):
            self.user_id = user_id

        def __call__(self, a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

    tool = Tool(Foo("123"))
    assert tool.name == "Foo"
    assert tool.desc == "Add two numbers."
    assert tool.args == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_tool_from_function_with_pydantic():
    tool = Tool(dummy_with_pydantic)

    assert tool.name == "dummy_with_pydantic"
    assert "model" in tool.args
    assert tool.args["model"]["type"] == "object"
    assert "field1" in tool.args["model"]["properties"]
    assert "field2" in tool.args["model"]["properties"]


def test_tool_callable():
    tool = Tool(dummy_function)
    result = tool(x=42, y="hello")
    assert result == "hello 42"


def test_tool_with_pydantic_callable():
    tool = Tool(dummy_with_pydantic)
    model = DummyModel(field1="test", field2=123)
    result = tool(model=model)
    assert result == "test 123"


def test_invalid_function_call():
    tool = Tool(dummy_function)
    with pytest.raises(ValueError):
        tool(x="not an integer", y="hello")


def test_parameter_desc():
    tool = Tool(dummy_function, arg_desc={"x": "The x parameter"})
    assert tool.args["x"]["description"] == "The x parameter"
