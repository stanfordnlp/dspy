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
    tool = Tool(name="test_tool", desc="A test tool", parameters={"param1": {"type": "string"}}, func=lambda x: x)
    assert tool.name == "test_tool"
    assert tool.desc == "A test tool"
    assert tool.parameters == {"param1": {"type": "string"}}
    assert callable(tool.func)


def test_from_function():
    tool = Tool.from_function(dummy_function)

    assert tool.name == "dummy_function"
    assert "A dummy function for testing" in tool.desc
    assert "x" in tool.parameters
    assert "y" in tool.parameters
    assert tool.parameters["x"]["type"] == "integer"
    assert tool.parameters["y"]["type"] == "string"


def test_tool_from_class():
    class Foo:
        def __init__(self, user_id: str):
            self.user_id = user_id

        def __call__(self, a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

    tool = Tool.from_function(Foo("123"))
    assert tool.name == "Foo"
    assert tool.desc == "Add two numbers."
    assert tool.parameters == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_from_function_with_pydantic():
    tool = Tool.from_function(dummy_with_pydantic)

    assert tool.name == "dummy_with_pydantic"
    assert "model" in tool.parameters
    assert tool.parameters["model"]["type"] == "object"
    assert "field1" in tool.parameters["model"]["properties"]
    assert "field2" in tool.parameters["model"]["properties"]


def test_convert_to_litellm_tool_format():
    tool = Tool.from_function(dummy_function)
    litellm_format = tool.convert_to_litellm_tool_format()

    assert litellm_format["type"] == "function"
    assert litellm_format["function"]["name"] == "dummy_function"
    assert "parameters" in litellm_format["function"]
    assert litellm_format["function"]["parameters"]["required"] == ["x", "y"]
    assert not litellm_format["function"]["parameters"]["additionalProperties"]


def test_tool_callable():
    tool = Tool.from_function(dummy_function)
    result = tool(x=42, y="hello")
    assert result == "hello 42"


def test_tool_with_pydantic_callable():
    tool = Tool.from_function(dummy_with_pydantic)
    model = DummyModel(field1="test", field2=123)
    result = tool(model=model)
    assert result == "test 123"


def test_invalid_function_call():
    tool = Tool.from_function(dummy_function)
    with pytest.raises(ValueError):
        tool(x="not an integer", y="hello")


def test_complex_nested_schema():
    tool = Tool.from_function(complex_dummy_function)

    assert tool.name == "complex_dummy_function"
    assert "profile" in tool.parameters

    profile_schema = tool.parameters["profile"]
    assert profile_schema["type"] == "object"

    # Check nested structure
    properties = profile_schema["properties"]
    assert "contact" in properties
    assert properties["contact"]["type"] == "object"

    contact_properties = properties["contact"]["properties"]
    assert "addresses" in contact_properties
    assert contact_properties["addresses"]["type"] == "array"
    assert contact_properties["addresses"]["items"]["type"] == "object"

    # Check converted litellm format
    expected_litellm_format = {
        "type": "function",
        "function": {
            "name": "complex_dummy_function",
            "description": (
                "Process user profile with complex nested structure.\n\n    Args:\n        profile: User "
                "profile containing nested contact and address information\n        priority: Priority "
                "level of the processing\n        notes: Optional processing notes\n    "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {
                        "properties": {
                            "user_id": {"title": "User Id", "type": "integer"},
                            "name": {"title": "Name", "type": "string"},
                            "age": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "title": "Age"},
                            "contact": {
                                "properties": {
                                    "email": {"title": "Email", "type": "string"},
                                    "phone": {
                                        "anyOf": [{"type": "string"}, {"type": "null"}],
                                        "default": None,
                                        "title": "Phone",
                                    },
                                    "addresses": {
                                        "items": {
                                            "properties": {
                                                "street": {"title": "Street", "type": "string"},
                                                "city": {"title": "City", "type": "string"},
                                                "zip_code": {"title": "Zip Code", "type": "string"},
                                                "is_primary": {
                                                    "default": False,
                                                    "title": "Is Primary",
                                                    "type": "boolean",
                                                },
                                            },
                                            "required": ["street", "city", "zip_code"],
                                            "title": "Address",
                                            "type": "object",
                                        },
                                        "title": "Addresses",
                                        "type": "array",
                                    },
                                },
                                "required": ["email", "addresses"],
                                "title": "ContactInfo",
                                "type": "object",
                            },
                            "tags": {"default": [], "items": {"type": "string"}, "title": "Tags", "type": "array"},
                        },
                        "required": ["user_id", "name", "contact"],
                        "title": "UserProfile",
                        "type": "object",
                    },
                    "priority": {"type": "integer"},
                    "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["profile", "priority", "notes"],
                "additionalProperties": False,
            },
        },
    }
    assert tool.convert_to_litellm_tool_format() == expected_litellm_format
