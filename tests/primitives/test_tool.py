import pytest
from pydantic import BaseModel
from dspy.primitives.tool import Tool
from typing import Any


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
    phone: str | None = None
    addresses: list[Address]


class UserProfile(BaseModel):
    user_id: int
    name: str
    age: int | None = None
    contact: ContactInfo
    tags: list[str] = []


def complex_dummy_function(profile: UserProfile, priority: int, notes: str | None = None) -> dict[str, Any]:
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
        "primary_address": primary_address.dict(),
        "notes": notes,
    }


def test_basic_initialization(self):
    tool = Tool(name="test_tool", desc="A test tool", parameters={"param1": {"type": "string"}}, func=lambda x: x)
    assert tool.name == "test_tool"
    assert tool.desc == "A test tool"
    assert tool.parameters == {"param1": {"type": "string"}}
    assert callable(tool.func)


def test_from_function(self):
    tool = Tool.from_function(dummy_function)

    assert tool.name == "dummy_function"
    assert "A dummy function for testing" in tool.desc
    assert "x" in tool.parameters
    assert "y" in tool.parameters
    assert tool.parameters["x"]["type"] == "integer"
    assert tool.parameters["y"]["type"] == "string"


def test_from_function_with_pydantic(self):
    tool = Tool.from_function(dummy_with_pydantic)

    assert tool.name == "dummy_with_pydantic"
    assert "model" in tool.parameters
    assert tool.parameters["model"]["type"] == "object"
    assert "field1" in tool.parameters["model"]["properties"]
    assert "field2" in tool.parameters["model"]["properties"]


def test_convert_to_litellm_tool_format(self):
    tool = Tool.from_function(dummy_function)
    litellm_format = tool.convert_to_litellm_tool_format()

    assert litellm_format["type"] == "function"
    assert litellm_format["function"]["name"] == "dummy_function"
    assert "parameters" in litellm_format["function"]
    assert litellm_format["function"]["parameters"]["required"] == ["x", "y"]
    assert not litellm_format["function"]["parameters"]["additionalProperties"]


def test_tool_callable(self):
    tool = Tool.from_function(dummy_function)
    result = tool(x=42, y="hello")
    assert result == "hello 42"


def test_tool_with_pydantic_callable(self):
    tool = Tool.from_function(dummy_with_pydantic)
    model = DummyModel(field1="test", field2=123)
    result = tool(model=model)
    assert result == "test 123"


def test_invalid_function_call(self):
    tool = Tool.from_function(dummy_function)
    with pytest.raises(TypeError):
        tool(x="not an integer", y="hello")


def test_complex_nested_schema(self):
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
            "description": "Process user profile with complex nested structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer"},
                            "name": {"type": "string"},
                            "age": {"type": "integer", "nullable": True},
                            "contact": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string"},
                                    "phone": {"type": "string", "nullable": True},
                                    "addresses": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "street": {"type": "string"},
                                                "city": {"type": "string"},
                                                "zip_code": {"type": "string"},
                                                "is_primary": {"type": "boolean", "default": False},
                                            },
                                            "required": ["street", "city", "zip_code"],
                                        },
                                    },
                                },
                                "required": ["email", "addresses"],
                            },
                            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                        },
                        "required": ["user_id", "name", "contact"],
                    },
                    "priority": {"type": "integer"},
                    "notes": {"type": "string", "nullable": True},
                },
                "required": ["profile", "priority"],
                "additionalProperties": False,
            },
        },
    }
    assert tool.convert_to_litellm_tool_format() == expected_litellm_format
