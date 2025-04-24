import pytest
from pydantic import BaseModel
from dspy.primitives.tool import Tool
from typing import Any, Optional
import asyncio


# Test fixtures
def dummy_function(x: int, y: str = "hello") -> str:
    """A dummy function for testing.

    Args:
        x: An integer parameter
        y: A string parameter
    """
    return f"{y} {x}"


class DummyModel(BaseModel):
    field1: str = "hello"
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


async def async_dummy_function(x: int, y: str = "hello") -> str:
    """An async dummy function for testing.

    Args:
        x: An integer parameter
        y: A string parameter
    """
    await asyncio.sleep(0.1)  # Simulate some async work
    return f"{y} {x}"


async def async_dummy_with_pydantic(model: DummyModel) -> str:
    """An async dummy function that accepts a Pydantic model."""
    await asyncio.sleep(0.1)  # Simulate some async work
    return f"{model.field1} {model.field2}"


async def async_complex_dummy_function(
    profile: UserProfile, priority: int, notes: Optional[str] = None
) -> dict[str, Any]:
    """Process user profile with complex nested structure asynchronously.

    Args:
        profile: User profile containing nested contact and address information
        priority: Priority level of the processing
        notes: Optional processing notes
    """
    # Simulate some async processing work
    await asyncio.sleep(0.1)

    primary_address = next(
        (addr for addr in profile.contact.addresses if addr.is_primary), profile.contact.addresses[0]
    )

    # Simulate more async work after finding primary address
    await asyncio.sleep(0.1)

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
    assert tool.args["y"]["default"] == "hello"


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
    assert tool.args["model"]["properties"]["field1"]["default"] == "hello"


def test_tool_from_function_with_pydantic_nesting():
    tool = Tool(complex_dummy_function)

    assert tool.name == "complex_dummy_function"
    assert "profile" in tool.args
    assert "priority" in tool.args
    assert "notes" in tool.args
    assert tool.args["profile"]["type"] == "object"
    assert tool.args["profile"]["properties"]["user_id"]["type"] == "integer"
    assert tool.args["profile"]["properties"]["name"]["type"] == "string"
    assert tool.args["profile"]["properties"]["age"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert tool.args["profile"]["properties"]["contact"]["type"] == "object"
    assert tool.args["profile"]["properties"]["contact"]["properties"]["email"]["type"] == "string"


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


def test_tool_with_default_args_without_type_hints():
    def foo(x=100):
        return x

    tool = Tool(foo)
    assert tool.args["x"]["default"] == 100
    assert not hasattr(tool.args["x"], "type")


def test_tool_call_parses_args():
    tool = Tool(dummy_with_pydantic)

    args = {
        "model": {
            "field1": "hello",
            "field2": 123,
        }
    }

    result = tool(**args)
    assert result == "hello 123"


def test_tool_call_parses_nested_list_of_pydantic_model():
    def dummy_function(x: list[list[DummyModel]]):
        return x

    tool = Tool(dummy_function)
    args = {
        "x": [
            [
                {
                    "field1": "hello",
                    "field2": 123,
                }
            ]
        ]
    }

    result = tool(**args)
    assert result == [[DummyModel(field1="hello", field2=123)]]


def test_tool_call_kwarg():
    def fn(x: int, **kwargs):
        return kwargs

    tool = Tool(fn)

    assert tool(x=1, y=2, z=3) == {"y": 2, "z": 3}


@pytest.mark.asyncio
async def test_async_tool_from_function():
    tool = Tool(async_dummy_function)

    assert tool.name == "async_dummy_function"
    assert "An async dummy function for testing" in tool.desc
    assert "x" in tool.args
    assert "y" in tool.args
    assert tool.args["x"]["type"] == "integer"
    assert tool.args["y"]["type"] == "string"
    assert tool.args["y"]["default"] == "hello"

    # Test async call
    result = await tool.acall(x=42, y="hello")
    assert result == "hello 42"


@pytest.mark.asyncio
async def test_async_tool_with_pydantic():
    tool = Tool(async_dummy_with_pydantic)

    assert tool.name == "async_dummy_with_pydantic"
    assert "model" in tool.args
    assert tool.args["model"]["type"] == "object"
    assert "field1" in tool.args["model"]["properties"]
    assert "field2" in tool.args["model"]["properties"]

    # Test async call with pydantic model
    model = DummyModel(field1="test", field2=123)
    result = await tool.acall(model=model)
    assert result == "test 123"

    # Test async call with dict
    result = await tool.acall(model={"field1": "test", "field2": 123})
    assert result == "test 123"


@pytest.mark.asyncio
async def test_async_tool_with_complex_pydantic():
    tool = Tool(async_complex_dummy_function)

    profile = UserProfile(
        user_id=1,
        name="Test User",
        contact=ContactInfo(
            email="test@example.com",
            addresses=[
                Address(street="123 Main St", city="Test City", zip_code="12345", is_primary=True),
                Address(street="456 Side St", city="Test City", zip_code="12345"),
            ],
        ),
    )

    result = await tool.acall(profile=profile, priority=1, notes="Test note")
    assert result["user_id"] == 1
    assert result["name"] == "Test User"
    assert result["priority"] == 1
    assert result["notes"] == "Test note"
    assert result["primary_address"]["street"] == "123 Main St"


@pytest.mark.asyncio
async def test_async_tool_invalid_call():
    tool = Tool(async_dummy_function)
    with pytest.raises(ValueError):
        await tool.acall(x="not an integer", y="hello")


@pytest.mark.asyncio
async def test_async_tool_with_kwargs():
    async def fn(x: int, **kwargs):
        return kwargs

    tool = Tool(fn)

    result = await tool.acall(x=1, y=2, z=3)
    assert result == {"y": 2, "z": 3}


@pytest.mark.asyncio
async def test_async_concurrent_calls():
    """Test that multiple async tools can run concurrently."""
    tool = Tool(async_dummy_function)

    # Create multiple concurrent calls
    tasks = [tool.acall(x=i, y=f"hello{i}") for i in range(5)]

    # Run them concurrently and measure time
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    # Verify results, `asyncio.gather` returns results in the order of the tasks
    assert results == [f"hello{i} {i}" for i in range(5)]

    # Check that it ran concurrently (should take ~0.1s, not ~0.5s)
    # We use 0.3s as threshold to account for some overhead
    assert end_time - start_time < 0.3
