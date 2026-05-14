import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

import dspy
from dspy.adapters.types.tool import Tool, ToolCalls, convert_input_schema_to_tool_args, from_lm_tool_call


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
    phone: str | None = None
    addresses: list[Address]


class UserProfile(BaseModel):
    user_id: int
    name: str
    age: int | None = None
    contact: ContactInfo
    tags: list[str] = []


class Note(BaseModel):
    content: str
    author: str


def complex_dummy_function(profile: UserProfile, priority: int, notes: list[Note] | None = None) -> dict[str, Any]:
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
    profile: UserProfile,
    priority: int,
    notes: list[Note] | None = None,
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

    # Reference should be resolved for nested pydantic models
    assert "$defs" not in str(tool.args["notes"])
    assert tool.args["notes"]["anyOf"][0]["type"] == "array"
    assert tool.args["notes"]["anyOf"][0]["items"]["type"] == "object"
    assert tool.args["notes"]["anyOf"][0]["items"]["properties"]["content"]["type"] == "string"
    assert tool.args["notes"]["anyOf"][0]["items"]["properties"]["author"]["type"] == "string"


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


def test_tool_str():
    def add(x: int, y: int = 0) -> int:
        """Add two integers."""
        return x + y

    tool = Tool(add)
    assert (
        str(tool)
        == "add, whose description is <desc>Add two integers.</desc>. It takes arguments {'x': {'type': 'integer'}, 'y': {'type': 'integer', 'default': 0}}."
    )


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

    result = await tool.acall(profile=profile, priority=1, notes=[Note(content="Test note", author="Test author")])
    assert result["user_id"] == 1
    assert result["name"] == "Test User"
    assert result["priority"] == 1
    assert result["notes"] == [Note(content="Test note", author="Test author")]
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_async_tool_call_in_sync_mode():
    tool = Tool(async_dummy_function)
    with dspy.context(allow_tool_async_sync_conversion=False):
        with pytest.raises(ValueError, match=r".*acall.*allow_tool_async_sync_conversion.*"):
            result = tool(x=1, y="hello")

    with dspy.context(allow_tool_async_sync_conversion=True):
        result = tool(x=1, y="hello")
        assert result == "hello 1"


TOOL_CALL_TEST_CASES = [
    ([], {"tool_calls": []}),
    (
        [{"name": "search", "args": {"query": "hello"}}],
        {
            "tool_calls": [{"name": "search", "args": {"query": "hello"}}],
        },
    ),
    (
        [
            {"name": "search", "args": {"query": "hello"}},
            {"name": "translate", "args": {"text": "world", "lang": "fr"}},
        ],
        {
            "tool_calls": [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world", "lang": "fr"}},
            ],
        },
    ),
    (
        [{"name": "get_time", "args": {}}],
        {
            "tool_calls": [{"name": "get_time", "args": {}}],
        },
    ),
]


@pytest.mark.parametrize("tool_calls_data,expected", TOOL_CALL_TEST_CASES)
def test_tool_calls_format_basic(tool_calls_data, expected):
    """Test ToolCalls.format with various basic scenarios."""
    tool_calls_list = [ToolCalls.ToolCall(**data) for data in tool_calls_data]
    tool_calls = ToolCalls(tool_calls=tool_calls_list)
    result = tool_calls.format()

    assert result == expected


def test_toolcalls_vague_match():
    """
    Test that ToolCalls can parse the data with slightly off format:

    - a single dict with "name" and "args"
    - a list of dicts with "name" and "args"
    - invalid input (should raise ValueError)
    """
    # Single dict with "name" and "args" should parse as one ToolCall
    data_single = {"name": "search", "args": {"query": "hello"}}
    tc = ToolCalls.model_validate(data_single)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[0].args == {"query": "hello"}

    # List of dicts with "name" and "args" should parse as multiple ToolCalls
    data_list = [
        {"name": "search", "args": {"query": "hello"}},
        {"name": "translate", "args": {"text": "world", "lang": "fr"}},
    ]
    tc = ToolCalls.model_validate(data_list)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[1].name == "translate"

    # Dict with "tool_calls" key containing a list of dicts
    data_tool_calls = {
        "tool_calls": [
            {"name": "search", "args": {"query": "hello"}},
            {"name": "get_time", "args": {}},
        ]
    }
    tc = ToolCalls.model_validate(data_tool_calls)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[1].name == "get_time"

    # Invalid input should raise ValueError
    with pytest.raises(ValueError):
        ToolCalls.model_validate({"foo": "bar"})
    with pytest.raises(ValueError):
        ToolCalls.model_validate([{"foo": "bar"}])

    # Raw wire-format dicts should raise a clear ValueError, not a deep pydantic error
    with pytest.raises(ValueError, match="each item must be a ToolCall"):
        ToolCalls.model_validate([{"type": "function", "function": {"name": "search", "arguments": '{"q": "hi"}'}}])


def test_tool_convert_input_schema_to_tool_args_no_input_params():
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(schema={"properties": {}})
    assert args == {}
    assert arg_types == {}
    assert arg_desc == {}


def test_tool_convert_input_schema_to_tool_args_lang_chain():
    # Example from langchain docs:
    # https://web.archive.org/web/20250723101359/https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(
        schema={
            "title": "fooSchema",
            "description": "The foo.",
            "type": "object",
            "properties": {
                "bar": {
                    "title": "Bar",
                    "description": "The bar.",
                    "type": "string",
                },
                "baz": {
                    "title": "Baz",
                    "type": "integer",
                },
            },
            "required": [
                "baz",
            ],
        }
    )
    assert args == {
        "bar": {"title": "Bar", "description": "The bar.", "type": "string"},
        "baz": {"title": "Baz", "type": "integer"},
    }
    assert arg_types == {
        "bar": str,
        "baz": int,
    }
    assert arg_desc == {
        "bar": "The bar.",
        "baz": "No description provided. (Required)",
    }




def test_tool_call_execute():
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    def add_numbers(a: int, b: int) -> int:
        return a + b

    tools = [
        dspy.Tool(get_weather),
        dspy.Tool(add_numbers)
    ]

    tool_call = dspy.ToolCalls.ToolCall(name="get_weather", args={"city": "Berlin"})
    result = tool_call.execute(functions=tools)
    assert result == "The weather in Berlin is sunny"

    # Test individual tool call with function dict
    tool_call2 = dspy.ToolCalls.ToolCall(name="add_numbers", args={"a": 7, "b": 13})
    result2 = tool_call2.execute(functions={"add_numbers": add_numbers})
    assert result2 == 20

    # Test individual tool call with no arguments
    def get_pi():
        return 3.14159

    tool_call3 = dspy.ToolCalls.ToolCall(name="get_pi", args={})
    result3 = tool_call3.execute(functions={"get_pi": get_pi})
    assert result3 == 3.14159

    # Test error case
    tool_call4 = dspy.ToolCalls.ToolCall(name="nonexistent", args={})
    try:
        tool_call4.execute(functions=tools)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)


def test_tool_call_execute_with_local_functions():
    def main():
        def local_add(a: int, b: int) -> int:
            return a + b

        def local_multiply(x: int, y: int) -> int:
            return x * y

        # Test individual execution with local function
        tool_call1 = dspy.ToolCalls.ToolCall(name="local_add", args={"a": 10, "b": 15})
        result1 = tool_call1.execute()  # Should find local function automatically
        assert result1 == 25

        tool_call2 = dspy.ToolCalls.ToolCall(name="local_multiply", args={"x": 4, "y": 7})
        result2 = tool_call2.execute()  # Should find local function automatically
        assert result2 == 28

        # Test locals take precedence over globals
        try:
            globals()["local_add"] = lambda a, b: a + b + 1000
            precedence_call = dspy.ToolCalls.ToolCall(name="local_add", args={"a": 1, "b": 2})
            result = precedence_call.execute()
            assert result == 3  # Should use local function (1+2=3), not global (1+2+1000=1003)
        finally:
            globals().pop("local_add", None)

    main()


# =============================================================================
# `from_lm_tool_call`: the single inbound boundary that normalizes every LiteLLM
# tool-call wire shape into a canonical `ToolCalls.ToolCall`. After this
# normalization, no downstream code should need to handle multiple wire shapes.
# =============================================================================


def test_from_lm_tool_call_chat_completions_dict_shape():
    item = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q":"hello"}'},
    }
    tc = from_lm_tool_call(item)
    assert isinstance(tc, ToolCalls.ToolCall)
    assert tc.name == "search"
    assert tc.args == {"q": "hello"}
    assert tc.id == "call_abc"


def test_from_lm_tool_call_chat_completions_pydantic_shape():
    """Real LiteLLM `ChatCompletionMessageToolCall` is a pydantic object;
    we go through model_dump first."""
    class Fn:
        name = "search"
        arguments = '{"q":"x"}'

    class CCMToolCall:
        id = "call_123"
        type = "function"
        function = Fn()

        def model_dump(self):
            return {
                "id": self.id,
                "type": self.type,
                "function": {"name": self.function.name, "arguments": self.function.arguments},
            }

    tc = from_lm_tool_call(CCMToolCall())
    assert tc.name == "search"
    assert tc.args == {"q": "x"}
    assert tc.id == "call_123"


def test_from_lm_tool_call_chat_completions_arguments_as_dict():
    """Some providers (and our own round-trips) put `arguments` as a dict."""
    item = {"type": "function", "function": {"name": "lookup", "arguments": {"k": "v"}}}
    tc = from_lm_tool_call(item)
    assert tc.args == {"k": "v"}


def test_from_lm_tool_call_chat_completions_empty_arguments_string():
    """`arguments=""` should normalize to `{}`, not crash."""
    item = {"type": "function", "function": {"name": "ping", "arguments": ""}}
    assert from_lm_tool_call(item).args == {}


def test_from_lm_tool_call_responses_api_dict_shape():
    item = {
        "type": "function_call",
        "name": "search",
        "arguments": '{"q":"y"}',
        "call_id": "call_xyz",
    }
    tc = from_lm_tool_call(item)
    assert tc.name == "search"
    assert tc.args == {"q": "y"}
    assert tc.id == "call_xyz"


def test_from_lm_tool_call_responses_api_pydantic_shape():
    class FunctionCallItem:
        type = "function_call"
        name = "search"
        arguments = '{"q":"z"}'
        call_id = "call_99"

        def model_dump(self):
            return {
                "type": self.type,
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

    tc = from_lm_tool_call(FunctionCallItem())
    assert tc.name == "search"
    assert tc.args == {"q": "z"}
    assert tc.id == "call_99"


def test_from_lm_tool_call_mockvalser_fallback_chat_completions_shape():
    """Cached LiteLLM pydantic whose model_dump raises TypeError — must fall
    back to attribute access via the `function` attribute."""
    class Fn:
        name = "search"
        arguments = '{"q":"cached"}'

    class CachedToolCall:
        id = "call_cached"
        type = "function"
        function = Fn()

        def model_dump(self):
            raise TypeError("'MockValSer' object cannot be converted to 'SchemaSerializer'")

    tc = from_lm_tool_call(CachedToolCall())
    assert tc.name == "search"
    assert tc.args == {"q": "cached"}
    assert tc.id == "call_cached"


def test_from_lm_tool_call_mockvalser_fallback_responses_shape():
    """Cached Responses API pydantic object — falls back to attribute access
    via the `name` attribute."""
    class CachedFunctionCall:
        name = "search"
        arguments = '{"q":"cached"}'
        call_id = "call_resp"

        def model_dump(self):
            raise TypeError("'MockValSer' object cannot be converted to 'SchemaSerializer'")

    tc = from_lm_tool_call(CachedFunctionCall())
    assert tc.name == "search"
    assert tc.args == {"q": "cached"}
    assert tc.id == "call_resp"


def test_from_lm_tool_call_mockvalser_fallback_no_recoverable_attrs_raises():
    """If model_dump fails AND there's no `function` or `name` attribute,
    raise TypeError."""
    class Unsalvageable:
        def model_dump(self):
            raise TypeError("'MockValSer' object cannot be converted to 'SchemaSerializer'")

    with pytest.raises(TypeError):
        from_lm_tool_call(Unsalvageable())


def test_from_lm_tool_call_unrecognized_shape_raises():
    with pytest.raises(ValueError, match="Unrecognized tool-call shape"):
        from_lm_tool_call({"type": "unknown", "name": "x"})


def test_from_lm_tool_call_non_dict_non_pydantic_raises():
    with pytest.raises(TypeError, match="Cannot normalize tool call from int"):
        from_lm_tool_call(42)


def test_toolcall_id_field_optional():
    """ToolCall.id should default to None when not provided."""
    tc = ToolCalls.ToolCall(name="f", args={})
    assert tc.id is None


def test_toolcall_id_round_trips():
    tc = ToolCalls.ToolCall(name="f", args={"a": 1}, id="call_zzz")
    assert tc.id == "call_zzz"


def test_toolcall_format_preserves_id_chat_round_trip():
    original = ToolCalls.ToolCall(name="search", args={"q": "hello"}, id="call_xyz")
    restored = from_lm_tool_call(original.to_lm_tool_call("chat"))
    assert restored == original


def test_toolcall_format_preserves_id_responses_round_trip():
    """Outbound symmetry: a tool call serialized for the Responses API must
    round-trip back through `from_lm_tool_call`."""
    original = ToolCalls.ToolCall(name="search", args={"q": "hello"}, id="call_xyz")
    restored = from_lm_tool_call(original.to_lm_tool_call("responses"))
    assert restored == original


def test_toolcall_format_returns_canonical_shape():
    """`format()` returns the model-agnostic canonical shape, not a wire dialect."""
    tc = ToolCalls.ToolCall(name="search", args={"q": "x"}, id="c1")
    assert tc.format() == {"name": "search", "args": {"q": "x"}}


def test_toolcall_format_omits_id_when_absent():
    """A ToolCall constructed without an id should serialize without an `id`
    (chat) or `call_id` (responses) key so we don't fabricate one on the wire."""
    chat = ToolCalls.ToolCall(name="search", args={"q": "x"}).to_lm_tool_call("chat")
    assert "id" not in chat
    responses = ToolCalls.ToolCall(name="search", args={"q": "x"}).to_lm_tool_call("responses")
    assert "call_id" not in responses


def test_toolcall_format_arguments_is_json_string_for_openai_assistant_message():
    """OpenAI Chat Completions requires `function.arguments` to be a
    JSON-encoded **string** when this payload is replayed as an assistant
    tool-call message. Same for the Responses API top-level `arguments`."""
    import json as _json

    chat = ToolCalls.ToolCall(name="search", args={"q": "hello", "n": 3}, id="c1").to_lm_tool_call("chat")
    assert isinstance(chat["function"]["arguments"], str)
    assert _json.loads(chat["function"]["arguments"]) == {"q": "hello", "n": 3}

    responses = ToolCalls.ToolCall(name="search", args={"q": "hello", "n": 3}, id="c1").to_lm_tool_call(
        "responses"
    )
    assert isinstance(responses["arguments"], str)
    assert _json.loads(responses["arguments"]) == {"q": "hello", "n": 3}


def test_toolcall_format_responses_uses_call_id_not_id():
    """The Responses API field is `call_id`, not `id`."""
    payload = ToolCalls.ToolCall(name="x", args={}, id="call_1").to_lm_tool_call("responses")
    assert payload["call_id"] == "call_1"
    assert "id" not in payload
    assert payload["type"] == "function_call"


def test_toolcall_format_empty_args_is_json_object_string():
    assert (
        ToolCalls.ToolCall(name="ping", args={}).to_lm_tool_call("chat")["function"]["arguments"]
        == "{}"
    )
    assert (
        ToolCalls.ToolCall(name="ping", args={}).to_lm_tool_call("responses")["arguments"]
        == "{}"
    )


def test_toolcall_format_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        ToolCalls.ToolCall(name="x", args={}).to_lm_tool_call("gemini")


def test_tool_definition_chat_completions_shape():
    tool = dspy.Tool(lambda city: city, name="get_weather", desc="weather")
    payload = tool.to_lm_tool_definition(model_type="chat")
    assert payload["type"] == "function"
    assert "function" in payload
    assert payload["function"]["name"] == "get_weather"
    assert payload["function"]["description"] == "weather"
    assert "parameters" in payload["function"]


def test_tool_definition_responses_api_shape():
    tool = dspy.Tool(lambda city: city, name="get_weather", desc="weather")
    payload = tool.to_lm_tool_definition(model_type="responses")
    assert payload["type"] == "function"
    # Responses API flattens: name/description/parameters at top level, no `function` wrapper.
    assert "function" not in payload
    assert payload["name"] == "get_weather"
    assert payload["description"] == "weather"
    assert "parameters" in payload


def test_tool_definition_rejects_unknown_model_type():
    tool = dspy.Tool(lambda city: city, name="x", desc="d")
    with pytest.raises(ValueError, match="Unknown model_type"):
        tool.to_lm_tool_definition(model_type="gemini")
