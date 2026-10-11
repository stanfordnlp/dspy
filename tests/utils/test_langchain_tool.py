import importlib
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("langchain_core") is None:
    pytest.skip(reason="langchain_core is not installed", allow_module_level=True)

from pydantic import BaseModel

from dspy.utils.langchain_tool import convert_langchain_tool


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_custom_simple_tool():
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = convert_langchain_tool(add)
    assert tool.name == "add"
    assert tool.desc == "Add two numbers."
    assert tool.args == {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}
    assert tool.arg_types == {"a": int, "b": int}
    assert tool.arg_desc == {"a": "No description provided. (Required)", "b": "No description provided. (Required)"}
    assert await tool.acall(a=1, b=2) == 3


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_custom_tool_with_custom_class():
    from langchain_core.tools import tool

    class Profile(BaseModel):
        name: str
        age: int

    @tool
    def get_age(profile: Profile) -> int:
        """Get the age of the profile."""
        return profile.age

    tool = convert_langchain_tool(get_age)
    assert tool.name == "get_age"
    assert tool.desc == "Get the age of the profile."
    assert tool.args == {"profile": {"title": "Profile", "type": "object", "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}}, "required": ["name", "age"]}}
    assert tool.arg_types == {"profile": Profile}
    assert tool.arg_desc == {"profile": "No description provided. (Required)"}
    assert await tool.acall(profile=Profile(name="John", age=20)) == 20


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_tool_with_json_schema_dict():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    calls = []

    async def search(query: str, limit: int = 5) -> str:
        calls.append((query, limit))
        return f"{query}:{limit}"

    async def ainvoke(kwargs):
        return await search(**kwargs)

    langchain_tool = SimpleNamespace(
        args_schema=schema,
        name="search",
        description="Search records.",
        ainvoke=ainvoke,
    )
    dspy_tool = convert_langchain_tool(langchain_tool)

    assert await dspy_tool.acall(query="cats") == "cats:5"
    assert dspy_tool.args == schema["properties"]
    assert dspy_tool.format_as_litellm_function_call()["function"]["parameters"]["required"] == ["query"]
    with pytest.raises(ValueError, match="arguments are invalid"):
        await dspy_tool.acall(query=3)
    assert calls == [("cats", 5)]


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_tool_preserves_nullable_types_and_required_schema():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": ["string", "null"]},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }

    async def search(query: str | None, limit: int = 5) -> str:
        return f"{query}:{limit}"

    source = SimpleNamespace(
        args_schema=schema,
        name="search",
        description="Search records.",
        ainvoke=lambda kwargs: search(**kwargs),
    )
    dspy_tool = convert_langchain_tool(source)

    assert dspy_tool.args == schema["properties"]
    assert dspy_tool.format_as_litellm_function_call()["function"]["parameters"]["required"] == ["query"]
    assert await dspy_tool.acall(query=None) == "None:5"
    assert await dspy_tool.acall(query="cats") == "cats:5"
    with pytest.raises(ValueError):
        await dspy_tool.acall(query=3)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_dict_schema_union_preserves_values_and_root_constraints():
    schema = {
        "type": "object",
        "properties": {"query": {"type": ["string", "integer", "null"]}},
        "required": ["query"],
        "anyOf": [{"properties": {"query": {"const": 3}}}, {"properties": {"query": {"type": "null"}}}],
    }

    async def ainvoke(kwargs):
        return kwargs["query"]

    converted = convert_langchain_tool(
        SimpleNamespace(args_schema=schema, name="search", description="Search.", ainvoke=ainvoke)
    )
    result = await converted.acall(query=3)
    assert result == 3 and type(result) is int
    assert await converted.acall(query=None) is None
    with pytest.raises(ValueError):
        await converted.acall(query="cats")


@pytest.mark.asyncio
@pytest.mark.extra
@pytest.mark.parametrize("required_default", [False, True])
async def test_dict_schema_preserves_required_and_rejects_invalid_calls(required_default):
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    if required_default:
        schema["properties"]["query"]["default"] = "cats"
    calls = []

    async def ainvoke(kwargs):
        calls.append(kwargs)
        return kwargs

    converted = convert_langchain_tool(
        SimpleNamespace(args_schema=schema, name="search", description="Search.", ainvoke=ainvoke)
    )
    assert converted.format_as_litellm_function_call()["function"]["parameters"] == schema
    assert await converted.acall(query="cats") == {"query": "cats"}
    for arguments in ({}, {"query": "cats", "unknown": 1}, {"query": 1}):
        with pytest.raises(ValueError):
            await converted.acall(**arguments)
    assert calls == [{"query": "cats"}]


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_real_langchain_tool_with_json_schema_dict():
    from langchain_core.tools import StructuredTool

    schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    async def search(query: str) -> str:
        return f"found:{query}"

    try:
        source = StructuredTool.from_function(
            coroutine=search, name="search", description="Search records.", args_schema=schema
        )
    except TypeError as error:
        if "args_schema must be a subclass of pydantic BaseModel" not in str(error):
            raise
        pytest.skip("This LangChain version does not support JSON-dict args_schema")

    assert await source.ainvoke({"query": "cats"}) == "found:cats"
    assert await convert_langchain_tool(source).acall(query="cats") == "found:cats"
