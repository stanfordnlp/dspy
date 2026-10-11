import importlib

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
    from langchain_core.tools import StructuredTool

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

    langchain_tool = StructuredTool.from_function(
        coroutine=search,
        name="search",
        description="Search records.",
        args_schema=schema,
    )
    dspy_tool = convert_langchain_tool(langchain_tool)

    assert await dspy_tool.acall(query="cats") == "cats:5"
    assert dspy_tool.args == schema["properties"]
    assert dspy_tool.format_as_litellm_function_call()["function"]["parameters"]["required"] == ["query"]
    with pytest.raises(ValueError, match="Arg query"):
        await dspy_tool.acall(query=3)
    assert calls == [("cats", 5)]
