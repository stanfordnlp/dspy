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
    assert tool.args == {
        "profile": {
            "title": "Profile",
            "type": "object",
            "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}},
            "required": ["name", "age"],
        }
    }
    assert tool.arg_types == {"profile": Profile}
    assert tool.arg_desc == {"profile": "No description provided. (Required)"}
    assert await tool.acall(profile=Profile(name="John", age=20)) == 20


@pytest.mark.asyncio
@pytest.mark.extra
async def test_from_langchain_hides_injected_tool_arg():
    from typing import Annotated

    from langchain_core.tools import InjectedToolArg, tool

    @tool
    def get_orders(status: str, user_id: Annotated[str, InjectedToolArg]) -> str:
        """List the orders of the signed-in user."""
        return f"orders of {user_id} with status {status}"

    converted = convert_langchain_tool(get_orders)
    assert list(converted.args) == ["status"]
    assert list(converted.arg_types) == ["status"]
    assert converted.as_function_tool().parameters["required"] == ["status"]
    # The wrapper accepts the extra key because it is defined as **kwargs, then
    # drops it. The forged user id must not reach the tool.
    with pytest.raises(RuntimeError, match="user_id") as raised:
        await converted.acall(status="open", user_id="u-2")
    assert "u-2" not in str(raised.value)
