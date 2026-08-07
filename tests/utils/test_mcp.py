import asyncio
import importlib
import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from dspy import Tool
from dspy.utils.mcp import _convert_mcp_tool_result, convert_mcp_tool

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)


def make_call_tool_result(field_style, texts=(), is_error=False):
    from mcp.types import TextContent

    fields = {
        "content": [TextContent(type="text", text=text) for text in texts],
        "structured_content" if field_style == "snake" else "structuredContent": {"result": "ignored"},
        "is_error" if field_style == "snake" else "isError": is_error,
    }
    return SimpleNamespace(**fields)


@pytest.mark.extra
def test_convert_mcp_tool_result_supports_both_field_styles_without_changing_results():
    assert _convert_mcp_tool_result(make_call_tool_result("camel", texts=["hi"])) == "hi"
    assert _convert_mcp_tool_result(make_call_tool_result("snake", texts=["a", "b"])) == ["a", "b"]


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
def test_error_result_raises(field_style):
    result = make_call_tool_result(field_style, texts=["boom"], is_error=True)
    with pytest.raises(RuntimeError, match="Failed to call a MCP tool: boom"):
        _convert_mcp_tool_result(result)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool_with_v2_client():
    if int(version("mcp").split(".")[0]) < 2:
        pytest.skip("The high-level MCP client is available in SDK v2")

    from mcp.client import Client
    from mcp.server import MCPServer

    server = MCPServer("test")

    @server.tool()
    def increment(value: int) -> int:
        return value + 1

    async with Client(server) as client:
        response = await client.list_tools()
        increment_tool = Tool.from_mcp_tool(client, response.tools[0])
        assert await increment_tool.acall(value=1) == "2"


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["tests/utils/resources/mcp_server.py"],
        env=None,
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=5)
            response = await session.list_tools()

            # Check add
            add_tool = convert_mcp_tool(session, response.tools[0])
            assert add_tool.name == "add"
            assert add_tool.desc == "Add two numbers"
            assert add_tool.args == {
                "a": {"title": "A", "type": "integer"},
                "b": {"title": "B", "type": "integer"},
            }
            assert add_tool.arg_types == {"a": int, "b": int}
            assert add_tool.arg_desc == {
                "a": "No description provided. (Required)",
                "b": "No description provided. (Required)",
            }
            assert await add_tool.acall(a=1, b=2) == "3"

            # Check hello
            hello_tool = convert_mcp_tool(session, response.tools[1])
            assert hello_tool.name == "hello"
            assert hello_tool.desc == "Greet people"
            assert hello_tool.args == {"names": {"title": "Names", "type": "array", "items": {"type": "string"}}}
            assert hello_tool.arg_types == {"names": list}
            assert hello_tool.arg_desc == {"names": "No description provided. (Required)"}
            assert await hello_tool.acall(names=["Bob", "Tom"]) == ["Hello, Bob!", "Hello, Tom!"]

            # Check error handling
            error_tool = convert_mcp_tool(session, response.tools[2])
            assert error_tool.name == "wrong_tool"
            assert error_tool.desc == "This tool raises an error"
            with pytest.raises(RuntimeError, match="error!"):
                await error_tool.acall()

            # Check nested Pydantic arg
            nested_pydantic_tool = convert_mcp_tool(session, response.tools[3])

            assert nested_pydantic_tool.name == "get_account_name"
            assert nested_pydantic_tool.desc == "This extracts the name from account"
            assert nested_pydantic_tool.args == {
                "account": {
                    "title": "Account",
                    "type": "object",
                    "required": ["profile", "account_id"],
                    "properties": {
                        "profile": {
                            "title": "Profile",
                            "type": "object",
                            "properties": {
                                "name": {"title": "Name", "type": "string"},
                                "age": {"title": "Age", "type": "integer"},
                            },
                            "required": ["name", "age"],
                        },
                        "account_id": {"title": "Account Id", "type": "string"},
                    },
                }
            }
            account_in_json = {
                "profile": {
                    "name": "Bob",
                    "age": 20,
                },
                "account_id": "123",
            }
            result = await nested_pydantic_tool.acall(account=account_in_json)
            assert result == "Bob"

            # Check no input parameter current_datetime tool
            current_datetime_tool = convert_mcp_tool(session, response.tools[4])
            assert current_datetime_tool.name == "current_datetime"
            assert current_datetime_tool.desc == "Get the current datetime"
            assert current_datetime_tool.args == {}
            assert current_datetime_tool.arg_types == {}
            assert current_datetime_tool.arg_desc == {}
            assert await current_datetime_tool.acall() == "2025-07-23T09:10:10.0+00:00"
