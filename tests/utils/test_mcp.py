import asyncio
import importlib
import json
import sys
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from dspy.utils.mcp import _convert_mcp_tool_result, _get_field, convert_mcp_tool

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)

_UNSET = object()


def make_call_tool_result(field_style, texts=(), structured=_UNSET, is_error=False):
    from mcp.types import TextContent

    fields = {"content": [TextContent(type="text", text=text) for text in texts]}
    if field_style == "snake":
        fields["is_error"] = is_error
        if structured is not _UNSET:
            fields["structured_content"] = structured
    else:
        fields["isError"] = is_error
        if structured is not _UNSET:
            fields["structuredContent"] = structured
    return SimpleNamespace(**fields)


def expected_result(tool, structured, text):
    if _get_field(tool, "output_schema", "outputSchema"):
        return structured
    return text


@pytest.mark.extra
def test_convert_mcp_tool_result_supports_both_field_styles():
    assert _convert_mcp_tool_result(make_call_tool_result("camel", texts=["hi"])) == "hi"
    assert _convert_mcp_tool_result(make_call_tool_result("snake", texts=["a", "b"])) == ["a", "b"]


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
def test_structured_content_is_preferred_without_unwrapping(field_style):
    result = make_call_tool_result(field_style, texts=["3"], structured={"result": 3})
    assert _convert_mcp_tool_result(result) == {"result": 3}

    result = make_call_tool_result(field_style, structured={"result": []})
    assert _convert_mcp_tool_result(result) == {"result": []}

    result = make_call_tool_result(field_style, texts=["fallback"], structured=None)
    assert _convert_mcp_tool_result(result) is None


@pytest.mark.extra
def test_explicit_null_structured_content_is_preserved():
    from mcp.types import CallToolResult

    if "structured_content" in CallToolResult.model_fields:
        result = CallToolResult(content=[], structured_content=None)
    elif "structuredContent" in CallToolResult.model_fields:
        result = CallToolResult(content=[], structuredContent=None)
    else:
        pytest.skip("This MCP version predates structured content")

    assert _convert_mcp_tool_result(result) is None


@pytest.mark.extra
@pytest.mark.parametrize("field_style", ["camel", "snake"])
def test_error_result_raises(field_style):
    result = make_call_tool_result(field_style, texts=["boom"], is_error=True)
    with pytest.raises(RuntimeError, match="Failed to call a MCP tool: boom"):
        _convert_mcp_tool_result(result)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool_accepts_snake_case_tool_and_compatible_client():
    class Client:
        async def call_tool(self, name, arguments=None):
            assert name == "increment"
            return make_call_tool_result("snake", texts=["2"], structured={"result": 2})

    tool = SimpleNamespace(
        name="increment",
        title="Increment",
        description=None,
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )
    converted = convert_mcp_tool(Client(), tool)

    assert converted.desc == "Increment"
    assert converted.args == {"value": {"type": "integer"}}
    assert await converted.acall(value=1) == {"result": 2}


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
        increment_tool = convert_mcp_tool(client, response.tools[0])
        assert await increment_tool.acall(value=1) == {"result": 2}


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
            mcp_tools = {tool.name: tool for tool in response.tools}
            tools = {name: convert_mcp_tool(session, tool) for name, tool in mcp_tools.items()}

            add_tool = tools["add"]
            assert add_tool.name == "add"
            assert add_tool.desc == "Add two numbers"
            assert add_tool.arg_types == {"a": int, "b": int}
            assert await add_tool.acall(a=1, b=2) == expected_result(mcp_tools["add"], {"result": 3}, "3")

            hello_tool = tools["hello"]
            assert hello_tool.arg_types == {"names": list}
            greetings = ["Hello, Bob!", "Hello, Tom!"]
            assert await hello_tool.acall(names=["Bob", "Tom"]) == expected_result(
                mcp_tools["hello"], {"result": greetings}, greetings
            )

            with pytest.raises(RuntimeError, match="error!"):
                await tools["wrong_tool"].acall()

            nested_pydantic_tool = tools["get_account_name"]
            account_schema = nested_pydantic_tool.args["account"]
            assert set(account_schema["properties"]) == {"profile", "account_id"}
            account_in_json = {
                "profile": {"name": "Bob", "age": 20},
                "account_id": "123",
            }
            assert await nested_pydantic_tool.acall(account=account_in_json) == "Bob"

            current_datetime_tool = tools["current_datetime"]
            assert current_datetime_tool.args == {}
            datetime = "2025-07-23T09:10:10.0+00:00"
            assert await current_datetime_tool.acall() == expected_result(
                mcp_tools["current_datetime"], {"result": datetime}, datetime
            )

            profile_result = await tools["get_profile"].acall()
            if isinstance(profile_result, str):
                profile_result = json.loads(profile_result)
            assert profile_result == {"name": "Ann", "age": 30}

            single_field_result = await tools["genuine_single_field"].acall()
            if isinstance(single_field_result, str):
                single_field_result = json.loads(single_field_result)
            assert single_field_result == {"result": 42}
