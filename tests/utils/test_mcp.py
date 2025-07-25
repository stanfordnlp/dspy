import asyncio
import importlib

import pytest

from dspy.utils.mcp import convert_mcp_tool
from dspy.utils.mcp import DspyMultiServerMCPClient

if importlib.util.find_spec("mcp") is None:
    pytest.skip(reason="mcp is not installed", allow_module_level=True)


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_mcp_tool():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    server_params = StdioServerParameters(
        command="python",
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
            assert add_tool.args == {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}
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
            with pytest.raises(
                RuntimeError, match="Failed to call a MCP tool: Error executing tool wrong_tool: error!"
            ):
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


@pytest.mark.asyncio
@pytest.mark.extra
async def test_mcp_tool_multiserver_single():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    client = DspyMultiServerMCPClient({
        'math': {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["tests/utils/resources/mcp_server.py"],
            "transport": "stdio"
        }
    })

    response = await client.get_tools()

    # Check add
    add_tool = response[0]
    assert add_tool.name == "add"
    assert add_tool.desc == "Add two numbers"
    assert add_tool.args == {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}
    assert add_tool.arg_types == {"a": int, "b": int}
    assert add_tool.arg_desc == {
        "a": "No description provided. (Required)",
        "b": "No description provided. (Required)",
    }
    assert await add_tool.acall(a=1, b=2) == "3"

    # Check hello
    hello_tool = response[1]
    assert hello_tool.name == "hello"
    assert hello_tool.desc == "Greet people"
    assert hello_tool.args == {"names": {"title": "Names", "type": "array", "items": {"type": "string"}}}
    assert hello_tool.arg_types == {"names": list}
    assert hello_tool.arg_desc == {"names": "No description provided. (Required)"}
    assert await hello_tool.acall(names=["Bob", "Tom"]) == ["Hello, Bob!", "Hello, Tom!"]

    # # Check error handling
    error_tool = response[2]
    assert error_tool.name == "wrong_tool"
    assert error_tool.desc == "This tool raises an error"
    with pytest.raises(
        RuntimeError, match="Failed to call a MCP tool: Error executing tool wrong_tool: error!"
    ):
        await error_tool.acall()

    # # Check nested Pydantic arg
    nested_pydantic_tool = response[3]

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

@pytest.mark.asyncio
@pytest.mark.extra
async def test_mcp_tool_multiserver_multiple():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from dspy.adapters.types.tool import Tool

    client = DspyMultiServerMCPClient({
        'math': {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["tests/utils/resources/mcp_server.py"],
            "transport": "stdio"
        },
        'weather': {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["tests/utils/resources/mcp_weather_server.py"],
            "transport": "stdio"
        },
        'time': {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["tests/utils/resources/mcp_time_server.py"],
            "transport": "stdio"
        }
    })


    all_tools = await client.get_tools()

    assert set(map(lambda t: t.name, all_tools)) == set(["add", "hello", "wrong_tool", "get_account_name", "get_weather", "get_time"])

    # Check that tools are Tool instances
    for tool in all_tools:
        assert isinstance(tool, Tool)

    # Check math server tools
    math_tools = await client.get_tools(server_name="math")
    assert len(math_tools) == 4
    math_tool_names = {tool.name for tool in math_tools}
    assert math_tool_names == {"add", "hello", "wrong_tool", "get_account_name"}

    # Check weather server tools
    weather_tools = await client.get_tools(server_name="weather")
    assert len(weather_tools) == 1
    assert weather_tools[0].name == "get_weather"

    # Check time server tools
    time_tools = await client.get_tools(server_name="time")
    assert len(time_tools) == 1
    assert time_tools[0].name == "get_time"

    # Test that we can call a math tool
    add_tool = next(tool for tool in all_tools if tool.name == "add")
    result = await add_tool.acall(a=2, b=3)
    assert result == "5"

    # Test that we can call a weather tool
    weather_tool = next(tool for tool in all_tools if tool.name == "get_weather")
    result = await weather_tool.acall(location="London")
    assert result == "It's always sunny in London"

    # Test that we can call a time tool
    time_tool = next(tool for tool in all_tools if tool.name == "get_time")
    result = await time_tool.acall(args="")
    assert result == "5:20:00 PM EST"
