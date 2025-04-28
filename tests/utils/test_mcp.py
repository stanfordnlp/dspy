import pytest
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dspy.utils import build_mcp_tool


@pytest.mark.asyncio
async def test_build_mcp_tool():
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
            add_tool = build_mcp_tool(session, response.tools[0])
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
            hello_tool = build_mcp_tool(session, response.tools[1])
            assert hello_tool.name == "hello"
            assert hello_tool.desc == "Greet people"
            assert hello_tool.args == {"names": {"title": "Names", "type": "array", "items": {"type": "string"}}}
            assert hello_tool.arg_types == {"names": list}
            assert hello_tool.arg_desc == {"names": "No description provided. (Required)"}
            assert await hello_tool.acall(names=["Bob", "Tom"]) == ["Hello, Bob!", "Hello, Tom!"]

            # Check error handling
            error_tool = build_mcp_tool(session, response.tools[2])
            assert error_tool.name == "wrong_tool"
            assert error_tool.desc == "This tool raises an error"
            with pytest.raises(
                RuntimeError, match="Failed to call a MCP tool: Error executing tool wrong_tool: error!"
            ):
                await error_tool.acall()
