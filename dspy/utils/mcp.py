from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import _list_all_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

if TYPE_CHECKING:
    import mcp

def _convert_mcp_tool_result(call_tool_result: "mcp.types.CallToolResult") -> str | list[Any]:
    from mcp.types import TextContent

    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    return tool_content or non_text_contents


def convert_mcp_tool(
        session: "mcp.client.session.ClientSession | None",
        tool: "mcp.types.Tool", 
        connection: "langchain_mcp_adapters.sessions.Connection | None" = None
    ) -> Tool:
    """Build a DSPy tool from an MCP tool.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.
        connection: The connection config to use for the MCP session.

    Returns:
        A dspy Tool object.
    """
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(tool.inputSchema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        if session is None:
            async with create_session(connection) as tool_session:
                await tool_session.initialize()
                call_tool_result = await tool_session.call_tool(tool.name, arguments=kwargs)
        else:
            call_tool_result = await session.call_tool(tool.name, arguments=kwargs)
            
        return _convert_mcp_tool_result(call_tool_result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)


async def load_mcp_tools(
    session: "mcp.client.session.ClientSession | None",
    *,
    connection: "langchain_mcp_adapters.sessions.Connection | None" = None
) -> list[Tool]:
    """Load all available MCP tools and convert them to DSPy tools.

    Args:
        session: The MCP client session. If None, connection must be provided.
        connection: Connection config to create a new session if session is None.

    Returns:
        List of DSPy tools. Tool annotations are returned as part
        of the tool metadata object.

    Raises:
        ValueError: If neither session nor connection is provided.
    """
    
    if session is None and connection is None:
        msg = "Either a session or a connection config must be provided"
        raise ValueError(msg)

    if session is None:
        # If a session is not provided, we will create one on the fly
        async with create_session(connection) as tool_session:
            await tool_session.initialize()
            tools = await _list_all_tools(tool_session)
    else:
        tools = await _list_all_tools(session)

    return [
        convert_mcp_tool(session, tool, connection=connection)
        for tool in tools
    ]


# thanks langchain
class DspyMultiServerMCPClient(MultiServerMCPClient):
    """Client for connecting to multiple MCP servers.

    Loads DSPy-compatible tools, prompts and resources from MCP servers.
    """

    async def get_tools(self, *, server_name: str | None = None) -> list[Tool]:
        """Get a list of all tools from all connected servers.

        Args:
            server_name: Optional name of the server to get tools from.
                If None, all tools from all servers will be returned (default).

        NOTE: a new session will be created for each tool call

        Returns:
            A list of DSPy tools

        """
        if server_name is not None:
            if server_name not in self.connections:
                msg = (
                    f"Couldn't find a server with name '{server_name}', "
                    f"expected one of '{list(self.connections.keys())}'"
                )
                raise ValueError(msg)
            return await load_mcp_tools(None, connection=self.connections[server_name])

        all_tools: list[Tool] = []
        load_mcp_tool_tasks = []
        for connection in self.connections.values():
            load_mcp_tool_task = asyncio.create_task(
                load_mcp_tools(None, connection=connection)
            )
            load_mcp_tool_tasks.append(load_mcp_tool_task)
        tools_list = await asyncio.gather(*load_mcp_tool_tasks)
        for tools in tools_list:
            all_tools.extend(tools)
        return all_tools