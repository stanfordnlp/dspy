from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, _MCPToolClient, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _get_field(obj: Any, snake_name: str, camel_name: str, default: Any = None) -> Any:
    """Read a field whose Python name changed between mcp SDK v1 and v2."""
    for name in (snake_name, camel_name):
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


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

    if _get_field(call_tool_result, "is_error", "isError", default=False):
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    return tool_content or non_text_contents


def convert_mcp_tool(session: _MCPToolClient, tool: "mcp.types.Tool") -> Tool:
    """Build a DSPy tool from an MCP tool.

    Both mcp SDK v1's ``ClientSession`` and v2's high-level ``Client`` satisfy
    the client interface used by this bridge.

    Args:
        session: An MCP client or session with an async ``call_tool`` method.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    input_schema = _get_field(tool, "input_schema", "inputSchema", default={})
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(input_schema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)
