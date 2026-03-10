from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp

__all__ = ["stdio_mcp_tools", "http_mcp_tools", "convert_mcp_tool"]


def _convert_mcp_tool_result(call_tool_result: mcp.types.CallToolResult) -> str | list[Any]:
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


def convert_mcp_tool(session: mcp.ClientSession, tool: mcp.types.Tool) -> Tool:
    """Build a DSPy tool from an MCP tool.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(tool.inputSchema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)


def _validate_tool_filters(include_tools: list[str] | None, exclude_tools: list[str] | None) -> None:
    if include_tools and exclude_tools:
        raise ValueError("Cannot specify both include_tools and exclude_tools.")


def _validate_tool_names(requested: list[str] | None, available: list[str]) -> None:
    if not requested:
        return
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(f"Tool names not found on MCP server: {unknown}. Available tools: {set(available)}")


def _filter_mcp_tools(
    tools: list[mcp.types.Tool],
    include_tools: list[str] | None,
    exclude_tools: list[str] | None,
) -> list[mcp.types.Tool]:
    if include_tools is not None:
        tools = [t for t in tools if t.name in set(include_tools)]
    elif exclude_tools is not None:
        excluded = set(exclude_tools)
        tools = [t for t in tools if t.name not in excluded]
    return tools


async def _build_tools_from_session(
    session: mcp.ClientSession,
    include_tools: list[str] | None,
    exclude_tools: list[str] | None,
) -> list[Tool]:
    """Initialize a session, list tools, validate names, filter, and convert.

    Shared helper used by both stdio_mcp_tools and http_mcp_tools to avoid
    duplicating the session setup logic.
    """
    await session.initialize()
    listed = await session.list_tools()
    # Use `is not None` rather than truthiness so that an empty list ([]) is
    # handled correctly and not silently treated as "no filter specified".
    requested = include_tools if include_tools is not None else exclude_tools
    _validate_tool_names(requested, [t.name for t in listed.tools])
    filtered = _filter_mcp_tools(listed.tools, include_tools, exclude_tools)
    return [convert_mcp_tool(session, t) for t in filtered]


@asynccontextmanager
async def stdio_mcp_tools(
    server_params: mcp.StdioServerParameters,
    *,
    include_tools: list[str] | None = None,
    exclude_tools: list[str] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[list[Tool]]:
    """Yield DSPy tools backed by a live MCP stdio session.

    Usage::

        params = StdioServerParameters(command="docker", args=[...])
        async with stdio_mcp_tools(params, exclude_tools=["dangerous"], timeout=30) as tools:
            react = dspy.ReAct(MySignature, tools=tools)
            result = await react.acall(...)

    Args:
        server_params: MCP stdio server parameters.
        include_tools: If set, only include tools whose names are in this list.
        exclude_tools: If set, exclude tools whose names are in this list.
        timeout: Read timeout in seconds for the MCP session. None means no timeout.

    Yields:
        A list of DSPy Tool objects.

    Raises:
        ValueError: If both include_tools and exclude_tools are specified, or if any
            requested tool names are not found on the MCP server.
    """
    _validate_tool_filters(include_tools, exclude_tools)

    from datetime import timedelta

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    read_timeout = timedelta(seconds=timeout) if timeout is not None else None
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=read_timeout) as session:
            yield await _build_tools_from_session(session, include_tools, exclude_tools)


@asynccontextmanager
async def http_mcp_tools(
    url: str,
    *,
    include_tools: list[str] | None = None,
    exclude_tools: list[str] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[list[Tool]]:
    """Yield DSPy tools backed by a live MCP streamable-HTTP session.

    Usage::

        async with http_mcp_tools("http://127.0.0.1:8003/mcp", include_tools=["search"], timeout=30) as tools:
            react = dspy.ReAct(MySignature, tools=tools)
            result = await react.acall(...)

    Args:
        url: The URL of the MCP streamable-HTTP server.
        include_tools: If set, only include tools whose names are in this list.
        exclude_tools: If set, exclude tools whose names are in this list.
        timeout: Read timeout in seconds for the MCP session. None means no timeout.

    Yields:
        A list of DSPy Tool objects.

    Raises:
        ValueError: If both include_tools and exclude_tools are specified, or if any
            requested tool names are not found on the MCP server.
    """
    _validate_tool_filters(include_tools, exclude_tools)

    from datetime import timedelta

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    read_timeout = timedelta(seconds=timeout) if timeout is not None else None
    # Note: timeout is applied as read_timeout_seconds to ClientSession only.
    # streamable_http_client does not currently expose a connection-level timeout parameter.
    async with streamable_http_client(url) as (read, write, _):
        async with ClientSession(read, write, read_timeout_seconds=read_timeout) as session:
            yield await _build_tools_from_session(session, include_tools, exclude_tools)
