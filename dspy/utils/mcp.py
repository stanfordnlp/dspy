from typing import TYPE_CHECKING, Any, Tuple, Type, Union

from dspy.primitives.tool import Tool, resolve_json_schema_reference

if TYPE_CHECKING:
    import mcp

TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _convert_input_schema_to_tool_args(
    schema: dict[str, Any],
) -> Tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
    """Convert an input schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: An input schema describing the tool's input parameters

    Returns:
        A tuple of (args, arg_types, arg_desc) for DSPy Tool definition.
    """
    args, arg_types, arg_desc = {}, {}, {}
    properties = schema.get("properties", None)
    if properties is None:
        return args, arg_types, arg_desc

    required = schema.get("required", [])

    defs = schema.get("$defs", {})

    for name, prop in properties.items():
        if len(defs) > 0:
            prop = resolve_json_schema_reference({"$defs": defs, **prop})
        args[name] = prop
        # MCP tools are validated through jsonschema using args, so arg_types are not strictly required.
        arg_types[name] = TYPE_MAPPING.get(prop.get("type"), Any)
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in required:
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc


def _convert_mcp_tool_result(call_tool_result: "mcp.types.CallToolResult") -> Union[str, list[Any]]:
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


def convert_mcp_tool(session: "mcp.client.session.ClientSession", tool: "mcp.types.Tool") -> Tool:
    """Build a DSPy tool from an MCP tool.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    args, arg_types, arg_desc = _convert_input_schema_to_tool_args(tool.inputSchema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)
