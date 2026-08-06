import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _mcp_major_version() -> int:
    try:
        return int(_package_version("mcp").split(".")[0])
    except (PackageNotFoundError, ValueError):
        return 1


_MCP_MAJOR_VERSION = _mcp_major_version()


def _get_field(obj: Any, v1_name: str, v2_name: str, default: Any = None) -> Any:
    """Read a field that was renamed between mcp SDK v1 (camelCase) and v2 (snake_case)."""
    primary, fallback = (v2_name, v1_name) if _MCP_MAJOR_VERSION >= 2 else (v1_name, v2_name)
    for name in (primary, fallback):
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _is_wrapped_result_schema(output_schema: dict[str, Any] | None) -> bool:
    """Detect the single-value envelope both official SDK servers generate.

    Servers built on FastMCP (v1) or MCPServer (v2) wrap non-object return values as
    ``{"result": value}``, with an output schema declaring exactly one required
    ``result`` property.
    """
    if not isinstance(output_schema, dict):
        return False
    properties = output_schema.get("properties")
    return (
        output_schema.get("type") == "object"
        and isinstance(properties, dict)
        and set(properties) == {"result"}
        and output_schema.get("required") == ["result"]
    )


def _matches_serialized(text: str, value: Any) -> bool:
    if text == str(value):
        return True
    try:
        return json.loads(text) == value
    except ValueError:
        return False


def _text_corroborates_envelope(text_contents: list[Any], inner: Any) -> bool:
    """Distinguish the SDK's wrapper envelope from a genuine single-``result``-field object.

    The two are indistinguishable by schema shape alone, so unwrapping requires positive
    evidence. SDK servers always emit text content alongside a wrapped value: the inner
    value's serialization as a single entry, or one entry per element for list returns.
    A genuine object's text renders as the full JSON object instead, and a result with
    no text content at all most likely did not come from an SDK wrapper.
    """
    texts = [content.text for content in text_contents]
    if len(texts) == 1 and _matches_serialized(texts[0], inner):
        return True
    if isinstance(inner, list) and len(texts) == len(inner):
        return all(_matches_serialized(text, element) for text, element in zip(texts, inner, strict=False))
    return False


def _convert_mcp_tool_result(
    call_tool_result: "mcp.types.CallToolResult",
    output_schema: dict[str, Any] | None = None,
) -> Any:
    from mcp.types import TextContent

    if getattr(call_tool_result, "result_type", "complete") == "input_required":
        requests = getattr(call_tool_result, "input_requests", None) or []
        requested = ", ".join(str(getattr(request, "method", request)) for request in requests)
        raise RuntimeError(
            "The MCP server needs additional input before it can complete this tool call "
            f"(requested: {requested or 'unknown'}). DSPy's MCP bridge cannot answer "
            "server-initiated input requests; use an MCP client with input handlers to call this tool."
        )

    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[Any] = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if _get_field(call_tool_result, "isError", "is_error", default=False):
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    structured_content = _get_field(call_tool_result, "structuredContent", "structured_content")
    if structured_content is not None:
        if (
            _is_wrapped_result_schema(output_schema)
            and isinstance(structured_content, dict)
            and set(structured_content) == {"result"}
            and _text_corroborates_envelope(text_contents, structured_content["result"])
        ):
            return structured_content["result"]
        return structured_content

    return tool_content or non_text_contents


def convert_mcp_tool(session: "mcp.ClientSession", tool: "mcp.types.Tool") -> Tool:
    """Build a DSPy tool from an MCP tool.

    Works with both mcp SDK v1 (spec revisions up to 2025-11-25) and v2 (spec
    revision 2026-07-28 and later). The session may be a ``ClientSession`` or any
    object exposing an equivalent async ``call_tool(name, arguments=...)`` method,
    such as the v2 ``mcp.client.Client``.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    input_schema = _get_field(tool, "inputSchema", "input_schema", default={})
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(input_schema)

    output_schema = _get_field(tool, "outputSchema", "output_schema")
    desc = tool.description or getattr(tool, "title", None)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result, output_schema)

    return Tool(func=func, name=tool.name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)
