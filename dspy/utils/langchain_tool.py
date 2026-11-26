from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    from langchain.tools import BaseTool


def convert_langchain_tool(tool: "BaseTool") -> Tool:
    """Build a DSPy tool from a LangChain tool.

    This function converts a LangChain tool (either created with @tool decorator
    or by subclassing BaseTool) into a DSPy Tool.

    Args:
        tool: The LangChain tool to convert.

    Returns:
        A DSPy Tool object.
    """
    # args_schema still lists InjectedToolArg fields. tool_call_schema is the
    # schema LangChain sends to the model, and those fields are not in it.
    args_schema = tool.args_schema
    call_schema = getattr(tool, "tool_call_schema", None) or args_schema
    if hasattr(call_schema, "model_json_schema"):
        schema = call_schema.model_json_schema()
    else:
        schema = call_schema
    args, _, arg_desc = convert_input_schema_to_tool_args(schema)
    visible = set(args)
    # A model that sees the error for a missing injected arg can retry and
    # supply its own value. Drop those keys so they never reach ainvoke.
    injected = set(args_schema.model_fields) - visible

    async def func(**kwargs):
        for key in injected:
            kwargs.pop(key, None)
        try:
            result = await tool.ainvoke(kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to call LangChain tool {tool.name}: {e!s}")

    # The args_schema of Langchain tool is a pydantic model, so we can get the type hints from the model fields
    arg_types = {
        key: field.annotation if field.annotation is not None else Any
        for key, field in args_schema.model_fields.items()
        if key in visible
    }

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)
