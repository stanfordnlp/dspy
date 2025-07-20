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
    async def func(**kwargs):
        try:
            result = await tool.ainvoke(kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to call LangChain tool {tool.name}: {e!s}")

    # Get args_schema from the tool
    # https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html#langchain_core.tools.base.BaseTool.args_schema
    args_schema = tool.args_schema
    args, _, arg_desc = convert_input_schema_to_tool_args(args_schema.model_json_schema())

    # The args_schema of Langchain tool is a pydantic model, so we can get the type hints from the model fields
    arg_types = {
        key: field.annotation if field.annotation is not None else Any
        for key, field in args_schema.model_fields.items()
    }

    return Tool(
        func=func,
        name=tool.name,
        desc=tool.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc
    )
