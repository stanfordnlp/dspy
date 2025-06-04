import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Type, get_origin, get_type_hints

from jsonschema import ValidationError, validate
from pydantic import BaseModel, TypeAdapter, create_model

from dspy.adapters.types.base_type import BaseType
from dspy.utils.callback import with_callbacks

if TYPE_CHECKING:
    import mcp
    from langchain.tools import BaseTool

_TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


class Tool(BaseType):
    """Tool class.

    This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
    functions for now.
    """

    func: Callable
    name: Optional[str] = None
    desc: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    arg_types: Optional[dict[str, Any]] = None
    arg_desc: Optional[dict[str, str]] = None
    has_kwargs: bool = False

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        desc: Optional[str] = None,
        args: Optional[dict[str, Any]] = None,
        arg_types: Optional[dict[str, Any]] = None,
        arg_desc: Optional[dict[str, str]] = None,
    ):
        """Initialize the Tool class.

        Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `dspy.Tool`
        automatically infer the values from the function. For values that are specified by the user, automatic inference
        will not be performed on them.

        Args:
            func (Callable): The actual function that is being wrapped by the tool.
            name (Optional[str], optional): The name of the tool. Defaults to None.
            desc (Optional[str], optional): The description of the tool. Defaults to None.
            args (Optional[dict[str, Any]], optional): The args and their schema of the tool, represented as a
                dictionary from arg name to arg's json schema. Defaults to None.
            arg_types (Optional[dict[str, Any]], optional): The argument types of the tool, represented as a dictionary
                from arg name to the type of the argument. Defaults to None.
            arg_desc (Optional[dict[str, str]], optional): Descriptions for each arg, represented as a
                dictionary from arg name to description string. Defaults to None.

        Example:

        ```python
        def foo(x: int, y: str = "hello"):
            return str(x) + y

        tool = Tool(foo)
        print(tool.args)
        # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
        ```
        """
        super().__init__(func=func, name=name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)
        self._parse_function(func, arg_desc)

    def _parse_function(self, func: Callable, arg_desc: Optional[dict[str, str]] = None):
        """Helper method that parses a function to extract the name, description, and args.

        This is a helper function that automatically infers the name, description, and args of the tool from the
        provided function. In order to make the inference work, the function must have valid type hints.
        """
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        name = getattr(func, "__name__", type(func).__name__)
        desc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", "")
        args = {}
        arg_types = {}

        # Use inspect.signature to get all arg names
        sig = inspect.signature(annotations_func)
        # Get available type hints
        available_hints = get_type_hints(annotations_func)
        # Build a dictionary of arg name -> type (defaulting to Any when missing)
        hints = {param_name: available_hints.get(param_name, Any) for param_name in sig.parameters.keys()}
        default_values = {param_name: sig.parameters[param_name].default for param_name in sig.parameters.keys()}

        # Process each argument's type to generate its JSON schema.
        for k, v in hints.items():
            arg_types[k] = v
            if k == "return":
                continue
            # Check if the type (or its origin) is a subclass of Pydantic's BaseModel
            origin = get_origin(v) or v
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                # Get json schema, and replace $ref with the actual schema
                v_json_schema = _resolve_json_schema_reference(v.model_json_schema())
                args[k] = v_json_schema
            else:
                args[k] = TypeAdapter(v).json_schema()
            if default_values[k] is not inspect.Parameter.empty:
                args[k]["default"] = default_values[k]
            if arg_desc and k in arg_desc:
                args[k]["description"] = arg_desc[k]

        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args or args
        self.arg_types = self.arg_types or arg_types
        self.has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    def _validate_and_parse_args(self, **kwargs):
        # Validate the args value comply to the json schema.
        for k, v in kwargs.items():
            if k not in self.args:
                if self.has_kwargs:
                    continue
                else:
                    raise ValueError(f"Arg {k} is not in the tool's args.")
            try:
                instance = v.model_dump() if hasattr(v, "model_dump") else v
                type_str = self.args[k].get("type")
                if type_str is not None and type_str != "Any":
                    validate(instance=instance, schema=self.args[k])
            except ValidationError as e:
                raise ValueError(f"Arg {k} is invalid: {e.message}")

        # Parse the args to the correct type.
        parsed_kwargs = {}
        for k, v in kwargs.items():
            if k in self.arg_types and self.arg_types[k] != Any:
                # Create a pydantic model wrapper with a dummy field `value` to parse the arg to the correct type.
                # This is specifically useful for handling nested Pydantic models like `list[list[MyPydanticModel]]`
                pydantic_wrapper = create_model("Wrapper", value=(self.arg_types[k], ...))
                parsed = pydantic_wrapper.model_validate({"value": v})
                parsed_kwargs[k] = parsed.value
            else:
                parsed_kwargs[k] = v
        return parsed_kwargs

    def format(self):
        return str(self)

    def format_as_litellm_function_call(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {
                    "type": "object",
                    "properties": self.args,
                    "required": list(self.args.keys()),
                },
            },
        }

    @with_callbacks
    def __call__(self, **kwargs):
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            raise ValueError("You are calling `__call__` on an async tool, please use `acall` instead.")
        return result

    @with_callbacks
    async def acall(self, **kwargs):
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            return await result
        else:
            # We should allow calling a sync tool in the async path.
            return result

    @classmethod
    def from_mcp_tool(cls, session: "mcp.client.session.ClientSession", tool: "mcp.types.Tool") -> "Tool":
        """
        Build a DSPy tool from an MCP tool and a ClientSession.

        Args:
            session: The MCP session to use.
            tool: The MCP tool to convert.

        Returns:
            A Tool object.
        """
        from dspy.utils.mcp import convert_mcp_tool

        return convert_mcp_tool(session, tool)

    @classmethod
    def from_langchain(cls, tool: "BaseTool") -> "Tool":
        """
        Build a DSPy tool from a LangChain tool.

        Args:
            tool: The LangChain tool to convert.

        Returns:
            A Tool object.

        Example:

        ```python
        import asyncio
        import dspy
        from langchain.tools import tool as lc_tool

        @lc_tool
        def add(x: int, y: int):
            "Add two numbers together."
            return x + y

        dspy_tool = dspy.Tool.from_langchain(add)

        async def run_tool():
            return await dspy_tool.acall(x=1, y=2)

        print(asyncio.run(run_tool()))
        # 3
        ```
        """
        from dspy.utils.langchain_tool import convert_langchain_tool

        return convert_langchain_tool(tool)

    def __repr__(self):
        return f"Tool(name={self.name}, desc={self.desc}, args={self.args})"

    def __str__(self):
        desc = f", whose description is <desc>{self.desc}</desc>.".replace("\n", "  ") if self.desc else "."
        arg_desc = f"It takes arguments {self.args}."
        return f"{self.name}{desc} {arg_desc}"


class ToolCalls(BaseType):
    class ToolCall(BaseModel):
        name: str
        args: dict[str, Any]

    tool_calls: list[ToolCall]

    @classmethod
    def from_dict_list(cls, tool_calls_dicts: list[dict[str, Any]]) -> "ToolCalls":
        """Convert a list of dictionaries to a ToolCalls instance.

        Args:
            dict_list: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

        Returns:
            A ToolCalls instance.

        Example:

            ```python
            tool_calls_dict = [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world"}}
            ]
            tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
            ```
        """
        tool_calls = [cls.ToolCall(**item) for item in tool_calls_dicts]
        return cls(tool_calls=tool_calls)

    @classmethod
    def description(cls) -> str:
        return (
            "Tool calls information, including the name of the tools and the arguments to be passed to it. "
            "Arguments must be provided in JSON format."
        )


def _resolve_json_schema_reference(schema: dict) -> dict:
    """Recursively resolve json model schema, expanding all references."""

    # If there are no definitions to resolve, return the main schema
    if "$defs" not in schema and "definitions" not in schema:
        return schema

    def resolve_refs(obj: Any) -> Any:
        if not isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"].split("/")[-1]
                return resolve_refs(schema["$defs"][ref_path])
            return {k: resolve_refs(v) for k, v in obj.items()}

        # Must be a list
        return [resolve_refs(item) for item in obj]

    # Resolve all references in the main schema
    resolved_schema = resolve_refs(schema)
    # Remove the $defs key as it's no longer needed
    resolved_schema.pop("$defs", None)
    return resolved_schema


def convert_input_schema_to_tool_args(
    schema: dict[str, Any],
) -> Tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
    """Convert an input json schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: An input json schema describing the tool's input parameters

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
            prop = _resolve_json_schema_reference({"$defs": defs, **prop})
        args[name] = prop
        arg_types[name] = _TYPE_MAPPING.get(prop.get("type"), Any)
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in required:
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc
