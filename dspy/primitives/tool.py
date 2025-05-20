import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, get_origin, get_type_hints

from jsonschema import ValidationError, validate
from pydantic import BaseModel, TypeAdapter, create_model

from dspy.utils.callback import with_callbacks

if TYPE_CHECKING:
    import mcp


class Tool:
    """Tool class.

    This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
    functions for now.
    """

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
        self.func = func
        self.name = name
        self.desc = desc
        self.args = args
        self.arg_types = arg_types
        self.arg_desc = arg_desc
        self.has_kwargs = False

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
                v_json_schema = resolve_json_schema_reference(v.model_json_schema())
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

    def __repr__(self):
        return f"Tool(name={self.name}, desc={self.desc}, args={self.args})"

    def __str__(self):
        desc = f", whose description is <desc>{self.desc}</desc>.".replace("\n", "  ") if self.desc else "."
        arg_desc = f"It takes arguments {self.args} in JSON format."
        return f"{self.name}{desc} {arg_desc}"


def resolve_json_schema_reference(schema: dict) -> dict:
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
