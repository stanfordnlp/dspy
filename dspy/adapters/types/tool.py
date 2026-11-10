import asyncio
import inspect
import json
from decimal import Decimal
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, get_origin, get_type_hints
from urllib.parse import quote

import json_repair
import pydantic
import pydantic_core
from jsonschema import ValidationError, validate
from pydantic import BaseModel, TypeAdapter, create_model

from dspy.adapters.types.base_type import Type
from dspy.dsp.utils.settings import settings
from dspy.lm15 import FunctionTool
from dspy.utils.callback import with_callbacks

if TYPE_CHECKING:
    import mcp
    from langchain.tools import BaseTool

_TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


class _MCPToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


def _json_schema_instance(value: Any, bytes_mode: Literal["utf8", "base64", "hex"] | None = None) -> Any:
    """``value`` as the JSON its schema describes, for JSON-schema validation. A model is checked as its fields, keyed
    by the names its validation schema uses (its dump would key them by serialization alias, and a custom type's dump
    is a prompt)."""
    if isinstance(value, pydantic.RootModel):
        return _json_schema_instance(value.root, type(value).model_config.get("ser_json_bytes", "utf8"))
    if isinstance(value, BaseModel):
        # Only the fields its JSON schema shows; e.g. Score's provider-evidence fields are hidden from it.
        shown = type(value).model_json_schema().get("properties")
        # A plain model's round-trip dump is the input its schema describes (a `Json` field as text). Python mode keeps
        # JSON-only serializers, which may change a field's type, out of the check. A custom type's dump is a prompt,
        # and a field holding a model is checked through that model, keyed as its schema is.
        own = {} if isinstance(value, Type) else value.model_dump(round_trip=True)
        # The schema describes bytes as a string in every mode, so encode them as the model's JSON does.
        bytes_mode = type(value).model_config.get("ser_json_bytes", "utf8")
        fields = {}
        for name, info in type(value).model_fields.items():
            # Key each field by the name its validation schema uses: a validation alias (or one of its choices), else the name.
            alias = info.validation_alias
            choices = alias.choices if isinstance(alias, pydantic.AliasChoices) else [alias]
            keys = [key for key in (*choices, info.alias, name) if isinstance(key, str)]
            key = next((key for key in keys if shown is None or key in shown), None)
            if key is not None:
                item = getattr(value, name)
                fields[key] = _json_field_value(own[name] if name in own and not _holds_model(item) else _json_schema_instance(item), bytes_mode)
        extra = {key: _json_field_value(_json_schema_instance(item), bytes_mode) for key, item in (value.model_extra or {}).items()}
        return fields | extra
    if isinstance(value, dict):
        return {k: _json_schema_instance(v, bytes_mode) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_schema_instance(v, bytes_mode) for v in value]
    # Root values need JSON conversion too, but numbers already support numeric schema constraints.
    if bytes_mode is not None and not isinstance(value, Number):
        return _json_field_value(value, bytes_mode)
    return value


def _is_json(value: Any) -> bool:
    """Whether ``value`` is plain JSON data: dicts with string keys, lists, strings, numbers, booleans and None. Not
    their subclasses, like a defaultdict, which JSON can't carry."""
    if type(value) is dict:
        return all(type(key) is str and _is_json(item) for key, item in value.items())
    if type(value) is list:
        return all(_is_json(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


def _holds_model(value: Any) -> bool:
    if isinstance(value, BaseModel):
        return True
    items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple, set, frozenset)) else ()
    return any(_holds_model(item) for item in items)


def _json_field_value(value: Any, bytes_mode: Literal["utf8", "base64", "hex"]) -> Any:
    """A model's field value in the JSON form its schema describes, e.g. a date as an ISO string."""
    try:
        return pydantic_core.to_jsonable_python(value, bytes_mode=bytes_mode)
    except (pydantic_core.PydanticSerializationError, ValueError):
        return value  # e.g. an arbitrary object, which the schema leaves unchecked


class Tool(Type):
    """Tool class.

    This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
    functions for now.
    """

    func: Callable
    name: str | None = None
    desc: str | None = None
    args: dict[str, Any] | None = None
    arg_types: dict[str, Any] | None = None
    arg_desc: dict[str, str] | None = None
    has_kwargs: bool = False

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        desc: str | None = None,
        args: dict[str, Any] | None = None,
        arg_types: dict[str, Any] | None = None,
        arg_desc: dict[str, str] | None = None,
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

        Examples:

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

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] | None = None):
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
                args[k] = _resolve_json_schema_reference(TypeAdapter(v).json_schema())
            if default_values[k] is not inspect.Parameter.empty:
                args[k]["default"] = _to_json_default(default_values[k])
            if arg_desc and k in arg_desc:
                args[k]["description"] = arg_desc[k]

        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args if self.args is not None else args
        self.arg_types = self.arg_types if self.arg_types is not None else arg_types
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
                instance = _json_schema_instance(v)
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
                if type(v) in (dict, list) and _is_json(v):
                    # JSON data, as an LM or a sandbox sends: read as JSON, where a strict model's tuple is a list.
                    parsed = pydantic_wrapper.model_validate_json(json.dumps({"value": v}))
                else:
                    parsed = pydantic_wrapper.model_validate({"value": v})
                parsed_kwargs[k] = parsed.value
            else:
                parsed_kwargs[k] = v
        return parsed_kwargs

    def format(self):
        return str(self)

    def as_function_tool(self) -> FunctionTool:
        """The lm15 tool declaration sent to models that call tools natively."""
        def rebase_refs(schema: Any, prefix: str) -> Any:
            if not isinstance(schema, dict) or "$id" in schema:
                return schema
            result = dict(schema)
            ref = result.get("$ref", "")
            if ref == "#" or ref.startswith("#/"):
                result["$ref"] = prefix + ref[1:]
            # Visit schemas, not literal data in defaults, const, enum, examples, or extensions.
            for key in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas", "dependencies"):
                if key in result:
                    result[key] = {name: rebase_refs(item, prefix) for name, item in result[key].items()}
            for key in (
                "items", "additionalItems", "additionalProperties", "contains", "propertyNames",
                "unevaluatedItems", "unevaluatedProperties", "not", "if", "then", "else",
                "allOf", "anyOf", "oneOf", "prefixItems",
            ):
                if key in result:
                    item = result[key]
                    result[key] = [rebase_refs(v, prefix) for v in item] if isinstance(item, list) else rebase_refs(item, prefix)
            return result

        # Each argument is a standalone schema. Its local references must follow it under properties.
        properties = {
            name: rebase_refs(schema, "#/properties/" + quote(name.replace("~", "~0").replace("/", "~1"), safe=""))
            for name, schema in self.args.items()
        }
        return FunctionTool(
            name=self.name,
            description=self.desc,
            parameters={
                "type": "object",
                "properties": properties,
                "required": [k for k in self.args if "default" not in self.args[k]],
            },
        )

    def _run_async_in_sync(self, coroutine):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Run the coroutine outside of "except" block to avoid propagation
            loop = None

        if loop is None:
            return asyncio.run(coroutine)

        coroutine.close()
        raise ValueError(
            "You are calling `__call__` on an async tool from within a running event loop, which cannot be "
            "converted to a sync call, even with `allow_tool_async_sync_conversion` enabled. Please use "
            "`await tool.acall(...)` instead."
        )

    @with_callbacks
    def __call__(self, **kwargs):
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            if settings.allow_tool_async_sync_conversion:
                return self._run_async_in_sync(result)
            else:
                raise ValueError(
                    "You are calling `__call__` on an async tool, please use `acall` instead or enable "
                    "async-to-sync conversion with `dspy.configure(allow_tool_async_sync_conversion=True)` "
                    "or `with dspy.context(allow_tool_async_sync_conversion=True):`."
                )
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
    def from_mcp_tool(
        cls,
        session: _MCPToolClient,
        tool: "mcp.types.Tool",
        *,
        result_mode: Literal["text", "structured"] = "text",
    ) -> "Tool":
        """
        Build a DSPy tool from an MCP tool and a compatible MCP client.

        Args:
            session: An MCP client or session with an async ``call_tool`` method.
            tool: The MCP tool to convert.
            result_mode: ``"text"`` preserves DSPy's existing text/non-text
                conversion. ``"structured"`` returns MCP structured content
                exactly when present, with the existing conversion as fallback.

        Returns:
            A Tool object.
        """
        from dspy.utils.mcp import convert_mcp_tool

        return convert_mcp_tool(session, tool, result_mode=result_mode)

    @classmethod
    def from_langchain(cls, tool: "BaseTool") -> "Tool":
        """
        Build a DSPy tool from a LangChain tool.

        Args:
            tool: The LangChain tool to convert.

        Returns:
            A Tool object.

        Examples:

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


class ToolCalls(Type):
    class ToolCall(Type):
        id: str | None = None
        name: str
        args: dict[str, Any] = pydantic.Field(json_schema_extra={"additionalProperties": True})

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
            schema = super().__get_pydantic_json_schema__(core_schema, handler)
            schema = handler.resolve_ref_schema(schema)
            properties = schema.get("properties")
            if isinstance(properties, dict):
                properties.pop("id", None)
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [field for field in required if field != "id"]
            return schema

        def format(self):
            return {"name": self.name, "args": self.args}

        def execute(self, functions: dict[str, Any] | list[Tool] | None = None) -> Any:
            """Execute this individual tool call and return its result.

            Args:
                functions: Functions to search for the tool. Can be:
                          - Dict mapping tool names to functions: {"tool_name": function}
                          - List of Tool objects: [Tool(function), ...]
                          - None: Will search in caller's locals and globals (automatic lookup)

            Returns:
                The result from executing this tool call.

            Raises:
                ValueError: If the tool function cannot be found.
                Exception: Any exception raised by the tool function.
            """
            func = None

            if functions is None:
                # Automatic lookup in caller's globals and locals
                frame = inspect.currentframe().f_back
                try:
                    caller_globals = frame.f_globals
                    caller_locals = frame.f_locals
                    func = caller_locals.get(self.name) or caller_globals.get(self.name)
                finally:
                    del frame

            elif isinstance(functions, dict):
                func = functions.get(self.name)
            elif isinstance(functions, list):
                for tool in functions:
                    if tool.name == self.name:
                        func = tool.func
                        break

            if func is None:
                raise ValueError(f"Tool function '{self.name}' not found. Please pass the tool functions to the `execute` method.")

            try:
                args = self.args or {}
                return func(**args)
            except Exception as e:
                raise RuntimeError(f"Error executing tool '{self.name}': {e}") from e

    tool_calls: list[ToolCall]
    tool_call_results: Any | None = None

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
        schema = handler.resolve_ref_schema(schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("tool_call_results", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [field for field in required if field != "tool_call_results"]
        return schema

    @classmethod
    def from_dict_list(cls, tool_calls_dicts: list[dict[str, Any]]) -> "ToolCalls":
        """Convert a list of dictionaries to a ToolCalls instance.

        Args:
            tool_calls_dicts: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

        Returns:
            A ToolCalls instance.

        Examples:

            ```python
            tool_calls_dict = [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world"}}
            ]
            tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
            ```
        """
        tool_calls = [cls.ToolCall(**_normalize_tool_call_dict(item)) for item in tool_calls_dicts]
        return cls(tool_calls=tool_calls)

    @classmethod
    def description(cls) -> str:
        return (
            "Tool calls must be a JSON object with `tool_calls`, a list of calls. "
            "Each call must include `name` and `args`. "
            'Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}'
        )

    def format(self) -> dict[str, Any]:
        return {
            "tool_calls": [tool_call.format() for tool_call in self.tool_calls],
        }

    @pydantic.model_serializer()
    def serialize_model(self):
        data = self.format()
        if self.tool_call_results is not None:
            data["tool_call_results"] = TypeAdapter(type(self.tool_call_results)).dump_python(
                self.tool_call_results,
                mode="json",
            )
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        # Handle case where data is a list of dicts with either DSPy or provider-shaped tool call keys.
        if isinstance(data, list) and all(isinstance(item, dict) and _is_tool_call_dict(item) for item in data):
            return {"tool_calls": [cls.ToolCall(**_normalize_tool_call_dict(item)) for item in data]}
        # Handle case where data is a dict
        elif isinstance(data, dict):
            if "tool_calls" in data:
                # Handle case where data is a dict with "tool_calls" key
                tool_calls_data = data["tool_calls"]
                if isinstance(tool_calls_data, list):
                    normalized = {
                        "tool_calls": [
                            cls.ToolCall(**_normalize_tool_call_dict(item)) if isinstance(item, dict) else item
                            for item in tool_calls_data
                        ]
                    }
                    if "tool_call_results" in data:
                        normalized["tool_call_results"] = data["tool_call_results"]
                    return normalized
                # Models in JSON mode write `"tool_calls": null` to mean "no
                # tool calls this turn"; treat it like an empty list.
                if tool_calls_data is None:
                    return {"tool_calls": []}
            elif _is_tool_call_dict(data):
                # Handle case where data is a dict with "name" and "args" keys
                return {"tool_calls": [cls.ToolCall(**_normalize_tool_call_dict(data))]}
            elif not data:
                # An empty dict is also "no tool calls".
                return {"tool_calls": []}

        # `None` (models in JSON mode) means "no tool calls this turn" and
        # normalizes to an empty call list, matching the native-path behavior
        # where the field stays None when the provider returns no calls.
        if data is None:
            return {"tool_calls": []}

        raise ValueError(f"Received invalid value for `dspy.ToolCalls`: {data}")


class ToolCallResults(pydantic.BaseModel):
    class ToolCallResult(pydantic.BaseModel):
        call_id: str | None = None
        name: str
        value: Any
        is_error: bool = False

    tool_call_results: list[ToolCallResult]

    @classmethod
    def from_tool_calls_and_values(
        cls,
        tool_calls: list[ToolCalls.ToolCall] | ToolCalls,
        values: list[Any],
        is_errors: list[bool] | None = None,
    ) -> "ToolCallResults":
        if isinstance(tool_calls, ToolCalls):
            tool_calls = tool_calls.tool_calls

        if len(tool_calls) != len(values):
            raise ValueError("`tool_calls` and `values` must have the same length.")

        if is_errors is None:
            is_errors = [False] * len(tool_calls)
        elif len(is_errors) != len(tool_calls):
            raise ValueError("`is_errors` must have the same length as `tool_calls` when provided.")

        return cls(
            tool_call_results=[
                cls.ToolCallResult(call_id=tool_call.id, name=tool_call.name, value=value, is_error=is_error)
                for tool_call, value, is_error in zip(tool_calls, values, is_errors, strict=True)
            ]
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, list):
            return {"tool_call_results": data}

        if isinstance(data, dict):
            if "tool_call_results" in data:
                return data
            if {"name", "value"}.issubset(data):
                return {"tool_call_results": [data]}

        raise ValueError(f"Received invalid value for `dspy.ToolCallResults`: {data}")


def _is_tool_call_dict(data: dict[str, Any]) -> bool:
    return ("name" in data and ("args" in data or "arguments" in data)) or "function" in data


def _normalize_tool_call_dict(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Received invalid tool call value for `dspy.ToolCalls`: {data}")

    if "function" in data:
        function = data.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError(f"Received invalid function value for `dspy.ToolCalls`: {function}")

        arguments = function.get("arguments", {})
        name = function.get("name") or data.get("name")
    else:
        arguments = data.get("args", data.get("arguments", {}))
        name = data.get("name")

    if isinstance(arguments, str):
        arguments = json_repair.loads(arguments)
    elif not isinstance(arguments, dict):
        arguments = {}

    return {
        "id": data.get("id") or data.get("call_id"),
        "name": name,
        "args": arguments,
    }


def _to_json_default(value: Any) -> Any:
    """Convert a parameter default to its JSON form, e.g. an Enum member to its value."""
    if isinstance(value, Decimal):
        # A JSON number fits both float and Decimal arguments; pydantic would give a string.
        return float(value)
    try:
        return pydantic_core.to_jsonable_python(value)
    except Exception:
        return value


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
) -> tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
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
