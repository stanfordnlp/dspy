import inspect
from typing import Any, Callable, Optional, get_origin, get_type_hints

from jsonschema import ValidationError, validate
from pydantic import BaseModel, TypeAdapter

from dspy.utils.callback import with_callbacks


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

        Args:
            func (Callable): The actual function that is being wrapped by the tool.
            name (Optional[str], optional): The name of the tool. Defaults to None.
            desc (Optional[str], optional): The description of the tool. Defaults to None.
            args (Optional[dict[str, Any]], optional): The args of the tool, represented as a dictionary
                from arg name to arg's json schema. Defaults to None.
            arg_types (Optional[dict[str, Any]], optional): The argument types of the tool, represented as a dictionary
                from arg name to the type of the argument. Defaults to None.
            arg_desc (Optional[dict[str, str]], optional): Descriptions for each arg, represented as a
                dictionary from arg name to description string. Defaults to None.
        """
        self.func = func
        self.name = name
        self.desc = desc
        self.args = args
        self.arg_types = arg_types
        self.arg_desc = arg_desc

        self._parse_function(func, arg_desc)

    def _resolve_pydantic_schema(self, model: type[BaseModel]) -> dict:
        """Recursively resolve Pydantic model schema, expanding all references."""
        schema = model.model_json_schema()

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

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] = None):
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

        # Process each argument's type to generate its JSON schema.
        for k, v in hints.items():
            arg_types[k] = v
            if k == "return":
                continue
            # Check if the type (or its origin) is a subclass of Pydantic's BaseModel
            origin = get_origin(v) or v
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                # Get json schema, and replace $ref with the actual schema
                v_json_schema = self._resolve_pydantic_schema(v)
                args[k] = v_json_schema
            else:
                args[k] = TypeAdapter(v).json_schema() or "Any"
            if arg_desc and k in arg_desc:
                args[k]["description"] = arg_desc[k]

        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args or args
        self.arg_types = self.arg_types or arg_types

    @with_callbacks
    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.args:
                raise ValueError(f"Arg {k} is not in the tool's args.")
            try:
                instance = v.model_dump() if hasattr(v, "model_dump") else v
                if not self.args[k] == "Any":
                    validate(instance=instance, schema=self.args[k])
            except ValidationError as e:
                raise ValueError(f"Arg {k} is invalid: {e.message}")
        return self.func(**kwargs)
