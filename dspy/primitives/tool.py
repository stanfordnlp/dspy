import inspect
from typing import Any, Callable, get_origin, get_type_hints

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
        name: str = None,
        desc: str = None,
        parameters: dict[str, Any] = None,
        arg_types: dict[str, Any] = None,
        func: Callable = None,
    ):
        """Initialize the Tool class.

        Args:
            name (str): The name of the tool.
            desc (str): The description of the tool.
            parameters (dict[str, Any]): The parameters of the tool, represented as a dictionary from parameter name to
                parameter's json schema.
            arg_types (dict[str, Any]): The argument types of the tool, represented as a dictionary from parameter name
                to the type of the argument.
            func (Callable): The actual function that is being wrapped by the tool.
        """
        self.name = name
        self.desc = desc
        self.parameters = parameters or {}
        self.arg_types = arg_types or {}
        self.func = func

    @staticmethod
    def _resolve_pydantic_schema(model: type[BaseModel]) -> dict:
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

    @classmethod
    def from_function(cls, func: Callable):
        """Class method that converts a python function to a `Tool`.

        This is a helper function that automatically infers the name, description, and parameters of the tool from the
        provided function. In order to make the inference work, the function must have valid type hints.

        Args:
            func (Callable): The function to be wrapped by the tool.

        Returns:
            Tool: The tool object.
        """
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        name = getattr(func, "__name__", type(func).__name__)
        desc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", "")
        parameters = {}
        arg_types = {}
        for k, v in get_type_hints(annotations_func).items():
            arg_types[k] = v
            if k == "return":
                continue
            if isinstance((origin := get_origin(v) or v), type) and issubclass(origin, BaseModel):
                schema = cls._resolve_pydantic_schema(origin)
                parameters[k] = schema
            else:
                parameters[k] = TypeAdapter(v).json_schema()

        return cls(name=name, desc=desc, parameters=parameters, arg_types=arg_types, func=func)

    def convert_to_litellm_tool_format(self):
        """Converts the tool to the format required by litellm for tool calling."""
        parameters = {
            "type": "object",
            "properties": self.parameters,
            "required": list(self.parameters.keys()),
            "additionalProperties": False,
        }

        tool_dict = {
            "type": "function",
            "function": {"name": self.name, "description": self.desc, "parameters": parameters},
        }
        return tool_dict

    @with_callbacks
    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.parameters:
                raise ValueError(f"Parameter {k} is not in the tool's parameters.")
            try:
                instance = v.model_dump() if hasattr(v, "model_dump") else v
                validate(instance=instance, schema=self.parameters[k])
            except ValidationError as e:
                raise ValueError(f"Parameter {k} is invalid: {e.message}")
        return self.func(**kwargs)
