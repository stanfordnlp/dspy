"""
Custom adapter for improving structured outputs using the information from Pydantic models.
Based on the format used by BAML: https://github.com/BoundaryML/baml
"""
import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import format_field_value as original_format_field_value
from dspy.signatures.signature import Signature
from pydantic import BaseModel


def _render_type_str(annotation: Any, _depth: int = 0, indent: int = 0) -> str:
    """Recursively renders a type annotation into a simplified string.

    Args:
        annotation: The type annotation to render
        _depth: Current recursion depth (prevents infinite recursion)
        indent: Current indentation level for nested structures
    """
    max_depth = 10
    if _depth > max_depth:  # Prevent excessive recursion
        return f"<max depth of {max_depth} exceeded>"

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Handle Optional[T] or T | None
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        # Render the non-None part of the union
        type_render = " or ".join(
            [_render_type_str(arg, _depth + 1, indent) for arg in non_none_args]
        )
        # Add 'or null' if None was part of the union
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Base types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"

    # Composite types
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)
    if origin is list:
        # For Pydantic models in lists, use bracket notation
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            # Build inner schema - the Pydantic model inside should use indent level for array contents
            inner_schema = _build_simplified_schema(inner_type, indent + 1)
            # Format with proper bracket notation and indentation
            current_indent = "  " * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        else:
            return f"{_render_type_str(inner_type, _depth + 1, indent)}[]"
    if origin is dict:
        return f"dict[{_render_type_str(args[0], _depth + 1, indent)}, {_render_type_str(args[1], _depth + 1, indent)}]"

    # Pydantic models (we'll recurse in the main function)
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        try:
            return _build_simplified_schema(annotation, indent)
        except Exception:
            return f"<{annotation.__name__}>"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_simplified_schema(
    model: type[BaseModel], indent: int = 0, _seen: set[type] | None = None
) -> str:
    """Builds a simplified, human-readable schema from a Pydantic model.

    Args:
        model: The Pydantic model to build schema for
        indent: Current indentation level
        _seen: Set to track visited models (prevents infinite recursion)
    """
    if _seen is None:
        _seen = set()

    if model in _seen:
        return f"<circular reference to {model.__name__}>"

    _seen.add(model)

    try:
        lines = []
        current_indent = "  " * indent
        next_indent = "  " * (indent + 1)

        lines.append(f"{current_indent}{{")

        fields = model.model_fields
        if not fields:
            lines.append(f"{next_indent}// No fields defined")
        for name, field in fields.items():
            alias = field.alias or name

            if field.description:
                lines.append(f"{next_indent}// {field.description}")

            # Check for a nested Pydantic model
            field_type_to_render = field.annotation

            # Unpack Optional[T] to get T
            origin = get_origin(field_type_to_render)
            if origin in (types.UnionType, Union):
                non_none_args = [
                    arg
                    for arg in get_args(field_type_to_render)
                    if arg is not type(None)
                ]
                if len(non_none_args) == 1:
                    field_type_to_render = non_none_args[0]

            # Unpack list[T] to get T
            origin = get_origin(field_type_to_render)
            if origin is list:
                field_type_to_render = get_args(field_type_to_render)[0]

            if inspect.isclass(field_type_to_render) and issubclass(
                field_type_to_render, BaseModel
            ):
                # Recursively build schema for nested models with circular reference protection
                nested_schema = _build_simplified_schema(
                    field_type_to_render, indent + 1, _seen
                )
                rendered_type = _render_type_str(
                    field.annotation, indent=indent + 1
                ).replace(field_type_to_render.__name__, nested_schema)
            else:
                rendered_type = _render_type_str(field.annotation, indent=indent + 1)

            line = f"{next_indent}{alias}: {rendered_type},"

            lines.append(line)

        lines.append(f"{current_indent}}}")
        return "\n".join(lines)
    except Exception as e:
        return f"<error building schema for {model.__name__}: {e}>"
    finally:
        _seen.discard(model)


class BAMLAdapter(JSONAdapter):
    """
    A DSPy adapter that improves the rendering of complex/nested Pydantic models to help LMs.

    This adapter generates a compact, human-readable schema representation for nested Pydantic output
    fields, inspired by the BAML project's JSON formatter (https://github.com/BoundaryML/baml).
    The resulting rendered schema is more token-efficient and easier for smaller LMs to follow than a
    raw JSON schema. It also includes Pydantic field descriptions as comments in the schema, which
    provide valuable additional context for the LM to understand the expected output.

    Example Usage:
    ```python
    import dspy
    from pydantic import BaseModel, Field
    from typing import Literal
    from baml_adapter import BAMLAdapter  # Import from your module

    # 1. Define your Pydantic models
    class PatientAddress(BaseModel):
        street: str
        city: str
        country: Literal["US", "CA"]
    class PatientDetails(BaseModel):
        name: str = Field(description="Full name of the patient.")
        age: int
        address: PatientAddress | None

    # 2. Define a signature using the Pydantic model as an output field
    class ExtractPatientInfo(dspy.Signature):
        '''Extract patient information from the clinical note.'''
        clinical_note: str = dspy.InputField()
        patient_info: PatientDetails = dspy.OutputField()

    # 3. Configure dspy to use the new adapter
    llm = dspy.OpenAI(model="gpt-4.1-mini")
    dspy.configure(lm=llm, adapter=BAMLAdapter())

    # 4. Run your program
    extractor = dspy.Predict(ExtractPatientInfo)
    note = "John Doe, 45 years old, lives at 123 Main St, Anytown. Resident of the US."
    result = extractor(clinical_note=note)
    print(result.patient_info)

    # Expected output:
    # PatientDetails(name='John Doe', age=45, address=PatientAddress(street='123 Main St', city='Anytown', country='US'))
    ```
    """

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Overrides the base method to generate a simplified schema for Pydantic models."""

        instruction = "You must produce a single, valid JSON object that strictly adheres to the following schema. Do not output anything else."

        output_schemas = []
        for name, field in signature.output_fields.items():
            field_type = field.annotation
            main_type = field_type

            # Find the core type if it's wrapped in Optional or Union
            origin = get_origin(field_type)
            if origin in (types.UnionType, Union):
                non_none_args = [
                    arg for arg in get_args(field_type) if arg is not type(None)
                ]
                if len(non_none_args) == 1:
                    main_type = non_none_args[0]

            if inspect.isclass(main_type) and issubclass(main_type, BaseModel):
                # We have a pydantic model, so build the simplified schema for it.
                # Assuming the entire output is one JSON object corresponding to this model.
                schema_str = _build_simplified_schema(main_type)
                output_schemas.append(schema_str)
            else:
                # Handle non-pydantic or primitive types simply
                type_str = _render_type_str(field_type, indent=0)
                output_schemas.append(
                    f"Output field `{name}` should be of type: {type_str}"
                )

        # Assuming a single Pydantic model output field is the common case
        return f"{instruction}\n\n[[ ## schema ## ]]:\n" + "\n\n".join(output_schemas)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Overrides the base method to render Pydantic input instances as clean JSON."""
        messages = [prefix]
        for key, field_info in signature.input_fields.items():
            if key in inputs:
                value = inputs.get(key)
                formatted_value = ""
                if isinstance(value, BaseModel):
                    # Use clean, indented JSON for Pydantic instances
                    formatted_value = value.model_dump_json(indent=2, by_alias=True)
                else:
                    # Fallback to the original dspy formatter for other types
                    formatted_value = original_format_field_value(
                        field_info=field_info, value=value
                    )

                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
