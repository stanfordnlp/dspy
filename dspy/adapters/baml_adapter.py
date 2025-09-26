"""
Custom adapter for improving structured outputs using the information from Pydantic models.
Based on the format used by BAML: https://github.com/BoundaryML/baml
"""

import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import format_field_value as original_format_field_value
from dspy.signatures.signature import Signature

# Changing the comment symbol to Python's # rather than other languages' // seems to help
COMMENT_SYMBOL = "#"


def _render_type_str(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Recursively renders a type annotation into a simplified string.

    Args:
        annotation: The type annotation to render
        depth: Current recursion depth (prevents infinite recursion)
        indent: Current indentation level for nested structures
    """
    # Non-nested types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        # Render the non-None part of the union
        type_render = " or ".join([_render_type_str(arg, depth + 1, indent) for arg in non_none_args])
        # Add "or null" if None was part of the union
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Literal[T1, T2, ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T]
    if origin is list:
        # For Pydantic models in lists, use bracket notation
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            # Build inner schema - the Pydantic model inside should use indent level for array contents
            inner_schema = _build_simplified_schema(inner_type, indent + 1, seen_models)
            # Format with proper bracket notation and indentation
            current_indent = "  " * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        else:
            return f"{_render_type_str(inner_type, depth + 1, indent)}[]"

    # dict[T1, T2]
    if origin is dict:
        return f"dict[{_render_type_str(args[0], depth + 1, indent)}, {_render_type_str(args[1], depth + 1, indent)}]"

    # fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Builds a simplified, human-readable schema from a Pydantic model.

    Args:
        pydantic_model: The Pydantic model to build schema for
        indent: Current indentation level
        seen_models: Set to track visited pydantic models (prevents infinite recursion)
    """
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        raise ValueError("BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.")

    # Add `pydantic_model` to `seen_models` with a placeholder value to avoid infinite recursion.
    seen_models.add(pydantic_model)

    lines = []
    current_indent = "  " * indent
    next_indent = "  " * (indent + 1)

    lines.append(f"{current_indent}{{")

    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}{COMMENT_SYMBOL} No fields defined")
    for name, field in fields.items():
        if field.description:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} {field.description}")
        elif field.alias and field.alias != name:
            # If there's an alias but no description, show the alias as a comment
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")

        rendered_type = _render_type_str(field.annotation, indent=indent + 1, seen_models=seen_models)
        line = f"{next_indent}{name}: {rendered_type},"

        lines.append(line)

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


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

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field description for the system message."""
        sections = []

        # Add input field descriptions
        if signature.input_fields:
            sections.append("Your input fields are:")
            for i, (name, field) in enumerate(signature.input_fields.items(), 1):
                type_name = getattr(field.annotation, "__name__", str(field.annotation))
                description = f": {field.description}" if field.description else ":"
                sections.append(f"{i}. `{name}` ({type_name}){description}")

        # Add output field descriptions
        if signature.output_fields:
            sections.append("Your output fields are:")
            for i, (name, field) in enumerate(signature.output_fields.items(), 1):
                type_name = getattr(field.annotation, "__name__", str(field.annotation))
                description = f": {field.description}" if field.description else ":"
                sections.append(f"{i}. `{name}` ({type_name}){description}")

        return "\n".join(sections)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Overrides the base method to generate a simplified schema for Pydantic models."""

        sections = []

        # Add structural explanation
        sections.append(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        )

        # Add input structure section
        if signature.input_fields:
            for name in signature.input_fields.keys():
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"{{{name}}}")
                sections.append("")  # Empty line after each input

        # Add output structure section
        if signature.output_fields:
            for name, field in signature.output_fields.items():
                field_type = field.annotation
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"Output field `{name}` should be of type: {_render_type_str(field_type, indent=0)}\n")

        # Add completed section
        sections.append("[[ ## completed ## ]]")

        return "\n".join(sections)

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
                    formatted_value = original_format_field_value(field_info=field_info, value=value)

                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
