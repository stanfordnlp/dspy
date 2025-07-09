import inspect
import xml.etree.ElementTree as ET
from typing import Any, Dict, Type, get_args, get_origin

import pydantic
from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback


class XMLAdapter(ChatAdapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None, ):
        super().__init__(callbacks)

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
        return self._dict_to_xml(
            {field.name: field_value for field, field_value in fields_with_values.items()},
        )

    def format_field_structure(self, signature: Type[Signature]) -> str:
        """
        Generate comprehensive instructions showing the XML format for both input and output fields.
        This helps the language model understand the expected structure.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        if signature.input_fields:
            parts.append("Inputs will have the following structure:")
            input_structure = self._generate_fields_xml_structure(signature.input_fields)
            parts.append(input_structure)

        parts.append("Outputs will have the following structure:")
        output_structure = self._generate_fields_xml_structure(signature.output_fields)
        parts.append(output_structure)

        return "\n\n".join(parts).strip()

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        """
        Generate a concise reminder of the expected XML output structure for the language model.
        """
        if not signature.output_fields:
            return "Respond with XML tags as specified."

        # Generate compact schema representation
        schemas = []
        for field_name, field_info in signature.output_fields.items():
            schema = self._generate_compact_xml_schema(field_name, field_info.annotation)
            schemas.append(schema)

        if len(schemas) == 1:
            return f"Respond with XML in the following structure: {schemas[0]}"
        else:
            schema_list = ", ".join(schemas)
            return f"Respond with XML containing the following structures: {schema_list}"

    def parse(self, signature: Type[Signature], completion: str) -> dict[str, Any]:
        if isinstance(completion, Prediction):
            completion = completion.completion
        try:
            # Wrap completion in a root tag to handle multiple top-level elements
            root = ET.fromstring(f"<root>{completion}</root>")
            parsed_dict = self._xml_to_dict(root)

            # Create a dynamic Pydantic model for the output fields only
            output_field_definitions = {
                name: (field.annotation, field) for name, field in signature.output_fields.items()
            }
            OutputModel = pydantic.create_model(
                f"{signature.__name__}Output",
                **output_field_definitions,
            )

            # If there's a single output field, the LM might not wrap it in the field name.
            if len(signature.output_fields) == 1:
                field_name = next(iter(signature.output_fields))
                if field_name not in parsed_dict:
                    parsed_dict = {field_name: parsed_dict}

            # Pre-process the dictionary to handle empty list cases
            for name, field in signature.output_fields.items():
                # Check if the field is a list type and the parsed value is an empty string
                if (
                    get_origin(field.annotation) is list
                    and name in parsed_dict
                    and parsed_dict[name] == ""
                ):
                    parsed_dict[name] = []

            # Validate the parsed dictionary against the dynamic output model
            validated_data = OutputModel(**parsed_dict)

            # Return a dictionary of field names to values (which can be Pydantic models)
            return {name: getattr(validated_data, name) for name in signature.output_fields}

        except ET.ParseError as e:
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse XML: {e}",
            ) from e
        except pydantic.ValidationError as e:
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=parsed_dict,
                message=f"Pydantic validation failed: {e}",
            ) from e

    def _generate_fields_xml_structure(self, fields: Dict[str, FieldInfo]) -> str:
        """Generate XML structure representation for a collection of fields."""
        if not fields:
            return ""

        structures = []
        for field_name, field_info in fields.items():
            structure = self._generate_xml_schema_structure(field_name, field_info.annotation)
            structures.append(structure)

        return "\n".join(structures)

    def _generate_xml_schema_structure(self, field_name: str, field_annotation: Type, indent: int = 0) -> str:
        """
        Generate XML schema structure for a field, handling nested models recursively.
        Returns properly indented XML showing the expected structure.
        """
        indent_str = "  " * indent

        # Handle Pydantic models by showing their nested structure
        if (inspect.isclass(field_annotation) and
            issubclass(field_annotation, pydantic.BaseModel) and
            hasattr(field_annotation, "model_fields")):

            lines = [f"{indent_str}<{field_name}>"]
            for sub_field_name, sub_field_info in field_annotation.model_fields.items():
                sub_structure = self._generate_xml_schema_structure(
                    sub_field_name, sub_field_info.annotation, indent + 1
                )
                lines.append(sub_structure)
            lines.append(f"{indent_str}</{field_name}>")
            return "\n".join(lines)

        # Handle lists by showing repeated elements
        elif get_origin(field_annotation) is list:
            args = get_args(field_annotation)
            if args:
                item_type = args[0]
                if (inspect.isclass(item_type) and
                    issubclass(item_type, pydantic.BaseModel) and
                    hasattr(item_type, "model_fields")):
                    # Show nested structure for Pydantic models in lists
                    example = self._generate_xml_schema_structure(field_name, item_type, indent)
                    return f"{example}\n{example}"
                else:
                    # Show simple repeated elements
                    placeholder = self._get_type_placeholder(item_type)
                    return f"{indent_str}<{field_name}>{placeholder}</{field_name}>\n{indent_str}<{field_name}>{placeholder}</{field_name}>"
            else:
                return f"{indent_str}<{field_name}>...</{field_name}>"

        # Handle simple types with type-appropriate placeholders
        else:
            placeholder = self._get_type_placeholder_with_hint(field_annotation, field_name)
            return f"{indent_str}<{field_name}>{placeholder}</{field_name}>"

    def _get_type_placeholder_with_hint(self, type_annotation: Type, field_name: str) -> str:
        """Get a placeholder value with type hint for a field."""
        if type_annotation is str:
            return f"{{{field_name}}}"
        elif type_annotation is int:
            return f"{{{field_name}}}  # must be a single int value"
        elif type_annotation is float:
            return f"{{{field_name}}}  # must be a single float value"
        elif type_annotation is bool:
            return f"{{{field_name}}}  # must be True or False"
        else:
            return f"{{{field_name}}}"

    def _generate_compact_xml_schema(self, field_name: str, field_annotation: Type) -> str:
        """
        Generate a compact XML schema representation for user_message_output_requirements.
        Returns a condensed format like: <person><name>...</name><age>...</age></person>
        """
        # Handle Pydantic models
        if (inspect.isclass(field_annotation) and
            issubclass(field_annotation, pydantic.BaseModel) and
            hasattr(field_annotation, "model_fields")):

            inner_elements = []
            for sub_field_name, sub_field_info in field_annotation.model_fields.items():
                sub_schema = self._generate_compact_xml_schema(sub_field_name, sub_field_info.annotation)
                inner_elements.append(sub_schema)

            inner_content = "".join(inner_elements)
            return f"<{field_name}>{inner_content}</{field_name}>"

        # Handle lists
        elif get_origin(field_annotation) is list:
            args = get_args(field_annotation)
            if args:
                item_type = args[0]
                item_schema = self._generate_compact_xml_schema(field_name, item_type)
                return item_schema  # Lists are represented by repeated elements
            else:
                return f"<{field_name}>...</{field_name}>"

        # Handle simple types
        else:
            return f"<{field_name}>...</{field_name}>"

    def _get_type_placeholder(self, type_annotation: Type) -> str:
        """Get a simple placeholder value for a type."""
        if type_annotation is str:
            return "..."
        elif type_annotation is int:
            return "0"
        elif type_annotation is float:
            return "0.0"
        elif type_annotation is bool:
            return "true"
        else:
            return "..."

    def _dict_to_xml(self, data: Any, root_tag: str = "output") -> str:
        def _recursive_serializer(obj):
            if isinstance(obj, pydantic.BaseModel):
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                return obj.dict()  # Fallback for Pydantic v1
            if isinstance(obj, dict):
                return {k: _recursive_serializer(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_recursive_serializer(i) for i in obj]
            return obj

        data = _recursive_serializer(data)

        def build_element(parent, tag, content):
            if isinstance(content, dict):
                element = ET.SubElement(parent, tag)
                for key, val in content.items():
                    build_element(element, key, val)
            elif isinstance(content, list):
                if not content:  # Handle empty list
                    ET.SubElement(parent, tag)
                for item in content:
                    build_element(parent, tag, item)
            else:
                element = ET.SubElement(parent, tag)
                element.text = str(content)

        root = ET.Element(root_tag)
        if isinstance(data, dict):
            for key, val in data.items():
                build_element(root, key, val)
        else:
            root.text = str(data)

        inner_xml = "".join(ET.tostring(e, encoding="unicode") for e in root)
        return inner_xml

    def _xml_to_dict(self, element: ET.Element) -> Any:
        if not list(element):
            return element.text or ""

        d = {}
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in d:
                if not isinstance(d[child.tag], list):
                    d[child.tag] = [d[child.tag]]
                d[child.tag].append(child_data)
            else:
                d[child.tag] = child_data
        return d
