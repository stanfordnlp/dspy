import pydantic
import xml.etree.ElementTree as ET
from typing import Any, Dict, Type, get_origin

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.primitives.prediction import Prediction


class XMLAdapter(ChatAdapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None):
        super().__init__(callbacks)

    def format_field_with_value(self, fields_with_values: Dict[FieldInfoWithName, Any]) -> str:
        return self._dict_to_xml(
            {field.name: field_value for field, field_value in fields_with_values.items()},
        )

    def user_message_output_requirements(self, signature: Type[Signature]) -> str:
        # TODO: Add a more detailed message that describes the expected output structure.
        return "Respond with the corresponding output fields wrapped in XML tags."

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

    def _dict_to_xml(self, data: Any, root_tag: str = "output") -> str:
        def _recursive_serializer(obj):
            if isinstance(obj, pydantic.BaseModel):
                if hasattr(obj, 'model_dump'):
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
