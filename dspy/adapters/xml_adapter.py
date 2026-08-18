import re
import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Union, get_args, get_origin, get_type_hints

import pydantic
from pydantic.fields import FieldInfo
from typing_extensions import is_typeddict

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.types.base_type import CUSTOM_TYPE_END_IDENTIFIER, CUSTOM_TYPE_START_IDENTIFIER
from dspy.adapters.utils import (
    apply_output_field_defaults,
    format_field_value,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError


class XMLAdapter(ChatAdapter):
    """Format and parse signature fields as XML.

    Structured values use nested elements, with repeated elements representing lists. JSON values
    inside a field remain accepted for backwards compatibility. XML text is escaped when DSPy
    formats examples, and malformed model output raises ``AdapterParseError`` instead of returning
    a silently truncated value.
    """

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            serialized = serialize_for_json(field_value)
            is_output = (field.info.json_schema_extra or {}).get("__dspy_field_type") == "output"
            if is_output and self._uses_nested_xml(field.info.annotation) and isinstance(serialized, (dict, list)):
                output.append(self._value_to_xml(serialized, field.name))
            else:
                formatted = format_field_value(field_info=field.info, value=field_value)
                if field.info.annotation is str:
                    formatted = self._escape_text(formatted)
                output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def format_field_structure(self, signature: type[Signature]) -> str:
        """XMLAdapter requires signature fields to be wrapped in XML tags."""
        parts = ["All interactions will be structured in the following way, with the appropriate values filled in."]

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            formatted_fields = []
            for field_name, field_info in fields.items():
                is_output = (field_info.json_schema_extra or {}).get("__dspy_field_type") == "output"
                if is_output and self._uses_nested_xml(field_info.annotation):
                    formatted_fields.append(self._xml_schema(field_name, field_info.annotation))
                else:
                    formatted_fields.append(
                        self.format_field_with_value(
                            {
                                FieldInfoWithName(name=field_name, info=field_info): translate_field_type(
                                    field_name, field_info
                                )
                            }
                        )
                    )
            return "\n\n".join(formatted_fields)

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        return "\n\n".join(parts).strip()

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]

        messages.append(
            self.format_field_with_value(
                {
                    FieldInfoWithName(name=k, info=v): inputs.get(k)
                    for k, v in signature.input_fields.items()
                    if k in inputs
                },
            )
        )

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        return self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in signature.output_fields)
        message += "."
        nested_fields = [
            self._xml_schema(name, field.annotation)
            for name, field in signature.output_fields.items()
            if self._uses_nested_xml(field.annotation)
        ]
        if nested_fields:
            message += f" Use this nested XML structure: {' '.join(nested_fields)}"
        return message

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        try:
            root = ET.fromstring(f"<dspy_root>{completion}</dspy_root>")
        except ET.ParseError as e:
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse XML: {e}",
            ) from e

        elements_by_name: dict[str, list[ET.Element]] = defaultdict(list)
        for element in root:
            if element.tag in signature.output_fields:
                elements_by_name[element.tag].append(element)

        fields = {}
        for name, field_info in signature.output_fields.items():
            elements = elements_by_name.get(name)
            if not elements:
                continue
            value: Any = None
            try:
                value = self._elements_to_value(elements, field_info.annotation)
                fields[name] = parse_value(value, field_info.annotation)
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="XMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {field_info} with value {value}: {e}",
                ) from e

        fields = apply_output_field_defaults(signature, fields)
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        return fields

    @classmethod
    def _value_to_xml(cls, value: Any, tag: str) -> str:
        if isinstance(value, list):
            if not value:
                return f"<{tag} />"
            return "".join(cls._value_to_xml(item, tag) for item in value)

        if isinstance(value, dict):
            inner = "".join(cls._value_to_xml(child, str(name)) for name, child in value.items())
            return f"<{tag}>{inner}</{tag}>"

        if value is None:
            return f"<{tag} />"
        return f"<{tag}>{cls._escape_text(str(value))}</{tag}>"

    @classmethod
    def _xml_schema(cls, tag: str, annotation: Any, expanded_types: frozenset[type] = frozenset()) -> str:
        annotation = cls._unwrap_optional(annotation)
        field_annotations = cls._structured_field_annotations(annotation)
        if field_annotations is not None:
            if annotation in expanded_types:
                return f"<{tag}>...</{tag}>"
            expanded_types |= {annotation}
            inner = "".join(
                cls._xml_schema(name, field_annotation, expanded_types)
                for name, field_annotation in field_annotations.items()
            )
            return f"<{tag}>{inner}</{tag}>"
        if get_origin(annotation) is list:
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            item = cls._xml_schema(tag, item_annotation, expanded_types)
            return f"{item} {item}"
        return f"<{tag}>...</{tag}>"

    @staticmethod
    def _escape_text(value: str) -> str:
        """Escape XML text while retaining DSPy's private multimodal markers."""
        marker_pattern = re.compile(
            f"({re.escape(CUSTOM_TYPE_START_IDENTIFIER)}.*?{re.escape(CUSTOM_TYPE_END_IDENTIFIER)})",
            re.DOTALL,
        )
        parts = marker_pattern.split(value)
        for index in range(0, len(parts), 2):
            parts[index] = parts[index].replace("&", "&amp;").replace("<", "&lt;")
        return "".join(parts)

    @classmethod
    def _elements_to_value(cls, elements: list[ET.Element], annotation: Any) -> Any:
        annotation = cls._unwrap_optional(annotation)
        origin = get_origin(annotation)

        if origin is list:
            item_annotation = get_args(annotation)[0] if get_args(annotation) else Any
            if len(elements) == 1 and not list(elements[0]):
                text = (elements[0].text or "").strip()
                if not text:
                    return []
                if text.startswith("["):
                    return text
            return [cls._element_to_value(element, item_annotation) for element in elements]
        if annotation is Any and len(elements) > 1:
            return [cls._element_to_value(element, Any) for element in elements]

        # Preserve the historical first-field-wins behavior for duplicate scalar fields.
        return cls._element_to_value(elements[0], annotation)

    @classmethod
    def _element_to_value(cls, element: ET.Element, annotation: Any) -> Any:
        annotation = cls._unwrap_optional(annotation)

        if annotation is str:
            if not list(element):
                return (element.text or "").strip()
            return cls._inner_xml(element).strip()

        field_annotations = cls._structured_field_annotations(annotation)
        if field_annotations is not None:
            if not list(element):
                return (element.text or "").strip()
            children_by_name: dict[str, list[ET.Element]] = defaultdict(list)
            for child in element:
                children_by_name[child.tag].append(child)
            return {
                name: cls._elements_to_value(children_by_name[name], field_annotation)
                for name, field_annotation in field_annotations.items()
                if name in children_by_name
            }

        origin = get_origin(annotation)
        if origin is list:
            return cls._elements_to_value([element], annotation)
        if origin is dict:
            args = get_args(annotation)
            value_annotation = args[1] if len(args) == 2 else Any
            children_by_name: dict[str, list[ET.Element]] = defaultdict(list)
            for child in element:
                children_by_name[child.tag].append(child)
            return {
                name: cls._elements_to_value(children, value_annotation) for name, children in children_by_name.items()
            }

        children_by_name: dict[str, list[ET.Element]] = defaultdict(list)
        for child in element:
            children_by_name[child.tag].append(child)
        if children_by_name:
            return {name: cls._elements_to_value(children, Any) for name, children in children_by_name.items()}
        return (element.text or "").strip()

    @staticmethod
    def _inner_xml(element: ET.Element) -> str:
        content = element.text or ""
        for child in element:
            content += ET.tostring(child, encoding="unicode")
        return content

    @staticmethod
    def _unwrap_optional(annotation: Any) -> Any:
        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(args) == 1:
                return args[0]
        return annotation

    @staticmethod
    def _structured_field_annotations(annotation: Any) -> dict[str, Any] | None:
        if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
            return {name: field.annotation for name, field in annotation.model_fields.items()}
        if is_typeddict(annotation):
            return get_type_hints(annotation)
        return None

    @classmethod
    def _uses_nested_xml(cls, annotation: Any) -> bool:
        """Keep DSPy's transport models on their established marker/JSON representation."""
        annotation = cls._unwrap_optional(annotation)
        origin = get_origin(annotation)
        if origin is list:
            args = get_args(annotation)
            if not args:
                return True
            item_annotation = cls._unwrap_optional(args[0])
            return not (
                isinstance(item_annotation, type)
                and issubclass(item_annotation, pydantic.BaseModel)
                and item_annotation.__module__.startswith("dspy.")
            )
        if origin is dict:
            return True
        if is_typeddict(annotation):
            return True
        return (
            isinstance(annotation, type)
            and issubclass(annotation, pydantic.BaseModel)
            and not annotation.__module__.startswith("dspy.")
        )
