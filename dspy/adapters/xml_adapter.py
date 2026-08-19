import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Union, get_args, get_origin

import pydantic
from pydantic import TypeAdapter
from typing_extensions import is_typeddict

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
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
    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, value in fields_with_values.items():
            serialized = serialize_for_json(value)
            is_output = (field.info.json_schema_extra or {}).get("__dspy_field_type") == "output"
            if is_output and self._uses_nested_xml(field.info.annotation) and isinstance(serialized, (dict, list)):
                output.append(self._value_to_xml(serialized, field.name))
                continue
            formatted = format_field_value(field_info=field.info, value=value)
            if is_output and field.info.annotation is str:
                formatted = formatted.replace("&", "&amp;").replace("<", "&lt;")
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def format_field_structure(self, signature: type[Signature]) -> str:
        def format_field(name, field):
            if (field.json_schema_extra or {}).get("__dspy_field_type") == "output" and self._uses_nested_xml(
                field.annotation
            ):
                return self._xml_schema(name, field.annotation)
            return self.format_field_with_value(
                {FieldInfoWithName(name=name, info=field): translate_field_type(name, field)}
            )

        fields = ("\n\n".join(format_field(name, field) for name, field in group.items()) for group in (signature.input_fields, signature.output_fields))
        return "\n\n".join(("All interactions will be structured in the following way, with the appropriate values filled in.", *fields))

    def format_user_message_content(self, signature: type[Signature], inputs: dict[str, Any], prefix: str = "", suffix: str = "", main_request: bool = False) -> str:
        fields = {FieldInfoWithName(name=k, info=v): inputs[k] for k, v in signature.input_fields.items() if k in inputs}
        messages = [prefix, self.format_field_with_value(fields)]
        if main_request:
            messages.append(self.user_message_output_requirements(signature))
        return "\n\n".join((*messages, suffix)).strip()

    def format_assistant_message_content(self, signature: type[Signature], outputs: dict[str, Any], missing_field_message=None) -> str:
        fields = {FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message) for k, v in signature.output_fields.items()}
        return self.format_field_with_value(fields)

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        fields = ", then ".join(f"`<{name}>`" for name in signature.output_fields)
        schemas = [self._xml_schema(name, field.annotation) for name, field in signature.output_fields.items() if self._uses_nested_xml(field.annotation)]
        return f"Respond with the corresponding output fields wrapped in XML tags {fields}." + (f" Use this nested XML structure: {' '.join(schemas)}" if schemas else "")

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        try:
            root = ET.fromstring(f"<dspy_root>{completion}</dspy_root>")
        except ET.ParseError as e:
            raise AdapterParseError(adapter_name="XMLAdapter", signature=signature, lm_response=completion, message=f"Failed to parse XML: {e}") from e
        elements = self._group_children(root)
        fields = {}
        for name in filter(elements.__contains__, signature.output_fields):
            field = signature.output_fields[name]
            schema, value = TypeAdapter(field.annotation).json_schema(), None
            for candidate in [schema, *schema.get("anyOf", [])]:
                try:
                    value = self._elements_to_value(elements[name], candidate, schema.get("$defs", {}))
                    fields[name] = parse_value(value, field.annotation)
                    break
                except Exception as e:
                    error = e
            else:
                raise AdapterParseError(
                    adapter_name="XMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {field} with value {value}: {error}",
                ) from error
        fields = apply_output_field_defaults(signature, fields)
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(adapter_name="XMLAdapter", signature=signature, lm_response=completion, parsed_result=fields)
        return fields

    @classmethod
    def _value_to_xml(cls, value: Any, tag: str, key: str | None = None) -> str:
        attrs = f' key="{key.replace("&", "&amp;").replace(chr(34), "&quot;").replace("<", "&lt;").replace(chr(9), "&#9;").replace(chr(10), "&#10;").replace(chr(13), "&#13;")}"' if key is not None else ""
        if isinstance(value, list):
            return "".join(cls._value_to_xml(item, tag, key) for item in value) if value else f"<{tag}{attrs} />"
        if isinstance(value, dict):
            return f"<{tag}{attrs}>{''.join(cls._value_to_xml(child, str(name)) if (str(name)[:1] + str(name)[1:].replace('-', '_').replace('.', '_')).isidentifier() else cls._value_to_xml(child, 'entry', str(name)) for name, child in value.items())}</{tag}>"
        return f"<{tag}{attrs}>{str(value).replace('&', '&amp;').replace('<', '&lt;')}</{tag}>" if value is not None else f"<{tag}{attrs} />"

    @classmethod
    def _xml_schema(cls, tag: str, annotation: Any) -> str:
        schema = TypeAdapter(annotation).json_schema()
        return cls._schema_to_xml(tag, schema, schema.get("$defs", {}), frozenset())

    @classmethod
    def _schema_to_xml(cls, tag: str, schema: dict, definitions: dict, seen: frozenset[str]) -> str:
        if ref := schema.get("$ref"):
            name = ref.rsplit("/", 1)[-1]
            return f"<{tag}>...</{tag}>" if name in seen else cls._schema_to_xml(tag, definitions[name], definitions, seen | {name})
        if choices := schema.get("anyOf"):
            return cls._schema_to_xml(
                tag, next((s for s in choices if s.get("type") != "null"), choices[0]), definitions, seen
            )
        if schema.get("type") == "array":
            item = cls._schema_to_xml(tag, schema.get("items", {}), definitions, seen)
            return f"{item} {item}"
        children = "".join(
            cls._schema_to_xml(name, child, definitions, seen) for name, child in schema.get("properties", {}).items()
        )
        return f"<{tag}>{children}</{tag}>" if children else f"<{tag}>...</{tag}>"

    @classmethod
    def _elements_to_value(cls, elements: list[ET.Element], schema: dict, definitions: dict) -> Any:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if choices := schema.get("anyOf"):
            if all(({"type": "null"} in choices, not list(elements[0]), not (elements[0].text or "").strip())):
                return None
            schema = next(choice for choice in choices if choice.get("type") != "null")
        if schema.get("type") == "array":
            if len(elements) == 1 and not list(elements[0]):
                text = (elements[0].text or "").strip()
                if any((not text, text.startswith("["))):
                    return [] if not text else text
            return [cls._elements_to_value([element], schema.get("items", {}), definitions) for element in elements]
        element = elements[0]
        if all((schema.get("type") == "string", list(element))):
            return (element.text or "") + "".join(ET.tostring(child, encoding="unicode") for child in element)
        children = cls._group_children(element)
        if not children:
            return values[0] if len(values := [(element.text or "").strip() for element in elements]) == 1 else values
        properties, child_schema = schema.get("properties", {}), (additional if isinstance(additional := schema.get("additionalProperties", {}), dict) else {})
        return {
            name: cls._elements_to_value(items, properties.get(name, child_schema), definitions)
            for name, items in children.items()
        }

    @staticmethod
    def _group_children(element: ET.Element) -> dict[str, list[ET.Element]]:
        children = defaultdict(list)
        for child in element:
            children[child.attrib.get("key", child.tag) if child.tag == "entry" else child.tag].append(child)
        return children

    @staticmethod
    def _uses_nested_xml(annotation: Any) -> bool:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        annotation = args[0] if len(args) == 1 and get_origin(annotation) in (Union, types.UnionType) else annotation
        origin = get_origin(annotation)
        item = get_args(annotation)[0] if origin is list and get_args(annotation) else annotation
        is_dspy_model = isinstance(item, type) and issubclass(item, pydantic.BaseModel) and item.__module__.startswith("dspy.")
        return not is_dspy_model and (
            origin in (list, dict)
            or is_typeddict(annotation)
            or (isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel))
        )
