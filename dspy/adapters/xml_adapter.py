import re
import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Union, get_args, get_origin
from xml.sax.saxutils import quoteattr

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
    """Schema-directed tags with literal leaves and standalone structural tag lines.

    Compact tags remain readable for compatibility. Neither form decodes entities.
    """

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, value in fields_with_values.items():
            serialized = serialize_for_json(value)
            is_output = (field.info.json_schema_extra or {}).get("__dspy_field_type") == "output"
            nested = isinstance(serialized, (dict, list)) and self._uses_nested_xml(field.info.annotation)
            if is_output and (serialized is None or nested):
                output.append(self._value_to_xml(serialized, field.name))
                continue
            formatted = format_field_value(field_info=field.info, value=value)
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def format_field_structure(self, signature: type[Signature]) -> str:
        parts = ["All interactions will be structured in the following way, with the appropriate values filled in."]
        for group in (signature.input_fields, signature.output_fields):
            fields = []
            for name, field in group.items():
                if name in signature.output_fields and self._uses_nested_xml(field.annotation):
                    fields.append(self._xml_schema(name, field.annotation))
                else:
                    value = {FieldInfoWithName(name=name, info=field): translate_field_type(name, field)}
                    fields.append(self.format_field_with_value(value))
            parts.append("\n\n".join(fields))
        return "\n\n".join(parts)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        fields = {
            FieldInfoWithName(name=k, info=v): inputs[k] for k, v in signature.input_fields.items() if k in inputs
        }
        messages = [prefix, self.format_field_with_value(fields)]
        if main_request:
            messages.append(self.user_message_output_requirements(signature))
        return "\n\n".join((*messages, suffix)).strip()

    def format_assistant_message_content(
        self, signature: type[Signature], outputs: dict[str, Any], missing_field_message=None
    ) -> str:
        fields = {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        }
        return self.format_field_with_value(fields)

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        fields = ", then ".join(f"`<{name}>`" for name in signature.output_fields)
        schemas = [
            self._xml_schema(name, field.annotation)
            for name, field in signature.output_fields.items()
            if self._uses_nested_xml(field.annotation)
        ]
        return (
            f"Respond with the corresponding output fields wrapped in XML tags {fields}."
            " Put structural tags on separate lines; leaf values are literal text, without escaping."
        ) + (f" Use this nested XML structure: {' '.join(schemas)}" if schemas else "")

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        context = "Failed to parse XML"
        schemas = {
            name: TypeAdapter(field.annotation).json_schema(by_alias=False)
            for name, field in signature.output_fields.items()
        }
        try:
            root = ET.Element("dspy_root")
            remaining = self._parse_children(completion.strip(), root, {"properties": schemas}, {})
            elements = self._group_children(root)
            if remaining or {child.tag for child in root} - signature.output_fields.keys():
                raise ET.ParseError("Unexpected output field or closing tag")
            for name, field in signature.output_fields.items():
                if name not in elements:
                    continue
                context = f"Failed to parse field {name}"
                field_schema = schemas[name]
                for candidate in [field_schema, *field_schema.get("anyOf", [])]:
                    try:
                        value = self._elements_to_value(elements[name], candidate, field_schema.get("$defs", {}))
                        try:
                            fields[name] = parse_value(value, field.annotation)
                        except pydantic.ValidationError:
                            fields[name] = TypeAdapter(field.annotation).validate_python(value, by_name=True)
                        break
                    except ValueError as e:
                        error = e
                else:
                    raise error
        except (ET.ParseError, ValueError) as e:
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"{context}: {e}",
            ) from e
        fields = apply_output_field_defaults(signature, fields)
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="XMLAdapter", signature=signature, lm_response=completion, parsed_result=fields
            )
        return fields

    @classmethod
    def _parse_children(cls, text: str, parent: ET.Element, schema: dict, definitions: dict) -> str:
        while text and not text.startswith("</"):
            opening = re.match(r"""<([\w.-]+)(?:\s+key=(?:"[^"]*"|'[^']*'))?\s*(/?)>""", text)
            if not opening:
                raise ET.ParseError("Expected a field opening tag")
            name = opening[1]
            # Only tag headers use XML parsing; scalar bodies remain literal.
            element = ET.fromstring(opening[0] if opening[2] else opening[0] + f"</{name}>")
            parent.append(element)
            text = text[opening.end() :]
            if not opening[2]:
                newline = re.match(r"[ \t]*\r?\n", text)
                text = text[newline.end() :] if newline else text
                key = element.get("key", name) if name == "entry" else name
                children = cls._schemas(schema, definitions, key)
                child_schema = children[0] if len(children) == 1 else {"anyOf": children}
                child_definitions = {**definitions, **child_schema.get("$defs", {})}
                choices = cls._schemas(child_schema, child_definitions)
                structured = any(s.get("type") in ("object", "array") or not s for s in choices)
                tag_line = re.match(r"<[^>\n]+>[ \t]*(?:\r?\n|$)", text.lstrip())
                element.text = ""
                nested = structured and (tag_line if newline else text.lstrip().startswith("<"))
                if nested:
                    text = cls._parse_children(text.lstrip(), element, child_schema, child_definitions)
                pattern = rf"</{re.escape(name)}\s*>"
                if newline and not nested:
                    pattern = rf"(?m)^[ \t]*</{re.escape(name)}[ \t]*>[ \t]*\r?$"
                closing = re.match(pattern, text) if nested else re.search(pattern, text)
                if closing is None:
                    raise ET.ParseError(f"Missing closing tag for {name}")
                if not nested:
                    body = text[: closing.start()]
                    element.text = body.removesuffix("\n").removesuffix("\r") if newline else body.strip()
                text = text[closing.end() :]
            text = text.lstrip()
        return text

    @classmethod
    def _schemas(cls, schema: dict, definitions: dict, child: str | None = None) -> list[dict]:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if "anyOf" in schema:
            return [s for branch in schema["anyOf"] for s in cls._schemas(branch, definitions, child)]
        if child is None:
            return [schema]
        if schema.get("type") == "array":
            item = schema.get("items", {})
            return [item] if child == "item" else cls._schemas(item, definitions, child)
        if schema.get("type") not in ("object", None):
            return []
        value = schema.get("properties", {}).get(child, schema.get("additionalProperties", {}))
        return [value if isinstance(value, dict) else {}]

    @classmethod
    def _value_to_xml(cls, value: Any, tag: str, key: str | None = None) -> str:
        attrs = f" key={quoteattr(key)}" if key is not None else ""
        if isinstance(value, list):
            body = "\n".join(cls._value_to_xml(item, "item") for item in value)
        elif isinstance(value, dict):
            children = []
            for name, child in value.items():
                name = str(name)
                valid_name = (name[:1] + name[1:].replace("-", "_").replace(".", "_")).isidentifier()
                children.append(
                    cls._value_to_xml(child, name) if valid_name else cls._value_to_xml(child, "entry", name)
                )
            body = "\n".join(children)
        elif value is None:
            return f"<{tag}{attrs} />"
        else:
            body = str(value)
        return f"<{tag}{attrs}>\n{body}\n</{tag}>"

    @classmethod
    def _xml_schema(cls, tag: str, annotation: Any) -> str:
        schema = TypeAdapter(annotation).json_schema(by_alias=False)

        def example(node, seen=frozenset()):
            if ref := node.get("$ref"):
                name = ref.rsplit("/", 1)[-1]
                return "..." if name in seen else example(schema["$defs"][name], seen | {name})
            if choices := node.get("anyOf"):
                return example(next(s for s in choices if s.get("type") != "null"), seen)
            if node.get("type") == "array":
                return [example(node.get("items", {}), seen)]
            return {name: example(child, seen) for name, child in node.get("properties", {}).items()} or "..."

        return cls._value_to_xml(example(schema), tag)

    @classmethod
    def _elements_to_value(cls, elements: list[ET.Element], schema: dict, definitions: dict) -> Any:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if choices := schema.get("anyOf"):
            if (
                len(elements) == 1
                and {"type": "null"} in choices
                and not list(elements[0])
                and elements[0].text is None
            ):
                return None
            choices = [choice for choice in choices if choice.get("type") != "null"]
            schema = choices[0]
            if list(elements[0]):
                schema = next((choice for choice in choices if choice.get("type") != "string"), schema)
        if len(elements) > 1 and schema.get("type") not in ("array", None):
            raise ValueError("Repeated elements require a list field")
        if schema.get("type") == "array":
            if len(elements) == 1 and not list(elements[0]):
                text = (elements[0].text or "").strip()
                if not text or text.startswith("["):
                    return [] if not text else text
            if len(elements) == 1 and (items := cls._group_children(elements[0]).get("item")):
                elements = items
            return [cls._elements_to_value([element], schema.get("items", {}), definitions) for element in elements]
        element = elements[0]
        if schema.get("type") not in ("object", None) and list(element):
            raise ValueError("Scalar values cannot contain child fields")
        children = cls._group_children(element)
        if not children:
            if schema.get("type") == "object" and not (element.text or "").strip():
                return {}
            values = [element.text or "" for element in elements]
            return values[0] if len(values) == 1 else values
        properties = schema.get("properties", {})
        child_schema = schema.get("additionalProperties", {})
        if not isinstance(child_schema, dict):
            child_schema = {}
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
        is_dspy_model = (
            isinstance(item, type) and issubclass(item, pydantic.BaseModel) and item.__module__.startswith("dspy.")
        )
        return not is_dspy_model and (
            origin in (list, dict)
            or is_typeddict(annotation)
            or (isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel))
        )
