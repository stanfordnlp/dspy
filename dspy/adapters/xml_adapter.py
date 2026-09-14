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
            if is_output and serialized is None:
                output.append(f"<{field.name} />")
                continue
            if is_output and self._uses_nested_xml(field.info.annotation) and isinstance(serialized, (dict, list)):
                output.append(self._value_to_xml(serialized, field.name))
                continue
            formatted = format_field_value(field_info=field.info, value=value)
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

        fields = (
            "\n\n".join(format_field(name, field) for name, field in group.items())
            for group in (signature.input_fields, signature.output_fields)
        )
        return "\n\n".join(
            (
                "All interactions will be structured in the following way, with the appropriate values filled in.",
                *fields,
            )
        )

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
        try:
            root = self._parse_fields(signature, completion)
        except ET.ParseError as e:
            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse XML: {e}",
            ) from e
        elements = self._group_children(root)
        fields = {}
        for name, field in signature.output_fields.items():
            if name not in elements:
                continue
            adapter = TypeAdapter(field.annotation)
            schema = adapter.json_schema(by_alias=False)
            value = None
            for candidate in [schema, *schema.get("anyOf", [])]:
                try:
                    value = self._elements_to_value(elements[name], candidate, schema.get("$defs", {}))
                    try:
                        fields[name] = parse_value(value, field.annotation)
                    except pydantic.ValidationError:
                        fields[name] = adapter.validate_python(value, by_name=True)
                    break
                except ValueError as e:
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
            raise AdapterParseError(
                adapter_name="XMLAdapter", signature=signature, lm_response=completion, parsed_result=fields
            )
        return fields

    @classmethod
    def _parse_fields(cls, signature: type[Signature], completion: str) -> ET.Element:
        root = ET.Element("dspy_root")
        schema = {
            "type": "object",
            "properties": {
                name: TypeAdapter(field.annotation).json_schema(by_alias=False)
                for name, field in signature.output_fields.items()
            },
        }
        remaining = completion.strip()
        while remaining:
            element, remaining = cls._parse_element(remaining, schema, {})
            if element.tag not in signature.output_fields:
                raise ET.ParseError(f"Unexpected output field {element.tag}")
            root.append(element)
        return root

    @classmethod
    def _parse_element(cls, text: str, parent_schema: dict, definitions: dict) -> tuple[ET.Element, str]:
        opening = re.match(r"""<([\w.-]+)(?:\s+key=(?:"[^"]*"|'[^']*'))?\s*(/?)>""", text)
        if not opening:
            raise ET.ParseError("Expected a field opening tag")
        name = opening[1]
        # XML is used only for the opening tag (not its body), including encoded mapping keys.
        element = ET.fromstring(opening[0] if opening[2] else opening[0] + f"</{name}>")
        remaining = text[opening.end() :]
        if opening[2]:
            return element, remaining.lstrip()
        newline = re.match(r"[ \t]*\r?\n", remaining)
        remaining = remaining[newline.end() :] if newline else remaining
        element.text = ""
        key = element.attrib.get("key", name) if name == "entry" else name
        schema = cls._child_schema(parent_schema, key, definitions)
        definitions = {**definitions, **schema.get("$defs", {})}
        choices = cls._schema_choices(schema, definitions)
        structured = any(s.get("type") in ("object", "array") or not s for s in choices)
        closing_pattern = rf"</{re.escape(name)}\s*>"
        child_tags = (
            re.match(r"<[^>\n]+>[ \t]*(?:\r?\n|$)", remaining.lstrip())
            if newline
            else remaining.lstrip().startswith("<")
        )
        if structured and child_tags:
            remaining = remaining.lstrip()
            while not (closing := re.match(closing_pattern, remaining)):
                child, remaining = cls._parse_element(remaining, schema, definitions)
                element.append(child)
            return element, remaining[closing.end() :].lstrip()
        # A multiline leaf ends only at its standalone closing tag, not tags inside code.
        if newline:
            closing_pattern = rf"(?m)^[ \t]*</{re.escape(name)}[ \t]*>[ \t]*\r?$"
        closing = re.search(closing_pattern, remaining)
        if closing is None:
            raise ET.ParseError(f"Missing closing tag for {name}")
        body = remaining[: closing.start()]
        element.text = body.removesuffix("\n").removesuffix("\r") if newline else body.strip()
        return element, remaining[closing.end() :].lstrip()

    @classmethod
    def _schema_choices(cls, schema: dict, definitions: dict) -> list[dict]:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if "anyOf" in schema:
            return [choice for branch in schema["anyOf"] for choice in cls._schema_choices(branch, definitions)]
        return [schema]

    @classmethod
    def _child_schema(cls, schema: dict, name: str, definitions: dict) -> dict:
        children = []
        for choice in cls._schema_choices(schema, definitions):
            if choice.get("type") == "array":
                item = choice.get("items", {})
                children.append(item if name == "item" else cls._child_schema(item, name, definitions))
            elif choice.get("type") == "object":
                child = choice.get("properties", {}).get(name, choice.get("additionalProperties", {}))
                children.append(child if isinstance(child, dict) else {})
        return {"anyOf": children} if len(children) > 1 else children[0] if children else {}

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
        return cls._schema_to_xml(tag, schema, schema.get("$defs", {}), frozenset())

    @classmethod
    def _schema_to_xml(cls, tag: str, schema: dict, definitions: dict, seen: frozenset[str]) -> str:
        if ref := schema.get("$ref"):
            name = ref.rsplit("/", 1)[-1]
            return (
                f"<{tag}>\n...\n</{tag}>"
                if name in seen
                else cls._schema_to_xml(tag, definitions[name], definitions, seen | {name})
            )
        if choices := schema.get("anyOf"):
            return cls._schema_to_xml(
                tag, next((s for s in choices if s.get("type") != "null"), choices[0]), definitions, seen
            )
        if schema.get("type") == "array":
            item = cls._schema_to_xml("item", schema.get("items", {}), definitions, seen)
            return f"<{tag}>\n{item}\n</{tag}>"
        children = "\n".join(
            cls._schema_to_xml(name, child, definitions, seen) for name, child in schema.get("properties", {}).items()
        )
        return f"<{tag}>\n{children or '...'}\n</{tag}>"

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
