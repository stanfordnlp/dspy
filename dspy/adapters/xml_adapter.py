import re
import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from typing import Any, Union, get_args, get_origin
from xml.sax.saxutils import quoteattr

import pydantic
import regex
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

# Avoid regex's compile/flag dispatch for every leaf and streaming chunk.
_compile_xml_pattern = lru_cache(maxsize=128)(regex.compile)
_xml_whitespace = re.compile(r"\s*")
_xml_newline = re.compile(r"[ \t]*\r?\n")


class XMLAdapter(ChatAdapter):
    """Schema-directed tags with literal leaves and standalone structural tag lines.

    Compact tags remain readable for compatibility. Neither form decodes entities.
    Declared objects reject extra XML fields; dictionaries permit arbitrary keys.
    Container closing boundaries are reserved, including inside literal strings.
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
        if not signature.output_fields:
            return ""

        def format_field(name, field):
            if self._uses_nested_xml(field.annotation):
                return self._xml_schema(name, field.annotation)
            annotations = (
                get_args(field.annotation)
                if get_origin(field.annotation) in (Union, types.UnionType)
                else (field.annotation,)
            )
            # Primitive types are already described above; JSON hints conflict with XML nulls.
            placeholder = (
                "{" + name + "}"
                if all(annotation in (str, int, float, bool, type(None)) for annotation in annotations)
                else translate_field_type(name, field)
            )
            return self.format_field_with_value({FieldInfoWithName(name=name, info=field): placeholder})

        return "Output format (replace placeholders):\n" + "\n\n".join(
            format_field(name, field) for name, field in signature.output_fields.items()
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
        if not signature.output_fields:
            return ""
        text = (
            "Use this format for text outputs. Put field tags on separate lines."
            " Write string values literally, without XML or JSON escaping."
        )
        nodes = []
        for field in signature.output_fields.values():
            schema = TypeAdapter(field.annotation).json_schema()
            if self._uses_nested_xml(field.annotation):
                nodes.append(schema)
            else:
                # DSPy types retain their own JSON conventions inside literal leaves.
                nodes.extend(s for s in [schema, *schema.get("anyOf", [])] if s.get("type") == "null")
        nullable = mapping = False
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                nullable |= node.get("type") == "null"
                mapping |= isinstance(node.get("additionalProperties"), dict) or node.get("additionalProperties") is True
                nodes.extend(node.values())
            elif isinstance(node, list):
                nodes.extend(node)
        if nullable:
            text += " Use <field/> for null and <field></field> for an empty string."
        if mapping:
            text += ' Dictionary keys: <entry key="...">; escape key attributes only.'
        return self.format_field_structure(signature) + "\n\n" + text

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        context = "Failed to parse XML"
        schemas = {
            name: TypeAdapter(field.annotation).json_schema(by_alias=False)
            for name, field in signature.output_fields.items()
        }
        try:
            root = ET.Element("dspy_root")
            prefix = _XMLPrefixParser(signature.output_fields)
            text = prefix.feed(completion.strip(), final=True)
            if text is None:
                raise ET.ParseError("Incomplete leading thought block")
            if prefix.fenced:
                fence = re.fullmatch(r"(.*)\r?\n```", text, re.DOTALL)
                if not fence:
                    raise ET.ParseError("Missing closing XML fence")
                text = fence[1].strip()
            # Like ChatAdapter, ignore introductory prose before the first field.
            text = re.sub(r"\A[^<]+(?=<)", "", text)
            parser = _XMLParser(root, {"properties": schemas}, {})
            parser.feed(text, final=True)
            elements = self._group_children(root)
            if parser.buffer or {child.tag for child in root} - signature.output_fields.keys():
                raise ET.ParseError("Unexpected output field or closing tag")
            for name, field in signature.output_fields.items():
                if name not in elements:
                    continue
                context = f"Failed to parse field {name}"
                field_schema = schemas[name]
                for candidate in [field_schema, *field_schema.get("anyOf", [])]:
                    try:
                        value = self._elements_to_value(elements[name], candidate, field_schema.get("$defs", {}))
                        if isinstance(value, (dict, list)):
                            fields[name] = TypeAdapter(field.annotation).validate_python(value, by_name=True, strict=False)
                        else:
                            fields[name] = parse_value(value, field.annotation)
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
    def _scalar_boundary(cls, schema: dict, definitions: dict, enclosing: str) -> str:
        """Recognize sibling headers or the enclosing close chain, without consuming them."""
        names = set()
        choices = cls._schemas(schema, definitions)
        siblings = []
        for choice in choices:
            if choice.get("type") == "array":
                names.add("item")
                siblings.extend(cls._schemas(choice.get("items", {}), definitions))
            else:
                siblings.append(choice)
        for sibling in siblings:
            names.update(sibling.get("properties", {}))
        if any(not (name[:1] + name[1:].replace("-", "_").replace(".", "_")).isidentifier() for name in names):
            names.add("entry")
        arbitrary = any(
            not s or (s.get("type") == "object" and s.get("additionalProperties", "properties" not in s) is not False)
            for s in siblings
        )
        tags = r"[\w.-]+" if arbitrary else "|".join(re.escape(name) for name in sorted(names))
        if not tags:
            return enclosing
        opening = rf"""<(?:{tags})(?:\s+key=(?:"[^"]*"|'[^']*'))?\s*/?>"""
        return rf"(?:{opening}|{enclosing})"

    @classmethod
    def _schemas(cls, schema: dict, definitions: dict, child: str | None = None) -> list[dict]:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if choices := schema.get("anyOf", schema.get("oneOf")):
            return [s for branch in choices for s in cls._schemas(branch, definitions, child)]
        if child is None:
            return [schema]
        if schema.get("type") == "array":
            item = schema.get("items", {})
            return [item] if child == "item" else cls._schemas(item, definitions, child)
        if schema.get("type") not in ("object", None):
            return []
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", "properties" not in schema)
        if child not in properties and additional is False:
            return []
        value = properties.get(child, additional)
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
            if choices := node.get("anyOf", node.get("oneOf")):
                return example(next(s for s in choices if s.get("type") != "null"), seen)
            if node.get("type") == "array":
                return [example(node.get("items", {}), seen)]
            if not node.get("properties") and isinstance(node.get("additionalProperties"), dict):
                return {"example key": example(node["additionalProperties"], seen)}
            return {name: example(child, seen) for name, child in node.get("properties", {}).items()} or "..."

        return cls._value_to_xml(example(schema), tag)

    @classmethod
    def _elements_to_value(cls, elements: list[ET.Element], schema: dict, definitions: dict) -> Any:
        schema = definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)
        if (
            len(elements) == 1
            and (schema.get("type") == "null" or {"type": "null"} in schema.get("anyOf", []))
            and not list(elements[0])
            and elements[0].text is None
        ):
            return None
        if choices := schema.get("anyOf", schema.get("oneOf")):
            discriminator = schema.get("discriminator", {})
            tags = cls._group_children(elements[0]).get(discriminator.get("propertyName"), [])
            if len(tags) == 1 and (ref := discriminator.get("mapping", {}).get((tags[0].text or "").strip())):
                return cls._elements_to_value(elements, {"$ref": ref}, definitions)
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


class _XMLParser:
    """Schema-directed traversal that can pause for a chunk without reparsing completed children."""

    def __init__(self, parent, schema, definitions, boundary=r"(?P<end>\Z)", single=False):
        self.parent = parent
        self.schema = schema
        self.definitions = definitions
        self.boundary = boundary
        self.single = single
        self.text = ""
        self.pos = 0
        self.final = False
        self.done = False
        self.reader = None

    @property
    def buffer(self):
        return self.text[self.pos:]

    def feed(self, text, final=False):
        self.text = self.buffer + text
        self.pos = 0
        self.final = final
        if self.reader is None:
            self.reader = self._children(self.parent, self.schema, self.definitions, self.boundary, self.single)
        try:
            next(self.reader)
        except StopIteration:
            self.done = True
        return ""

    def _match(self, pattern):
        while True:
            match = _compile_xml_pattern(pattern).match(self.text, self.pos, partial=not self.final)
            if not match:
                raise ET.ParseError("Expected a field tag")
            if not match.partial:
                return match
            yield

    def _children(self, parent, schema, definitions, boundary, single=False):
        enclosing = boundary if boundary.startswith("</") else None
        boundary = XMLAdapter._scalar_boundary(schema, definitions, boundary)
        while True:
            self.pos = _xml_whitespace.match(self.text, self.pos).end()
            if not self.final and (
                self.pos == len(self.text) or (self.pos == len(self.text) - 1 and self.text[self.pos] == "<")
            ):
                yield
                continue
            if self.pos == len(self.text) or self.text.startswith("</", self.pos):
                return
            opening = yield from self._match(r"""<([\w.-]+)(?:\s+key=(?:"[^"]*"|'[^']*'))?\s*(/?)>""")
            name = opening[1]
            element = ET.fromstring(opening[0] if opening[2] else opening[0] + f"</{name}>")
            key = element.get("key", name) if name == "entry" else name
            children = XMLAdapter._schemas(schema, definitions, key)
            if not children:
                raise ET.ParseError(f"Unexpected field {key}")
            parent.append(element)
            self.pos = opening.end()
            if not opening[2]:
                child_schema = children[0] if len(children) == 1 else {"anyOf": children}
                child_definitions = {**definitions, **child_schema.get("$defs", {})}
                choices = XMLAdapter._schemas(child_schema, child_definitions)
                structured = any(s.get("type") in ("object", "array") or not s for s in choices)
                nested = False
                while structured:
                    newline = _xml_newline.match(self.text, self.pos)
                    nested_start = _xml_whitespace.match(self.text, self.pos).end()
                    ambiguous = newline and any(s.get("type") == "string" for s in choices)
                    tag_line = _compile_xml_pattern(r"<[^>\n]+>[ \t]*(?:\r?\n|$)").match(
                        self.text, nested_start, partial=not self.final
                    ) if ambiguous else None
                    if not self.final and (
                        nested_start == len(self.text)
                        or (tag_line and (tag_line.partial or tag_line.end() == len(self.text)))
                    ):
                        yield
                        continue
                    nested = bool(tag_line) if ambiguous else self.text.startswith("<", nested_start)
                    break
                element.text = ""
                if nested:
                    self.pos = nested_start
                    yield from self._children(
                        element, child_schema, child_definitions, rf"</{re.escape(name)}\s*>\s*{boundary}"
                    )
                    closing = yield from self._match(rf"</{re.escape(name)}\s*>")
                    self.pos = closing.end()
                else:
                    parser = _XMLScalarParser(name, boundary, enclosing)
                    values = []
                    while True:
                        value, self.pos = parser.read(self.text, self.pos, final=self.final)
                        values.append(value)
                        if parser.done:
                            element.text = "".join(values)
                            break
                        if self.final:
                            raise ET.ParseError(f"Missing closing tag for {name}")
                        yield
            if single:
                return


class _XMLPrefixParser:
    """Discard leading thought blocks and one XML fence, without retaining thought bodies."""

    def __init__(self, output_fields):
        self.output_fields = output_fields
        self.buffer = ""
        self.thought = None
        self.fenced = False
        self.done = False

    def feed(self, text: str, final: bool = False) -> str | None:
        if self.done:
            return text
        self.buffer += text
        while True:
            if self.thought:
                closing = f"</{self.thought}>"
                end = self.buffer.find(closing)
                if end < 0:
                    # Only a split closing tag can matter in the next chunk.
                    self.buffer = self.buffer[-(len(closing) - 1):]
                    return None
                self.buffer = self.buffer[end + len(closing):]
                self.thought = None
            self.buffer = self.buffer.lstrip()
            opening = _compile_xml_pattern(r"<(think|thinking)>|```(?:xml)?[ \t]*\r?\n").match(
                self.buffer, partial=not final
            )
            if opening:
                if opening.partial:
                    return None
                if opening[1] and opening[1] not in self.output_fields:
                    self.thought = opening[1]
                    self.buffer = self.buffer[opening.end():]
                    continue
                if not opening[1] and not self.fenced:
                    self.fenced = True
                    self.buffer = self.buffer[opening.end():]
                    continue
            self.done = True
            text, self.buffer = self.buffer, ""
            return text


class _XMLScalarParser:
    """Read a literal leaf after its opening tag, in full or incrementally."""

    def __init__(self, name: str, boundary: str, enclosing: str | None = None):
        self.name = re.escape(name)
        self.boundary = boundary
        self.enclosing = enclosing
        self.buffer = ""
        self.multiline = None
        self.at_line_start = True
        self.value_started = False
        self.done = False

    def feed(self, text: str, final: bool = False) -> str:
        self.buffer += text
        value, pos = self.read(self.buffer, 0, final=final)
        self.buffer = self.buffer[pos:]
        return value

    def read(self, text: str, pos: int, final: bool) -> tuple[str, int]:
        """Scan a shared response without copying its unconsumed suffix."""
        if self.multiline is None:
            newline = _xml_newline.match(text, pos)
            if not newline and not final and re.compile(r"[ \t]*\r?").fullmatch(text, pos):
                return "", pos
            self.multiline = newline is not None
            if newline:
                pos = newline.end()

        if self.multiline:
            # A tag at a chunk boundary may still be followed by literal text on the same line.
            ending = r"(?:\n|$)" if final else r"\n"
            # Accept an end-of-line closing tag attached to text (e.g. a Markdown code fence).
            pattern = rf"</{self.name}[ \t]*>[ \t]*\r?{ending}"
        else:
            pattern = rf"</{self.name}\s*>"
        if self.enclosing:
            # A missing leaf close must not consume the next object or list item.
            pattern = rf"{pattern}|(?P<scope>{self.enclosing})"
        closing_pattern = _compile_xml_pattern(pattern)
        continuation_pattern = _compile_xml_pattern(rf"\s*{self.boundary}")
        end = len(text)
        for candidate in closing_pattern.finditer(text, pos, partial=not final):
            if candidate.partial:
                end = candidate.start()
                break
            if self.enclosing and candidate.group("scope") is not None:
                if not final and candidate.group("end") is not None:
                    end = candidate.start()
                    break
                raise ET.ParseError(f"Missing closing tag for {self.name} before container end")
            # Match in place rather than copying the response suffix for each candidate.
            continuation = continuation_pattern.match(text, candidate.end(), partial=not final)
            if continuation is None:
                if self.enclosing and _compile_xml_pattern(self.enclosing).match(text, candidate.start()):
                    raise ET.ParseError(f"Missing closing tag for {self.name} before container end")
                # An inner closing tag followed by literal text is part of the value.
                continue
            end = candidate.start()
            # Chunk boundaries are not EOF; hold the candidate until its continuation is known.
            self.done = not continuation.partial and (final or continuation.group("end") is None)
            break

        if self.multiline:
            line_start = text.rfind("\n", pos, end) + 1
            if line_start > pos or self.at_line_start:
                line_start = max(pos, line_start)
                if not text[line_start:end].strip(" \t"):
                    end = line_start
        value = text[pos:end]
        if self.done:
            consumed = candidate.end()
            value = value.removesuffix("\n").removesuffix("\r") if self.multiline else value.rstrip()
        else:
            # Retain the possible framing newline (or compact trailing whitespace) until the delimiter is known.
            if self.multiline:
                value = value.removesuffix("\n").removesuffix("\r")
            else:
                value = value.rstrip()
            consumed = pos + len(value)
        if value:
            self.at_line_start = value.endswith("\n")
        if not self.multiline and not self.value_started:
            value = value.lstrip()
        self.value_started |= bool(value)
        return value, consumed
