import ast
import enum
import inspect
import json
import types
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.types.base_type import Type as DspyType
from dspy.signatures.utils import get_dspy_field_type


def serialize_for_json(value: Any) -> Any:
    """
    Formats the specified value so that it can be serialized as a JSON string.

    Args:
        value: The value to format as a JSON string.
    Returns:
        The formatted value, which is serializable as a JSON string.
    """
    # Attempt to format the value as a JSON-compatible object using pydantic, falling back to
    # a string representation of the value if that fails (e.g. if the value contains an object
    # that pydantic doesn't recognize or can't serialize)
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


def format_field_value(field_info: FieldInfo, value: Any, assume_text=True) -> str | dict:
    """
    Formats the value of the specified field according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself.

    Args:
      field_info: Information about the field, including its DSPy field type and annotation.
      value: The value of the field.
    Returns:
      The formatted value of the field, represented as a string.
    """
    string_value = None
    if isinstance(value, list) and field_info.annotation is str:
        # If the field has no special type requirements, format it as a nice numbered list for the LM.
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = serialize_for_json(value)
        if isinstance(jsonable_value, dict) or isinstance(jsonable_value, list):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            # If the value is not a Python representation of a JSON object or Array
            # (e.g. the value is a JSON string), just use the string representation of the value
            # to avoid double-quoting the JSON string (which would hurt accuracy for certain
            # tasks, e.g. tasks that rely on computing string length)
            string_value = str(jsonable_value)

    if assume_text:
        return string_value
    else:
        return {"type": "text", "text": string_value}


def _get_json_schema(field_type):
    def move_type_to_front(d):
        # Move the 'type' key to the front of the dictionary, recursively, for LLM readability/adherence.
        if isinstance(d, Mapping):
            return {
                k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != "type", item[0]))
            }
        elif isinstance(d, list):
            return [move_type_to_front(item) for item in d]
        return d

    schema = pydantic.TypeAdapter(field_type).json_schema()
    schema = move_type_to_front(schema)
    return schema


def translate_field_type(field_name, field_info):
    field_type = field_info.annotation

    if get_dspy_field_type(field_info) == "input" or field_type is str:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
        enum_vals = "; ".join(str(member.value) for member in field_type)
        desc = f"must be one of: {enum_vals}"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        desc = (
            # Strongly encourage the LM to avoid choosing values that don't appear in the
            # literal or returning a value of the form 'Literal[<selected_value>]'
            f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
        )
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_get_json_schema(field_type), ensure_ascii=False)}"

    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


def find_enum_member(enum, identifier):
    """
    Finds the enum member corresponding to the specified identifier, which may be the
    enum member's name or value.

    Args:
        enum: The enum to search for the member.
        identifier: If the enum is explicitly-valued, this is the value of the enum member to find.
                    If the enum is auto-valued, this is the name of the enum member to find.
    Returns:
        The enum member corresponding to the specified identifier.
    """
    # Check if the identifier is a valid enum member value *before* checking if it's a valid enum
    # member name, since the identifier will be a value for explicitly-valued enums. This handles
    # the (rare) case where an enum member value is the same as another enum member's name in
    # an explicitly-valued enum
    for member in enum:
        if member.value == identifier:
            return member

    # If the identifier is not a valid enum member value, check if it's a valid enum member name,
    # since the identifier will be a member name for auto-valued enums
    if identifier in enum.__members__:
        return enum[identifier]

    raise ValueError(f"{identifier} is not a valid name or value for the enum {enum.__name__}")


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value

        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)

    if origin in (Union, types.UnionType) and type(None) in get_args(annotation) and str in get_args(annotation):
        # Handle union annotations, e.g., `str | None`, `Optional[str]`, `Union[str, int, None]`, etc.
        return TypeAdapter(annotation).validate_python(value)

    candidate = json_repair.loads(value)  # json_repair.loads returns "" on failure.
    if candidate == "" and value != "":
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            candidate = value

    try:
        return TypeAdapter(annotation).validate_python(candidate)
    except pydantic.ValidationError as e:
        if inspect.isclass(annotation) and issubclass(annotation, DspyType):
            try:
                # For dspy.Type, try parsing from the original value in case it has a custom parser
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e
        raise


def get_annotation_name(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        else:
            return str(annotation)

    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(a) if isinstance(a, str) else get_annotation_name(a)
            for a in args
        )
        return f"{get_annotation_name(origin)}[{args_str}]"
    else:
        args_str = ", ".join(get_annotation_name(a) for a in args)
        return f"{get_annotation_name(origin)}[{args_str}]"


def get_field_description_string(fields: dict) -> str:
    field_descriptions = []
    for idx, (k, v) in enumerate(fields.items()):
        field_message = f"{idx + 1}. `{k}`"
        field_message += f" ({get_annotation_name(v.annotation)})"
        desc = v.json_schema_extra["desc"] if v.json_schema_extra["desc"] != f"${{{k}}}" else ""

        custom_types = DspyType.extract_custom_type_from_annotation(v.annotation)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"

        field_message += f": {desc}"
        field_message += (
            f"\nConstraints: {v.json_schema_extra['constraints']}" if v.json_schema_extra.get("constraints") else ""
        )
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()


def _format_input_list_field_value(value: list[Any]) -> str:
    """
    Formats the value of an input field of type list[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(value[0])

    return "\n".join([f"[{idx + 1}] {_format_blob(txt)}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    """
    Formats the specified text blobs so that an LM can parse it correctly within a list
    of multiple text blobs.

    Args:
        blob: The text blob to format.
    Returns:
        The formatted text blob.
    """
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"

    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def _quoted_string_for_literal_type_annotation(s: str) -> str:
    """
    Return the specified string quoted for inclusion in a literal type annotation.
    """
    has_single = "'" in s
    has_double = '"' in s

    if has_single and not has_double:
        # Only single quotes => enclose in double quotes
        return f'"{s}"'
    elif has_double and not has_single:
        # Only double quotes => enclose in single quotes
        return f"'{s}'"
    elif has_single and has_double:
        # Both => enclose in single quotes; escape each single quote with \'
        escaped = s.replace("'", "\\'")
        return f"'{escaped}'"
    else:
        # Neither => enclose in single quotes
        return f"'{s}'"


class LargePayloadHashManager:
    """
    Used by `format` in adapters.base.py
    That function formats the input prompt as one string, which means it stringifies
    large data (dspy types like Image, Audio). Then it splits that prompt string into
    a list, where special types get their own item. But stringifying large data is slow
    and memory-intensive. Instead, we replace the large data with a hash token before
    building the prompt, and restore the original data at the end.

    This class facilitates that with these two methods:
    - replace_large_data: replace large data with hash tokens for inputs, demos, history
    - restore_large_data: restore the original data from hash token for LLM messages

    Notes:
    - Non‑destructive: never mutates inputs; returns new structures mirroring shape.
    - Threshold-based: strings longer than `LARGE_DATA_THRESHOLD` are hashed
    - Only Image type is implemented - TODO for audio, etc.
    """

    # Configurable threshold for what constitutes "large" data
    LARGE_DATA_THRESHOLD = 1000  # characters

    def __init__(self):
        """Initialize the manager with fresh hash mappings."""
        self.hash_to_data = {}  # hash_id -> original_data
        self.data_to_hash = {}  # original_data -> hash_id
        self.hash_counter = 0

    def replace_large_data(self, obj: Any) -> Any:
        """Return a copy of `obj` with large string fields replaced by hash tokens."""
        if isinstance(obj, Type):
            # Handle DSPy custom types: Images
            return self._replace_in_custom_type(obj)
        elif hasattr(obj, "items") and not isinstance(obj, dict):
            new_mapping: dict[str, Any] = {}
            for k, v in obj.items():
                new_mapping[k] = self.replace_large_data(v)
            return new_mapping
        elif isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_dict[k] = self.replace_large_data(v)
            return new_dict
        elif isinstance(obj, list):
            new_list = []
            for item in obj:
                new_list.append(self.replace_large_data(item))
            return new_list
        else:
            return obj

    def restore_large_data(self, obj: Any) -> Any:
        """Return a copy of `obj` with hash tokens restored to original data."""
        if isinstance(obj, dict):
            if self._is_message_content_with_large_data(obj):
                return self._restore_in_message_content(obj)
            else:
                return {k: self.restore_large_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.restore_large_data(item) for item in obj]
        elif isinstance(obj, str):
            if self._is_hash_identifier(obj):
                return self.hash_to_data.get(obj, obj)
            else:
                return obj
        else:
            return obj

    def _create_hash_for_data(self, data: str) -> str:
        """Create a unique hash identifier for large data."""
        # Check if we've already hashed this exact data (deduplication)
        if data in self.data_to_hash:
            return self.data_to_hash[data]

        # Create new hash identifier
        hash_id = f"__DSPY_LARGE_DATA_HASH_{self.hash_counter}__"
        self.hash_counter += 1

        # Store bidirectional mapping
        self.hash_to_data[hash_id] = data
        self.data_to_hash[data] = hash_id

        return hash_id

    def _is_large_data(self, data: Any) -> bool:
        """Check if data is large enough to warrant optimization."""
        if isinstance(data, str):
            return len(data) > self.LARGE_DATA_THRESHOLD
        return False

    def _replace_in_custom_type(self, obj: Type) -> Type:
        """For Image type, replace large payload fields in known custom types; otherwise return `obj`."""
        if isinstance(obj, History):
            new_messages = self.replace_large_data(obj.messages)
            if new_messages is not obj.messages:  # Messages changed
                return type(obj)(messages=new_messages)
            else:
                return obj

        elif hasattr(obj, "url") and self._is_large_data(obj.url):
            # TODO: it only handles image url, not other types yet
            hash_url = self._create_hash_for_data(obj.url)
            return type(obj)(url=hash_url)

        return obj

    def _is_message_content_with_large_data(self, obj: dict) -> bool:
        """Whether this dict is an image content block with a payload field."""
        # {"type": "image_url", "image_url": {"url": "hash_id"}}
        return obj.get("type") == "image_url" and "image_url" in obj and "url" in obj["image_url"]

    def _restore_in_message_content(self, obj: dict) -> dict:
        """Restore payloads inside image/audio message content blocks."""
        obj_copy = copy.deepcopy(obj)

        if obj.get("type") == "image_url" and "image_url" in obj:
            url = obj["image_url"].get("url", "")
            if self._is_hash_identifier(url):
                obj_copy["image_url"]["url"] = self.hash_to_data.get(url, url)

        return obj_copy

    def _is_hash_identifier(self, text: str) -> bool:
        """True if `text` looks like a DSPy large-data hash token."""
        return isinstance(text, str) and text.startswith("__DSPY_LARGE_DATA_HASH_")
