import json
from typing import Any, List, Union

from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from .image_utils import Image, encode_image


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


def format_field_value(field_info: FieldInfo, value: Any, assume_text=True) -> Union[str, dict]:
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
    elif isinstance(value, Image) or field_info.annotation == Image:
        # This validation should happen somewhere else
        # Safe to import PIL here because it's only imported when an image is actually being formatted
        try:
            import PIL
        except ImportError:
            raise ImportError("PIL is required to format images; Run `pip install pillow` to install it.")
        image_value = value
        if not isinstance(image_value, Image):
            if isinstance(image_value, dict) and "url" in image_value:
                image_value = image_value["url"]
            elif isinstance(image_value, str):
                image_value = encode_image(image_value)
            elif isinstance(image_value, PIL.Image.Image):
                image_value = encode_image(image_value)
            assert isinstance(image_value, str)
            image_value = Image(url=image_value)
        return {"type": "image_url", "image_url": image_value.model_dump()}
    else:
        return {"type": "text", "text": string_value}


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


def _format_input_list_field_value(value: List[Any]) -> str:
    """
    Formats the value of an input field of type List[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(value[0])

    return "\n".join([f"[{idx+1}] {_format_blob(txt)}" for idx, txt in enumerate(value)])


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
