import json
import re
from typing import Any, get_args, get_origin

import json_repair
import pydantic

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Example:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type"""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom types from the annotation.

        This is used to extract all custom types from the annotation of a field, while the annotation can
        have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
        """
        # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
        # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result = []
        # Recurse into all type args
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        formatted = self.format()
        if isinstance(formatted, list):
            return f"{CUSTOM_TYPE_START_IDENTIFIER}{formatted}{CUSTOM_TYPE_END_IDENTIFIER}"
        return formatted


def split_message_content_for_custom_types(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split user message content into a list of content blocks.

    This method splits each user message's content in the `messages` list to be a list of content block, so that
    the custom types like `dspy.Image` can be properly formatted for better quality. For example, the split content
    may look like below if the user message has a `dspy.Image` object:

    ```
    [
        {"type": "text", "text": "{text_before_image}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
        {"type": "text", "text": "{text_after_image}"},
    ]
    ```

    This is implemented by finding the `<<CUSTOM-TYPE-START-IDENTIFIER>>` and `<<CUSTOM-TYPE-END-IDENTIFIER>>`
    in the user message content and splitting the content around them. The `<<CUSTOM-TYPE-START-IDENTIFIER>>`
    and `<<CUSTOM-TYPE-END-IDENTIFIER>>` are the reserved identifiers for the custom types as in `dspy.Type`.

    Args:
        messages: a list of messages sent to the LM. The format is the same as [OpenAI API's messages
            format](https://platform.openai.com/docs/guides/chat-completions/response-format).

    Returns:
        A list of messages with the content split into a list of content blocks around custom types content.
    """
    for message in messages:
        if message["role"] != "user":
            # Custom type messages are only in user messages
            continue

        # DSPy adapter always formats user input into a string content before custom type splitting
        content: str = message["content"]

        # Look for JSON array patterns that contain custom types
        # Pattern: [..."<<CUSTOM-TYPE-START-IDENTIFIER>>...<<CUSTOM-TYPE-END-IDENTIFIER>>"...]
        json_array_pattern = r"\[([^[\]]*" + re.escape(CUSTOM_TYPE_START_IDENTIFIER) + r".*?" + re.escape(CUSTOM_TYPE_END_IDENTIFIER) + r"[^[\]]*)\]"

        # Find all JSON arrays containing custom types
        array_matches = list(re.finditer(json_array_pattern, content, re.DOTALL))

        if array_matches:
            # Process content by replacing JSON arrays with their processed content
            result = []
            last_end = 0

            for array_match in array_matches:
                array_start, array_end = array_match.span()

                # Add text before this array (process any standalone custom types)
                if array_start > last_end:
                    text_before = content[last_end:array_start]
                    text_result = _process_standalone_custom_types(text_before)
                    result.extend(text_result)

                # Process the JSON array
                array_content = array_match.group(0)  # Full match including [ ]
                array_result = _process_json_array_with_custom_types(array_content)
                result.extend(array_result)

                last_end = array_end

            # Add any remaining content after the last array
            if last_end < len(content):
                remaining_text = content[last_end:]
                text_result = _process_standalone_custom_types(remaining_text)
                result.extend(text_result)

            if result:
                message["content"] = result
                continue

        # Original logic for standalone custom types
        pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        result = []
        last_end = 0

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            # Add text before the current block
            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            # Parse the JSON inside the block
            custom_type_content = match.group(1).strip()
            try:
                try:
                    # Replace single quotes with double quotes to make it valid JSON
                    parsed = json.loads(custom_type_content.replace("'", '"'))
                except json.JSONDecodeError:
                    parsed = json_repair.loads(custom_type_content)
                for custom_type_content in parsed:
                    result.append(custom_type_content)

            except json.JSONDecodeError:
                # fallback to raw string if it's not valid JSON
                parsed = {"type": "text", "text": custom_type_content}
                result.append(parsed)

            last_end = end

        if last_end == 0:
            # No custom type found, return the original message
            continue

        # Add any remaining text after the last match
        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages


def _process_json_array_with_custom_types(array_content):
    """Process a JSON array that contains custom types"""
    try:
        # Replace custom type markers with placeholders
        custom_type_pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        matches = list(re.finditer(custom_type_pattern, array_content, re.DOTALL))

        temp_content = array_content
        placeholders = {}

        for i, match in enumerate(matches):
            placeholder = f"__CUSTOM_TYPE_PLACEHOLDER_{i}__"
            placeholders[placeholder] = match.group(1).strip()
            temp_content = temp_content.replace(match.group(0), placeholder)

        # Parse the JSON array
        array_items = json.loads(temp_content)

        # Process each item in the array
        result = []
        for item in array_items:
            if isinstance(item, str) and item in placeholders:
                custom_type_content = placeholders[item]
                try:
                    parsed = json_repair.loads(custom_type_content.replace("'", '"'))
                    for parsed_item in parsed:
                        result.append(parsed_item)
                except json.JSONDecodeError:
                    result.append({"type": "text", "text": custom_type_content})
            else:
                result.append({"type": "text", "text": str(item)})

        return result

    except (json.JSONDecodeError, ValueError):
        # If JSON parsing fails, fall back to standalone processing
        return _process_standalone_custom_types(array_content)


def _process_standalone_custom_types(content):
    """Process standalone custom types (original logic)"""
    pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
    result = []
    last_end = 0

    for match in re.finditer(pattern, content, re.DOTALL):
        start, end = match.span()

        # Add text before the current block
        if start > last_end:
            result.append({"type": "text", "text": content[last_end:start]})

        # Parse the custom type
        custom_type_content = match.group(1).strip()
        try:
            parsed = json_repair.loads(custom_type_content.replace("'", '"'))
            for parsed_item in parsed:
                result.append(parsed_item)
        except json.JSONDecodeError:
            result.append({"type": "text", "text": custom_type_content})

        last_end = end

    # If no custom types found, return as text
    if last_end == 0:
        return [{"type": "text", "text": content}]

    # Add any remaining text
    if last_end < len(content):
        result.append({"type": "text", "text": content[last_end:]})

    return result
