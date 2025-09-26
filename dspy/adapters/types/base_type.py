import json
import re
from typing import Any, Optional, get_args, get_origin

import json_repair
import pydantic
from litellm import ModelResponseStream

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
            return (
                f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}"
            )
        return formatted

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the custom type is streamable."""
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk: ModelResponseStream) -> Optional["Type"]:
        """
        Parse a stream chunk into the custom type.

        Args:
            chunk: A stream chunk.

        Returns:
            A custom type object or None if the chunk is not for this custom type.
        """
        return None


    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Type"]:
        """Parse a LM response into the custom type.

        Args:
            response: A LM response.

        Returns:
            A custom type object.
        """
        return None

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

        pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        result = []
        last_end = 0
        # DSPy adapter always formats user input into a string content before custom type splitting
        content: str = message["content"]

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            # Add text before the current block
            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            # Parse the JSON inside the block
            custom_type_content = match.group(1).strip()
            parsed = None

            for parse_fn in [json.loads, _parse_doubly_quoted_json, json_repair.loads]:
                try:
                    parsed = parse_fn(custom_type_content)
                    break
                except json.JSONDecodeError:
                    continue

            if parsed:
                for custom_type_content in parsed:
                    result.append(custom_type_content)
            else:
                # fallback to raw string if it's not valid JSON
                result.append({"type": "text", "text": custom_type_content})

            last_end = end

        if last_end == 0:
            # No custom type found, return the original message
            continue

        # Add any remaining text after the last match
        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages


def _parse_doubly_quoted_json(json_str: str) -> Any:
    """
    Parse a doubly quoted JSON string into a Python dict.
    `dspy.Type` can be json-encoded twice if included in either list or dict, e.g., `list[dspy.experimental.Document]`
    """
    return json.loads(json.loads(f'"{json_str}"'))
