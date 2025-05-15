import json
import re
from typing import Any

import json_repair
import pydantic


class BaseType(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Example:

        ```python
        class Image(BaseType):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @pydantic.model_serializer()
    def serialize_model(self):
        return f"<<CUSTOM-TYPE-START-IDENTIFIER>>{self.format()}<<CUSTOM-TYPE-END-IDENTIFIER>>"


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
    and `<<CUSTOM-TYPE-END-IDENTIFIER>>` are the reserved identifiers for the custom types as in `dspy.BaseType`.

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

        pattern = r"<<CUSTOM-TYPE-START-IDENTIFIER>>(.*?)<<CUSTOM-TYPE-END-IDENTIFIER>>"
        result = []
        last_end = 0
        content = message["content"]

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            # Add text before the current block
            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            # Parse the JSON inside the block
            custom_type_content = match.group(1).strip()
            try:
                parsed = json_repair.loads(custom_type_content)
                for custom_type_content in parsed:
                    result.append(custom_type_content)
            except json.JSONDecodeError:
                # fallback to raw string if it's not valid JSON
                parsed = {"type": "text", "text": custom_type_content}
                result.append(parsed)

            last_end = end

        # Add any remaining text after the last match
        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages
