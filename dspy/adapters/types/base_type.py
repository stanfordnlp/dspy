import json
import re
import warnings
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

import json_repair
import pydantic

from dspy.clients.base_lm import BaseLM

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.signatures.signature import Signature

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
"""Legacy custom-type marker, deprecated since DSPy 3.3 and scheduled for removal in DSPy 3.5."""

CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"
"""Legacy custom-type marker, deprecated since DSPy 3.3 and scheduled for removal in DSPy 3.5."""


def warn_legacy_type_method(method: str) -> None:
    """Warn that a legacy `dspy.Type` rendering/parsing hook is deprecated."""
    warnings.warn(
        f"{method} is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
        "Adapter type rendering and parsing now use normalized LM types in the adapter pipeline.",
        DeprecationWarning,
        stacklevel=3,
    )


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Examples:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        """Return the legacy provider-shaped representation of this value.

        Deprecated:
            Since DSPy 3.3. Adapter type rendering now happens through normalized
            LM types in the adapter pipeline. This compatibility hook will be
            removed in DSPy 3.5.
        """
        warnings.warn(
            "Type.format() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
            "Adapter type rendering now happens through normalized LM types in the adapter pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError

    def to_lm_parts(self):
        """Return normalized LM parts for adapter rendering, if supported.

        Subclasses should override this to participate in the marker-free
        adapter pipeline. Returning ``None`` keeps the legacy marker-based
        serializer as a compatibility fallback.
        """
        return None

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
        """Serialize this value through the legacy custom-type marker protocol.

        Deprecated:
            Since DSPy 3.3. The marker protocol is kept only for compatibility
            and will be removed in DSPy 3.5.
        """
        warnings.warn(
            "Type.serialize_model() and the custom-type marker protocol are deprecated since DSPy 3.3 "
            "and will be removed in DSPy 3.5. Adapter type rendering now uses normalized LM types.",
            DeprecationWarning,
            stacklevel=2,
        )
        formatted = self.format()
        if isinstance(formatted, list):
            return (
                f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}"
            )
        return formatted

    @classmethod
    def adapt_to_native_lm_feature(
        cls,
        signature: type["Signature"],
        field_name: str,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> type["Signature"]:
        """Adapt the custom type to the native LM feature if possible.

        Deprecated:
            Since DSPy 3.3. Native feature planning is moving to the adapter
            pipeline and normalized LM types. This hook will be removed in DSPy
            3.5.

        When the LM and configuration supports the related native LM feature, e.g., native tool calling, native
        reasoning, etc., we adapt the signature and `lm_kwargs` to enable the native LM feature.

        Args:
            signature: The DSPy signature for the LM call.
            field_name: The name of the field in the signature to adapt to the native LM feature.
            lm: The LM instance.
            lm_kwargs: The keyword arguments for the LM call, subject to in-place updates if adaptation if required.

        Returns:
            The adapted signature. If the custom type is not natively supported by the LM, return the original
            signature.
        """
        warnings.warn(
            "Type.adapt_to_native_lm_feature() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
            "Native feature planning now belongs to the adapter pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        return signature

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the custom type is streamable.

        Deprecated:
            Since DSPy 3.3. Stream parsing is moving to normalized LM stream
            events in the adapter pipeline. This hook will be removed in DSPy
            3.5.
        """
        warnings.warn(
            "Type.is_streamable() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
            "Stream parsing now belongs to normalized LM stream handling.",
            DeprecationWarning,
            stacklevel=2,
        )
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk: "ModelResponseStream") -> Optional["Type"]:
        """
        Parse a stream chunk into the custom type.

        Deprecated:
            Since DSPy 3.3. Stream parsing is moving to normalized LM stream
            events in the adapter pipeline. This hook will be removed in DSPy
            3.5.

        Args:
            chunk: A stream chunk.

        Returns:
            A custom type object or None if the chunk is not for this custom type.
        """
        warnings.warn(
            "Type.parse_stream_chunk() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
            "Stream parsing now belongs to normalized LM stream handling.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Type"]:
        """Parse a LM response into the custom type.

        Deprecated:
            Since DSPy 3.3. Response parsing is moving to normalized LM responses
            in the adapter pipeline. This hook will be removed in DSPy 3.5.

        Args:
            response: A LM response.

        Returns:
            A custom type object.
        """
        warnings.warn(
            "Type.parse_lm_response() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
            "Response parsing now belongs to normalized LM response handling in the adapter pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None


def split_message_content_for_custom_types(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split user message content into a list of content blocks.

    Deprecated:
        Since DSPy 3.3. Adapter type rendering now uses normalized LM parts
        instead of marker strings. This compatibility function will be removed
        in DSPy 3.5.

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
    warnings.warn(
        "split_message_content_for_custom_types() is deprecated since DSPy 3.3 and will be removed in DSPy 3.5. "
        "Adapter type rendering now uses normalized LM parts.",
        DeprecationWarning,
        stacklevel=2,
    )

    for message in messages:
        if message["role"] != "user":
            # Custom type messages are only in user messages
            continue

        content = message["content"]
        if isinstance(content, str):
            result, changed = _split_custom_type_markers_in_text(content)
            if changed:
                message["content"] = result
            continue

        if isinstance(content, list):
            result = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                    split_blocks, block_changed = _split_custom_type_markers_in_text(block["text"])
                    result.extend(split_blocks)
                    changed = changed or block_changed
                else:
                    result.append(block)
            if changed:
                message["content"] = result

    return messages


def _split_custom_type_markers_in_text(content: str) -> tuple[list[dict[str, Any]], bool]:
    pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
    result = []
    last_end = 0

    for match in re.finditer(pattern, content, re.DOTALL):
        start, end = match.span()

        if start > last_end:
            result.append({"type": "text", "text": content[last_end:start]})

        custom_type_content = match.group(1).strip()
        parsed = None

        for parse_fn in [json.loads, _parse_doubly_quoted_json, json_repair.loads]:
            try:
                parsed = parse_fn(custom_type_content)
                break
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        if parsed:
            for custom_type_content in parsed:
                result.append(custom_type_content)
        else:
            result.append({"type": "text", "text": custom_type_content})

        last_end = end

    if last_end == 0:
        return [{"type": "text", "text": content}], False

    if last_end < len(content):
        result.append({"type": "text", "text": content[last_end:]})

    return result, True


def _parse_doubly_quoted_json(json_str: str) -> Any:
    """
    Parse a doubly quoted JSON string into a Python dict.
    `dspy.Type` can be json-encoded twice if included in either list or dict, e.g., `list[dspy.experimental.Document]`
    """
    return json.loads(json.loads(f'"{json_str}"'))
