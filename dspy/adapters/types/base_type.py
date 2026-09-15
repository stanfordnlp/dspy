import json
import re
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

import pydantic

from dspy._vendor.lm15.serde import part_from_dict, part_to_dict
from dspy._vendor.lm15.types import Part, _is_part
from dspy.clients.base_lm import BaseLM
from dspy.lm15 import Response, TextPart

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"
_MARKER_PATTERN = re.compile(rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}", re.DOTALL)


class TypeFormatError(TypeError):
    """A `dspy.Type.format()` implementation returned something adapters cannot send."""


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses implement `format` to
    return either a string or a list of `dspy.lm15` content parts (`ImagePart`, `AudioPart`,
    `DocumentPart`, ...). Adapters place those parts in the user message exactly where the field
    value appears in the rendered prompt.

    Examples:

        ```python
        from dspy.lm15 import image

        class Image(Type):
            url: str

            def format(self) -> list:
                return [image(url=self.url)]
        ```
    """

    def format(self) -> list[Part] | str:
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
            if not all(_is_part(part) for part in formatted):
                raise TypeFormatError(
                    f"{type(self).__name__}.format() must return a string or dspy.lm15 content parts; "
                    "OpenAI-style content dictionaries are no longer accepted (DSPy 3.5)."
                )
            payload = json.dumps([part_to_dict(part) for part in formatted], ensure_ascii=False)
            return f"{CUSTOM_TYPE_START_IDENTIFIER}{payload}{CUSTOM_TYPE_END_IDENTIFIER}"
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
        return signature

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the custom type is streamable."""
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk) -> Optional["Type"]:
        """
        Parse a stream chunk into the custom type.

        Args:
            chunk: A listener-facing stream chunk (`chunk.choices[0].delta`).

        Returns:
            A custom type object or None if the chunk is not for this custom type.
        """
        return None

    @classmethod
    def parse_lm_response(cls, response: Response) -> Optional["Type"]:
        """Read the native representation of this type out of an lm15 `Response`.

        Args:
            response: The model's `dspy.lm15.Response`.

        Returns:
            A custom type object, or None when the response carries nothing for it.
        """
        return None


def parts_from_text(text: str) -> tuple[Part, ...]:
    """Expand the custom-type markers in rendered text into lm15 content parts.

    Adapters render every field value into one string. A `dspy.Type` whose
    `format()` returns content parts serializes them behind a marker, so that
    the parts land in the user message exactly where the field appears.
    """
    parts: list[Part] = []
    last_end = 0
    for match in _MARKER_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            parts.append(TextPart(text[last_end:start]))
        parts.extend(_parts_from_payload(match.group(1).strip()))
        last_end = end
    if last_end < len(text) or not parts:
        parts.append(TextPart(text[last_end:]))
    # Adjacent text stays one part, so a text-only message remains plain text.
    merged: list[Part] = []
    for part in parts:
        if merged and isinstance(part, TextPart) and isinstance(merged[-1], TextPart):
            merged[-1] = TextPart(merged[-1].text + part.text)
        else:
            merged.append(part)
    return tuple(merged)


def _parts_from_payload(payload: str) -> list[Part]:
    for parse in (json.loads, _parse_doubly_quoted_json):
        try:
            data = parse(payload)
            break
        except (ValueError, TypeError):
            continue
    else:
        return [TextPart(payload)]
    if not isinstance(data, list):
        return [TextPart(payload)]
    return [part_from_dict(item) if isinstance(item, dict) else TextPart(str(item)) for item in data]


def _parse_doubly_quoted_json(json_str: str) -> Any:
    """
    Parse a doubly quoted JSON string into a Python dict.
    `dspy.Type` can be json-encoded twice if included in either list or dict, e.g., `list[dspy.experimental.Document]`
    """
    return json.loads(json.loads(f'"{json_str}"'))


def split_data_uri(value: str) -> tuple[str, str] | None:
    """`data:<media_type>;base64,<payload>` -> (media_type, payload), else None."""
    if not value.startswith("data:"):
        return None
    head, sep, payload = value[5:].partition(",")
    if not sep or not head.endswith(";base64") or not payload:
        raise ValueError("A data URI must look like data:<media-type>;base64,<payload>")
    return head[: -len(";base64")], payload
