from typing import Any

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
