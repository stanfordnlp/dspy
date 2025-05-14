import json
from typing import Any, Union

import pydantic


class BaseType(pydantic.BaseModel):
    """
    Base class to support creating custom types for DSPy signatures.

    Example (supporting images through custom type):

    class Image(dspy.BaseType):
        url: str

        def format(self):
            return [{"type": "image_url", "image_url": {"url": self.url}}]

        @classmethod
        def from_url(cls, url: str):
            return cls(url=url)

    """

    def format(self) -> Union[list[dict[str, Any]], str]:
        raise NotImplementedError

    @pydantic.model_serializer()
    def serialize_model(self):
        return f"<<CUSTOM-TYPE-START-IDENTIFIER>>{self.format()}<<CUSTOM-TYPE-END-IDENTIFIER>>"
