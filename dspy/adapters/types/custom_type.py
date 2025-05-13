from typing import Any

class CustomType:
    """
    Base class to support creating custom types for DSPy signatures.

    Example (supporting images through custom type):

    class Image(pydantic.BaseModel, dspy.CustomType):
        url: str

        def __custom_format__(self):
            return [{"type": "image_url", "image_url": {"url": self.url}}]

        @classmethod
        def from_url(cls, url: str):
            return cls(url=url)

    """
    def __custom_format__(self) -> list[dict[str, Any]]:
        raise NotImplementedError("CustomType subclasses must implement __custom_format__")