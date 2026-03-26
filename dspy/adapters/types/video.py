import base64
import mimetypes
import os
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20 MB (raw file size; prevents loading very large files into memory)


class Video(Type):
    """A video input type for DSPy.

    Currently only supported by Gemini models via litellm. Supports two modes:

    1. **Inline**: Pass video bytes directly (suitable for videos under 20 MB).
    2. **File reference**: Reference a video already uploaded via Google's Files API (Google's preferred method)

    Example:
        ```python
        import dspy

        class Describe(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            description: str = dspy.OutputField()

        program = dspy.Predict(Describe)

        # Inline from a local file (< 20 MB)
        result = program(video=dspy.Video.from_file("video.mp4"))

        # Reference a file uploaded via Google's Files API
        result = program(video=dspy.Video.from_url("https://generativelanguage.googleapis.com/..."))
        ```
    """

    url: str | None = None
    inline_data: str | None = None
    mime_type: str = "video/mp4"

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        if isinstance(values, cls):
            return {"url": values.url, "inline_data": values.inline_data, "mime_type": values.mime_type}

        if isinstance(values, str):
            if values.startswith(("http://", "https://", "gs://")):
                return {"url": values}
            if os.path.isfile(values):
                return _encode_video_from_file(values)
            raise ValueError(f"'{values}' is not a valid URL or existing file path.")

        if isinstance(values, dict):
            if "url" in values or "inline_data" in values:
                return values
            raise ValueError("Dict must contain 'url' or 'inline_data'.")

        if isinstance(values, bytes):
            encoded = base64.b64encode(values).decode("utf-8")
            return {"inline_data": encoded}

        raise TypeError(
            f"Unsupported input type {type(values).__name__}. "
            "Expected a URL string, file path, bytes, or dict with 'url'/'inline_data'."
        )

    @pydantic.model_validator(mode="after")
    def validate_source(self):
        has_url = self.url is not None
        has_inline_data = self.inline_data is not None
        if has_url == has_inline_data:
            raise ValueError("Exactly one of 'url' or 'inline_data' must be provided")
        return self

    def format(self) -> list[dict[str, Any]]:
        file_dict = {}
        if self.inline_data is not None:
            file_dict["file_data"] = f"data:{self.mime_type};base64,{self.inline_data}"
        else:
            file_dict["file_id"] = self.url
            file_dict["format"] = self.mime_type
        return [{"type": "file", "file": file_dict}]

    @classmethod
    def from_url(cls, url: str, mime_type: str = "video/mp4") -> "Video":
        """Create a Video from a Google Files API reference or other URL.

        The URL is passed directly to the model provider without downloading.
        """
        return cls(url=url, mime_type=mime_type)

    @classmethod
    def from_file(cls, file_path: str, mime_type: str | None = None) -> "Video":
        """Create a Video from a local file, encoding it as inline base64 data."""
        result = _encode_video_from_file(file_path, mime_type=mime_type)
        return cls(**result)

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str = "video/mp4") -> "Video":
        """Create a Video from raw bytes."""
        encoded = base64.b64encode(data).decode("utf-8")
        return cls(inline_data=encoded, mime_type=mime_type)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if self.url is not None:
            return f"Video(url='{self.url}')"
        return f"Video(inline_data=<base64:{len(self.inline_data or '')} chars>)"


def _encode_video_from_file(file_path: str, mime_type: str | None = None) -> dict:
    """Encode a local video file as base64."""
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "video/mp4"

    file_size = os.path.getsize(file_path)
    if file_size > MAX_INLINE_SIZE:
        raise ValueError(
            f"File '{file_path}' is {file_size / (1024 * 1024):.1f} MB, which exceeds the "
            f"{MAX_INLINE_SIZE / (1024 * 1024):.0f} MB limit for inline video data. "
            "Consider uploading via Google's Files API and using Video.from_url() instead."
        )

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return {"inline_data": encoded, "mime_type": mime_type}
