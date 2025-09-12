import base64
import io
import json
import mimetypes
import os
import re
import warnings
from functools import lru_cache
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import CUSTOM_TYPE_END_IDENTIFIER, CUSTOM_TYPE_START_IDENTIFIER, Type

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(Type):
    url: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, url: Any = None, *, download: bool = False, **data):
        """Create an Image.

        Parameters
        ----------
        url:
            The image source. Supported values include

            - ``str``: HTTP(S)/GS URL or local file path
            - ``bytes``: raw image bytes
            - ``PIL.Image.Image``: a PIL image instance
            - ``dict`` with a single ``{"url": value}`` entry (legacy form)
            - already encoded data URI

        download:
            Whether remote URLs should be downloaded to infer their MIME type.

        Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.
        """
        if url is not None and "url" not in data:
            # Support a positional argument while allowing ``url=`` in **data.
            if isinstance(url, dict) and set(url.keys()) == {"url"}:
                # Legacy dict form from previous model validator.
                data["url"] = url["url"]
            else:
                # ``url`` may be a string, bytes, or a PIL image.
                data["url"] = url

        if "url" in data:
            # Normalize any accepted input into a base64 data URI or plain URL.
            data["url"] = encode_image(data["url"], download_images=download)

        # Delegate the rest of initialization to pydantic's BaseModel.
        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        """Validate and normalize image input data."""
        if isinstance(data, cls):
            return data

        # Handle positional argument case where data is not a dict
        if not isinstance(data, dict):
            # Convert non-dict input to dict format
            data = {"url": data}

        # Handle legacy dict form with single "url" key
        if isinstance(data.get("url"), dict) and set(data["url"].keys()) == {"url"}:
            data["url"] = data["url"]["url"]

        # Extract download parameter if present, defaulting to False
        download = data.pop("download", False) if isinstance(data, dict) else False

        if "url" not in data:
            raise ValueError("url field is required for Image")

        # Normalize any accepted input into a base64 data URI or plain URL
        data["url"] = encode_image(data["url"], download_images=download)

        return data

    @lru_cache(maxsize=32)
    def format(self) -> list[dict[str, Any]] | str:
        # URL is already encoded in the model validator, no need to re-encode
        return [{"type": "image_url", "image_url": {"url": self.url}}]

    @classmethod
    def from_url(cls, url: str, download: bool = False):
        warnings.warn(
            "Image.from_url is deprecated; use Image(url) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(url, download=download)

    @classmethod
    def from_file(cls, file_path: str):
        warnings.warn(
            "Image.from_file is deprecated; use Image(file_path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(file_path)

    @classmethod
    def from_PIL(cls, pil_image):  # noqa: N802
        warnings.warn(
            "Image.from_PIL is deprecated; use Image(pil_image) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(pil_image)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            image_type = self.url.split(";")[0].split("/")[-1]
            return f"Image(url=data:image/{image_type};base64,<IMAGE_BASE_64_ENCODED({len_base64!s})>)"
        return f"Image(url='{self.url}')"


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def encode_image(image: Union[str, bytes, "PILImage.Image", dict], download_images: bool = False) -> str:
    """
    Encode an image or file to a base64 data URI.

    Args:
        image: The image or file to encode. Can be a PIL Image, file path, URL, or data URI.
        download_images: Whether to download images from URLs.

    Returns:
        str: The data URI of the file or the URL if download_images is False.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(image, dict) and "url" in image:
        # NOTE: Not doing other validation for now
        return image["url"]
    elif isinstance(image, str):
        if image.startswith("data:"):
            # Already a data URI
            return image
        elif _is_custom_type_format(image):
            # Extract image URL from custom type format
            url = _extract_url_from_custom_type(image)
            return encode_image(url, download_images)
        elif os.path.isfile(image):
            # File path
            return _encode_image_from_file(image)
        elif is_url(image):
            # URL
            if download_images:
                return _encode_image_from_url(image)
            else:
                # Return the URL as is
                return image
        else:
            # Unsupported string format
            raise ValueError(f"Unrecognized file string: {image}; If this file type should be supported, please open an issue.")
    elif PIL_AVAILABLE and isinstance(image, PILImage.Image):
        # PIL Image
        return _encode_pil_image(image)
    elif isinstance(image, bytes):
        # Raw bytes
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process image bytes.")
        img = PILImage.open(io.BytesIO(image))
        return _encode_pil_image(img)
    elif isinstance(image, Image):
        return image.url
    else:
        print(f"Unsupported image type: {type(image)}")
        raise ValueError(f"Unsupported image type: {type(image)}")


def _encode_image_from_file(file_path: str) -> str:
    """Encode a file from a file path to a base64 data URI."""
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Use mimetypes to guess directly from the file path
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_image_from_url(image_url: str) -> str:
    """Encode a file from a URL to a base64 data URI."""
    response = requests.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type:
        mime_type = content_type
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(image_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {image_url}")

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_pil_image(image: "PILImage") -> str:
    """Encode a PIL Image object to a base64 data URI."""
    buffered = io.BytesIO()
    file_format = image.format or "PNG"
    image.save(buffered, format=file_format)

    # Get the correct MIME type using the image format
    file_extension = file_format.lower()
    mime_type, _ = mimetypes.guess_type(f"file.{file_extension}")
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for image format: {file_format}")

    encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _get_file_extension(path_or_url: str) -> str:
    """Extract the file extension from a file path or URL."""
    extension = os.path.splitext(urlparse(path_or_url).path)[1].lstrip(".").lower()
    return extension or "png"  # Default to 'png' if no extension found


def _is_custom_type_format(text: str) -> bool:
    """Check if the text contains custom type format."""
    pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
    return bool(re.search(pattern, text))


def _extract_url_from_custom_type(text: str) -> str:
    """Extract the image URL from custom type format string."""
    pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        raise ValueError("No custom type format found in string")

    custom_content = match.group(1).strip()

    try:
        # Try to parse as JSON
        parsed = json.loads(custom_content.replace("'", '"'))

        # Extract URL from the image_url structure
        if isinstance(parsed, list) and len(parsed) > 0:
            first_item = parsed[0]
            if isinstance(first_item, dict) and first_item.get("type") == "image_url":
                image_url_dict = first_item.get("image_url", {})
                if "url" in image_url_dict:
                    return image_url_dict["url"]

        raise ValueError("Could not find image URL in custom type format")

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in custom type format: {custom_content}")


def is_image(obj) -> bool:
    """Check if the object is an image or a valid media file reference."""
    if PIL_AVAILABLE and isinstance(obj, PILImage.Image):
        return True
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return True
        elif _is_custom_type_format(obj):
            return True
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    return False
