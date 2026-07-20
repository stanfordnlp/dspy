import base64
import io
import mimetypes
import os
import warnings
from functools import lru_cache
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type

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

    def __init__(self, source: Any = None, **data):
        """Create an Image.

        Parameters
        ----------
        source:
            The image source. Supported values include

            - ``str``: HTTP(S)/GS URL or an encoded data URI
            - ``bytes``: raw image bytes
            - ``PIL.Image.Image``: a PIL image instance
            - ``dict`` with a single ``{"url": value}`` entry (legacy form)
            - already encoded data URI

        Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.

        Local files and remote resources must be loaded explicitly with
        :meth:`from_file` and :meth:`from_url`, respectively.
        """

        if source is not None and "url" in data:
            raise TypeError("Image received both `source` and `url`; provide only one image source")

        if source is not None:
            # Support a positional source while retaining the Pydantic ``url`` field.
            if isinstance(source, dict) and set(source.keys()) == {"url"}:
                # Legacy dict form from previous model validator.
                data["url"] = source["url"]
            else:
                data["url"] = source

        if "url" in data:
            # Normalize any accepted input into a base64 data URI or plain URL.
            data["url"] = encode_image(data["url"])

        # Delegate the rest of initialization to pydantic's BaseModel.
        super().__init__(**data)

    @lru_cache(maxsize=32)
    def format(self) -> list[dict[str, Any]] | str:
        try:
            image_url = encode_image(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format image for DSPy: {e}")
        return [{"type": "image_url", "image_url": {"url": image_url}}]

    @classmethod
    def from_url(cls, url: str, verify: bool = True) -> "Image":
        """Download an HTTP(S) resource and encode it as a data URI."""
        if not _is_http_url(url):
            raise ValueError(f"Image.from_url requires an HTTP(S) URL, received: {url}")
        return cls(_encode_image_from_url(url, verify=verify))

    @classmethod
    def from_file(cls, file_path: str) -> "Image":
        """Read a local file and encode it as a data URI."""
        return cls(_encode_image_from_file(file_path))

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


def _is_http_url(string: str) -> bool:
    """Check if a string is an HTTP(S) URL."""
    try:
        result = urlparse(string)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except ValueError:
        return False


def encode_image(image: Union[str, bytes, "PILImage.Image", dict]) -> str:
    """
    Normalize an in-memory image or preserve a remote image reference.

    Args:
        image: A PIL Image, bytes, URL reference, data URI, or Image.

    Returns:
        str: A data URI or remote URL reference.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(image, dict) and "url" in image:
        return encode_image(image["url"])
    elif isinstance(image, str):
        if image.startswith("data:"):
            # Already a data URI
            return image
        elif is_url(image):
            return image
        else:
            raise ValueError(f"Unrecognized image string: {image}. Local files must be loaded with Image.from_file().")
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


def _encode_image_from_url(image_url: str, verify: bool = True) -> str:
    """Encode a file from a URL to a base64 data URI.

    Args:
        image_url: The URL of the image to download.
        verify: Whether to verify SSL certificates. Set to False for self-signed certs.
    """
    response = requests.get(image_url, verify=verify)
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


def is_image(obj) -> bool:
    """Check if the object is an image or a valid media file reference."""
    if PIL_AVAILABLE and isinstance(obj, PILImage.Image):
        return True
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return True
        elif is_url(obj):
            return True
    return False
