import base64
import io
import mimetypes
import os
import warnings
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type

try:
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(Type):
    """An image input represented by a remote URL or data URI.

    Construction and adapter parsing never access the filesystem or network. Use
    :meth:`from_path` to read a local file, :meth:`from_url` to download a remote
    resource, :meth:`from_bytes` for encoded image bytes, or :meth:`from_pil` for a
    PIL image.
    """

    url: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @pydantic.field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Image.url must be an HTTP(S)/GS URL or data URI string")
        return encode_image(value)

    @lru_cache(maxsize=32)  # noqa: B019
    def format(self) -> list[dict[str, Any]] | str:
        try:
            image_url = encode_image(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format image for DSPy: {e}")
        return [{"type": "image_url", "image_url": {"url": image_url}}]

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        timeout: float | tuple[float, float] = 30.0,
        verify: bool | str = True,
    ) -> "Image":
        """Download an HTTP(S) resource and encode it as a data URI.

        This is a synchronous, blocking network operation.

        Security: this performs an explicit, caller-initiated fetch and applies no
        SSRF protection beyond requiring an HTTP(S) scheme. Like ``requests.get``, it
        will reach private, loopback, or cloud-metadata hosts and follow redirects to
        them. When ``url`` is derived from untrusted input, the caller is responsible
        for validating the host against an allowlist before calling this method.

        Args:
            url: The HTTP(S) URL to download.
            timeout: Connection and read timeout in seconds, either as one value or a tuple.
            verify: Whether to verify TLS certificates, or a path to a CA bundle.
        """
        if not _is_http_url(url):
            raise ValueError(f"Image.from_url requires an HTTP(S) URL, received: {url}")
        return cls(url=_encode_image_from_url(url, timeout=timeout, verify=verify))

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Image":
        """Read a local file and encode it as a data URI."""
        return cls(url=_encode_image_from_file(path))

    @classmethod
    def from_bytes(cls, image_bytes: bytes) -> "Image":
        """Create an Image from encoded image bytes."""
        return cls(url=_encode_image_from_bytes(image_bytes))

    @classmethod
    def from_pil(cls, pil_image: Any) -> "Image":
        """Create an Image from a PIL image."""
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process PIL images.")
        if not isinstance(pil_image, PILImage.Image):
            raise TypeError(f"pil_image must be a PIL image, received: {type(pil_image)}")
        return cls(url=_encode_pil_image(pil_image))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "Image":
        """Deprecated alias for :meth:`from_path`."""
        warnings.warn(
            "Image.from_file is deprecated and will be removed in 3.4; use Image.from_path instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_path(path)

    @classmethod
    def from_PIL(cls, pil_image):  # noqa: N802
        warnings.warn(
            "Image.from_PIL is deprecated and will be removed in 3.4; use Image.from_pil(pil_image) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_pil(pil_image)

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


def encode_image(image: str | Image) -> str:
    """
    Normalize an in-memory image or preserve a remote image reference.

    Args:
        image: A URL reference, data URI, or Image.

    Returns:
        str: A data URI or remote URL reference.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(image, str):
        if image.startswith("data:"):
            # Already a data URI
            return image
        elif is_url(image):
            return image
        else:
            raise ValueError(f"Unrecognized image string: {image}. Local files must be loaded with Image.from_path().")
    elif isinstance(image, Image):
        return image.url
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def _encode_image_from_file(path: str | os.PathLike[str]) -> str:
    """Encode a file from a file path to a base64 data URI."""
    with open(path, "rb") as file:
        file_data = file.read()

    path_string = os.fspath(path)
    mime_type, _ = mimetypes.guess_type(path_string)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {path_string}")

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_image_from_url(
    image_url: str,
    *,
    timeout: float | tuple[float, float],
    verify: bool | str,
) -> str:
    """Encode a file from a URL to a base64 data URI.

    Args:
        image_url: The URL of the image to download.
        timeout: Connection and read timeout in seconds, either as one value or a tuple.
        verify: Whether to verify SSL certificates. Set to False for self-signed certs.
    """
    response = requests.get(image_url, timeout=timeout, verify=verify)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type:
        mime_type = content_type.split(";", 1)[0].strip()
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(image_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {image_url}")

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_image_from_bytes(image_bytes: bytes) -> str:
    """Encode image bytes to a base64 data URI."""
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required to process image bytes.")
    try:
        image = PILImage.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        raise ValueError(f"Bytes could not be identified as an image: {e}") from e
    return _encode_pil_image(image)


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
