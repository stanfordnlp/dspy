import base64
import io
import os
from typing import Union
from urllib.parse import urlparse

import pydantic
import requests

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(pydantic.BaseModel):
    url: str

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        # Allow the model to accept either a URL string or a dictionary with a single 'url' key
        if isinstance(values, str):
            # if a string, assume it’s the URL directly and wrap it in a dict
            return {"url": values}
        elif isinstance(values, dict) and set(values.keys()) == {"url"}:
            # if it's a dict, ensure it has only the 'url' key
            return values
        elif isinstance(values, cls):
            return values.model_dump()
        else:
            raise TypeError("Expected a string URL or a dictionary with a key 'url'.")

    # If all my inits just call encode_image, should that be in this class
    @classmethod
    def from_url(cls, url: str, download: bool = False):
        return cls(url=encode_image(url, download))

    @classmethod
    def from_file(cls, file_path: str):
        return cls(url=encode_image(file_path))

    @classmethod
    def from_PIL(cls, pil_image):
        import PIL

        return cls(url=encode_image(PIL.Image.open(pil_image)))

    def __repr__(self):
        len_base64 = len(self.url.split("base64,")[1])
        return f"Image(url = {self.url.split('base64,')[0]}base64,<IMAGE_BASE_64_ENCODED({str(len_base64)})>)"


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def encode_image(image: Union[str, bytes, "PILImage.Image", dict], download_images: bool = False) -> str:
    """
    Encode an image to a base64 data URI.

    Args:
        image: The image to encode. Can be a PIL Image, file path, URL, or data URI.
        download_images: Whether to download images from URLs.

    Returns:
        str: The data URI of the image or the URL if download_images is False.

    Raises:
        ValueError: If the image type is not supported.
    """
    if isinstance(image, dict) and "url" in image:
        # NOTE: Not doing other validation for now
        return image["url"]
    elif isinstance(image, str):
        if image.startswith("data:image/"):
            # Already a data URI
            return image
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
            raise ValueError(f"Unsupported image string: {image}")
    elif PIL_AVAILABLE and isinstance(image, PILImage.Image):
        # PIL Image
        return _encode_pil_image(image)
    elif isinstance(image, bytes):
        # Raw bytes
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process image bytes.")
        img = Image.open(io.BytesIO(image))
        return _encode_pil_image(img)
    elif isinstance(image, Image):
        return image.url
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def _encode_image_from_file(file_path: str) -> str:
    """Encode an image from a file path to a base64 data URI."""
    with open(file_path, "rb") as image_file:
        image_data = image_file.read()
    file_extension = _get_file_extension(file_path)
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/{file_extension};base64,{encoded_image}"


def _encode_image_from_url(image_url: str) -> str:
    """Encode an image from a URL to a base64 data URI."""
    response = requests.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("image/"):
        file_extension = content_type.split("/")[-1]
    else:
        # Fallback to file extension from URL or default to 'png'
        file_extension = _get_file_extension(image_url) or "png"
    encoded_image = base64.b64encode(response.content).decode("utf-8")
    return f"data:image/{file_extension};base64,{encoded_image}"


def _encode_pil_image(image: "Image.Image") -> str:
    """Encode a PIL Image object to a base64 data URI."""
    buffered = io.BytesIO()
    file_extension = (image.format or "PNG").lower()
    image.save(buffered, format=file_extension)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{file_extension};base64,{encoded_image}"


def _get_file_extension(path_or_url: str) -> str:
    """Extract the file extension from a file path or URL."""
    extension = os.path.splitext(urlparse(path_or_url).path)[1].lstrip(".").lower()
    return extension or "png"  # Default to 'png' if no extension found


def is_image(obj) -> bool:
    """Check if the object is an image or a valid image reference."""
    if PIL_AVAILABLE and isinstance(obj, Image.Image):
        return True
    if isinstance(obj, (bytes, bytearray)):
        return True
    if isinstance(obj, str):
        if obj.startswith("data:image/"):
            return True
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    return False
