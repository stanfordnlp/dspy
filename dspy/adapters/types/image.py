import base64
import io
import os
from typing import Any, Dict, List, Union
from urllib.parse import urlparse
import re
import mimetypes

import pydantic
import requests

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(pydantic.BaseModel):
    url: str
    
    model_config = {
        'frozen': True,
        'str_strip_whitespace': True,
        'validate_assignment': True,
        'extra': 'forbid',
    }
        
    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        # Allow the model to accept either a URL string or a dictionary with a single 'url' key
        if isinstance(values, str):
            # if a string, assume it's the URL directly and wrap it in a dict
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
        return cls(url=encode_image(pil_image))

    @pydantic.model_serializer()
    def serialize_model(self):
        return "<DSPY_IMAGE_START>" + self.url + "<DSPY_IMAGE_END>"

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            image_type = self.url.split(";")[0].split("/")[-1]
            return f"Image(url=data:image/{image_type};base64,<IMAGE_BASE_64_ENCODED({str(len_base64)})>)"
        return f"Image(url='{self.url}')"

def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
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
            print(f"Unsupported file string: {image}")
            raise ValueError(f"Unsupported file string: {image}")
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


def _encode_pil_image(image: 'PILImage') -> str:
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
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    return False

def try_expand_image_tags(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Try to expand image tags in the messages."""
    for message in messages:
        # NOTE: Assumption that content is a string
        if "content" in message and "<DSPY_IMAGE_START>" in message["content"]:
            message["content"] = expand_image_tags(message["content"])
    return messages

def expand_image_tags(text: str) -> Union[str, List[Dict[str, Any]]]:
    """Expand image tags in the text. If there are any image tags, 
    turn it from a content string into a content list of texts and image urls.
    
    Args:
        text: The text content that may contain image tags
        
    Returns:
        Either the original string if no image tags, or a list of content dicts
        with text and image_url entries
    """
    image_tag_regex = r'"?<DSPY_IMAGE_START>(.*?)<DSPY_IMAGE_END>"?'
    
    # If no image tags, return original text
    if not re.search(image_tag_regex, text):
        return text
        
    final_list = []
    remaining_text = text
    
    while remaining_text:
        match = re.search(image_tag_regex, remaining_text)
        if not match:
            if remaining_text.strip():
                final_list.append({"type": "text", "text": remaining_text.strip()})
            break
            
        # Get text before the image tag
        prefix = remaining_text[:match.start()].strip()
        if prefix:
            final_list.append({"type": "text", "text": prefix})
            
        # Add the image
        image_url = match.group(1)
        final_list.append({"type": "image_url", "image_url": {"url": image_url}})
        
        # Update remaining text
        remaining_text = remaining_text[match.end():].strip()
    
    return final_list
