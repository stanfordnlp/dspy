import base64
import io
import os
import requests
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

def encode_image(image) -> str:
    """Encode an image to base64 format.
    Returns:
        str: The base64 encoded image.
        str: The image format.
    """
    assert Image is not None, "Pillow is not installed. Please install it to use image fields."
    if isinstance(image, str) and image.startswith("data:image/"):
        return image
    if hasattr(Image, 'Image') and isinstance(image, Image.Image):
        buffered = io.BytesIO()
        file_extension = image.format
        image.save(buffered, format=file_extension)
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/{file_extension.lower()};base64,{encoded_image}"
    elif isinstance(image, str):
        if os.path.isfile(image):
            file_extension = image.split(".")[-1]
            with open(image, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/{file_extension.lower()};base64,{encoded_image}"
        else:
            # TODO(Isaac): Allow for users to decide to download the image or not
            # VLLM and OpenAI both support passing urls directly
            return image
    else:
        raise ValueError(f"Unsupported image type. Must be PIL Image, file path, or URL. Got {type(image)}")

def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        file_extension = image_url.split(".")[-1]
        encoded_image = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/{file_extension};base64,{encoded_image}"

def is_image(obj) -> bool:
    if PIL_AVAILABLE and isinstance(obj, Image.Image):
        return True
    if isinstance(obj, str) and obj.startswith("data:image/"):
        return True
    return False