import base64
import io
import os
import requests

try:
    from PIL import Image
except ImportError:
    Image = None

def encode_image(image):
    assert Image is not None, "Pillow is not installed. Please install it to use image fields."
    if hasattr(Image, 'Image') and isinstance(image, Image.Image):
        # PIL Image (including PngImageFile, JpegImageFile, etc.)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif isinstance(image, str):
        if os.path.isfile(image):
            # Local file path
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            # Assume it's a URL
            return encode_image_base64_from_url(image)
    else:
        raise ValueError(f"Unsupported image type. Must be PIL Image, file path, or URL. Got {type(image)}")

def encode_image_base64_from_url(image_url: str) -> str:
    """Encode an image retrieved from a remote url to base64 format."""
    with requests.get(image_url) as response:
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')