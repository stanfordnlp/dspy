import logging
import requests
import base64
from typing import Any, List, Optional, Union

from dsp.modules.lm import LM

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

class GPT4Vision(LM):
    """Wrapper around OpenAI's GPT-4 Vision API."""

    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.api_key = api_key
        self.kwargs.update(kwargs)

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                file_extension = image_path.split(".")[-1].lower()
                if file_extension not in [".png", ".jpeg", ".jpg", ".webp", ".gif"]:
                    raise ValueError(f"Unsupported image format: {file_extension}")
                return base64.b64encode(image_data).decode('utf-8')
        except IOError as e:
            logger.error("Error reading image file: %s", str(e))
            raise e
        except Exception as e:
            logger.error("Unexpected error occurred during image encoding: %s", str(e))
            raise e

    def prepare_image_data(self, images: List[dict], is_url: bool = True) -> List[dict]:
        """Prepares the image data for the API call."""
        if not isinstance(images, list):
            raise ValueError("Images must be provided as a list of dictionaries")

        supported_formats = [".png", ".jpeg", ".jpg", ".webp", ".gif"]
        max_width = 2048
        max_height = 2048

        image_data = []
        for image in images:
            if is_url:
                if "url" not in image:
                    raise ValueError("Image dictionary must contain a 'url' key for URL-based images")
                url = image["url"]
                if not url.startswith(("http://", "https://")):
                    raise ValueError("Invalid image URL")
                # You can perform additional URL validation if needed
            else:
                if "data" not in image:
                    raise ValueError("Image dictionary must contain a 'data' key for base64-encoded images")
                base64_data = image["data"]
                if not base64_data.startswith("data:image/"):
                    raise ValueError("Invalid base64-encoded image data")
                # You can perform additional base64 validation if needed

            width = image.get("width")
            height = image.get("height")
            if width is not None and height is not None:
                if width > max_width or height > max_height:
                    raise ValueError(f"Image dimensions exceed the maximum allowed size ({max_width}x{max_height})")

            # Check the image file format based on the file extension
            file_extension = url.split(".")[-1].lower() if is_url else base64_data.split("/")[1].split(";")[0]
            if file_extension not in supported_formats:
                raise ValueError(f"Unsupported image format: {file_extension}")

            if is_url:
                image_data.append({"type": "image_url", "image_url": {"url": url, "detail": image.get("detail", "auto"), "width": width, "height": height}})
            else:
                image_data.append({"type": "image_url", "image_url": {"url": base64_data, "detail": image.get("detail", "auto"), "width": width, "height": height}})

        return image_data

    def calculate_image_tokens(self, images: List[dict]) -> int:
        """Calculates the token cost of the images."""
        total_tokens = 0
        for image in images:
            if image.get("detail", "auto") == "low":
                total_tokens += 85
            else:
                # Calculate token cost for high detail images
                width, height = image["width"], image["height"]

                # Scale down the image to fit within a 2048 x 2048 square
                if max(width, height) > 2048:
                    scale_factor = 2048 / max(width, height)
                    width = int(width * scale_factor)
                    height = int(height * scale_factor)

                # Scale the image such that the shortest side is 768px long
                if min(width, height) > 768:
                    scale_factor = 768 / min(width, height)
                    width = int(width * scale_factor)
                    height = int(height * scale_factor)

                # Count the number of 512px square tiles needed
                num_tiles = ((width + 511) // 512) * ((height + 511) // 512)

                # Calculate the token cost
                total_tokens += 170 * num_tiles + 85

        return total_tokens

    def basic_request(self, prompt: Union[str, List[dict]], **kwargs) -> Any:
        """Makes a basic request to the GPT-4 Vision model, supporting text and image prompts."""
        if not isinstance(prompt, list):
            prompt = [{"type": "text", "content": prompt}]
        else:
            prompt = [{"type": "text", "content": p["content"]} if p["type"] == "text" else {"type": "image_url", "image_url": p["image_url"]} for p in prompt]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.kwargs["model"],
            "messages": prompt,
            **self.kwargs
        }

        image_tokens = self.calculate_image_tokens([p["image_url"] for p in prompt if p["type"] == "image_url"])
        data["max_tokens"] = data.get("max_tokens", 300) + image_tokens

        if openai is None:
            raise ImportError("The 'openai' module is required for GPT4Vision but could not be imported.")

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        except requests.exceptions.RequestException as e:
            logger.error("Error occurred during API request: %s", str(e))
            raise e

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            logger.error("Bad request: %s", response.text)
            raise ValueError("Invalid request payload")
        elif response.status_code == 401:
            logger.error("Unauthorized: %s", response.text)
            raise ValueError("Invalid API key")
        elif response.status_code == 429:
            logger.error("Rate limit exceeded: %s", response.text)
            raise ValueError("Rate limit exceeded. Please try again later.")
        else:
            logger.error("Unexpected error occurred: %s", response.text)
            raise ValueError(f"Unexpected error occurred with status code {response.status_code}")

    def __call__(self, prompts: List[dict], **kwargs) -> List[Any]:
        """
        Retrieves responses from GPT-4 Vision based on the prompts.

        Args:
            prompts (List[dict]): A list of prompts with text or images.
            **kwargs: Additional arguments for the API call.

        Returns:
            List[Any]: The list of responses from the model.
        """
        return self.request(prompts, **kwargs)

    # Add any additional methods needed for specific GPT-4 Vision API functionalities.