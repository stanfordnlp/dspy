import base64
import io
import logging
import mimetypes
import os
import warnings
from typing import Any, Union
from urllib.parse import urlparse

import litellm
import pydantic
import requests

from dspy.adapters.types.base_type import BaseType

logger = logging.getLogger(__name__)


class PDF(BaseType):
    url: str

    model_config = {
        "frozen": True,
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }

    @staticmethod
    def supports_pdf_input(model: str) -> bool:
        """
        Check if a model supports PDF input based on model name.
        
        Args:
            model: The model name to check (e.g., "openai/gpt-4o")
            
        Returns:
            bool: True if the model supports PDF input, False otherwise
        """
        # First try LiteLLM's supports_pdf_input if available (in newer versions)
        try:
            if hasattr(litellm, "supports_pdf_input"):
                return litellm.supports_pdf_input(model)
        except (AttributeError, ImportError, Exception):
            pass  # Fall back to our own logic if litellm function isn't available
            
        # Manual check for known compatible models
        model = model.lower() if model else ""
        
        # Anthropic models
        if "claude-3" in model or "claude3" in model:
            return True
            
        # Vertex AI models
        if "gemini" in model:
            return True
            
        # Bedrock Claude models
        if "anthropic.claude" in model:
            return True
            
        # For models we're not sure about, warn but assume incompatible
        logger.warning(f"Unable to verify PDF support for model: {model}")
        return False
    
    def format(self) -> Union[list[dict[str, Any]], str]:
        """
        Format the PDF for inclusion in prompts according to LiteLLM document understanding format.
        
        Returns:
            A formatted representation of the PDF for inclusion in prompts
        
        Raises:
            ValueError: If formatting fails
        """
        try:
            # Check the current model from the DSPy context if available
            from dspy.dsp.utils.settings import settings
            current_model = None
            if hasattr(settings, "llm") and settings.llm and hasattr(settings.llm, "model"):
                current_model = settings.llm.model
                
                # Warn user if model doesn't support PDF input
                if current_model and not self.supports_pdf_input(current_model):
                    warnings.warn(
                        f"PDF input may not be supported by the current model '{current_model}'. "
                        f"PDF input is typically supported by Anthropic, Vertex AI, and Bedrock models. "
                        f"This may result in errors or unexpected behavior.",
                        UserWarning
                    )
            
            pdf_url = encode_pdf(self.url)
            
            # Format according to LiteLLM document understanding format
            if pdf_url.startswith("data:"):
                # Base64 encoded data
                return [{"type": "file", "file": {"file_data": pdf_url}}]
            else:
                # URL
                return [{"type": "file", "file": {"file_id": pdf_url, "format": "application/pdf"}}]
        except Exception as e:
            raise ValueError(f"Failed to format PDF for DSPy: {e}")

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

    @classmethod
    def from_url(cls, url: str, download: bool = False):
        return cls(url=encode_pdf(url, download))

    @classmethod
    def from_file(cls, file_path: str):
        return cls(url=encode_pdf(file_path))

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            return f"PDF(url=data:application/pdf;base64,<PDF_BASE_64_ENCODED({len_base64!s})>)"
        return f"PDF(url='{self.url}')"
        
    @staticmethod
    def get_supported_models_info():
        """
        Get information about models that are known to support PDF input.
        
        Returns:
            str: A string with information about supported models and formats
        """
        return """
        PDF Support Information:
        
        1. Supported Model Families:
           - Anthropic Models (claude-3-opus, claude-3-sonnet, claude-3-haiku)
           - Vertex AI Gemini models (gemini-pro, gemini-pro-vision)
           - Vertex AI Claude models
           - Bedrock Claude models
           - Bedrock Gemini models
        
        2. Supported Input Formats:
           - Base64 encoded data: Recommended for most models
           - Direct URLs: Works with some models but may be less reliable
           
        3. File Size Limits:
           - Anthropic models: Generally up to 10 MB
           - Gemini models: Up to 20 MB
           
        4. Recommendations:
           - Use PDF.from_file() for local files
           - Use PDF.from_url(url, download=True) to ensure content is properly encoded
           - Check compatibility with PDF.supports_pdf_input("provider/model-name")
        """


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def encode_pdf(pdf: Union[str, bytes, dict], download_pdf: bool = False) -> str:
    """
    Encode a PDF file to a base64 data URI.

    Args:
        pdf: The PDF to encode. Can be a file path, URL, or data URI.
        download_pdf: Whether to download PDFs from URLs.

    Returns:
        str: The data URI of the PDF file or the URL if download_pdf is False.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(pdf, dict) and "url" in pdf:
        return pdf["url"]
    elif isinstance(pdf, str):
        if pdf.startswith("data:"):
            # Already a data URI
            return pdf
        elif os.path.isfile(pdf):
            # File path
            return _encode_pdf_from_file(pdf)
        elif is_url(pdf):
            # URL
            if download_pdf:
                return _encode_pdf_from_url(pdf)
            else:
                # Return the URL as is
                return pdf
        else:
            # Unsupported string format
            raise ValueError(f"Unsupported file string: {pdf}")
    elif isinstance(pdf, bytes):
        # Raw bytes
        encoded_data = base64.b64encode(pdf).decode("utf-8")
        return f"data:application/pdf;base64,{encoded_data}"
    elif isinstance(pdf, PDF):
        return pdf.url
    else:
        raise ValueError(f"Unsupported PDF type: {type(pdf)}")


def _encode_pdf_from_file(file_path: str) -> str:
    """Encode a PDF file from a file path to a base64 data URI."""
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Verify it's a PDF file
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type != "application/pdf":
        # Still allow it but warning may be appropriate
        mime_type = "application/pdf"

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_pdf_from_url(pdf_url: str) -> str:
    """Encode a PDF file from a URL to a base64 data URI."""
    response = requests.get(pdf_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type and "pdf" in content_type.lower():
        mime_type = content_type
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(pdf_url)
        if mime_type is None or "pdf" not in mime_type.lower():
            mime_type = "application/pdf"

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def is_pdf(obj) -> bool:
    """Check if the object is a PDF or a valid PDF file reference."""
    if isinstance(obj, str):
        if obj.startswith("data:") and "application/pdf" in obj:
            return True
        elif os.path.isfile(obj) and obj.lower().endswith(".pdf"):
            return True
        elif is_url(obj) and obj.lower().endswith(".pdf"):
            return True
    return False