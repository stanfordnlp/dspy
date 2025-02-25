import base64
import os
import pydantic
import requests

from typing import Any, Dict, List, Union
from urllib.parse import urlparse


class Audio(pydantic.BaseModel):
    url: str

    model_config = {
        "frozen": True,
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        # Allow the model to accept either a URL string or a dictionary with a single 'url' key
        if isinstance(values, str):
            return {"url": values}
        elif isinstance(values, dict) and set(values.keys()) == {"url"}:
            return values
        elif isinstance(values, cls):
            return values.model_dump()
        else:
            raise TypeError("Expected a string URL or a dictionary with a key 'url'.")

    @classmethod
    def from_url(cls, url: str, download: bool = False):
        return cls(url=encode_audio(url, download))

    @classmethod
    def from_file(cls, file_path: str):
        return cls(url=encode_audio(file_path))

    @classmethod
    def from_bytes(cls, audio_bytes: bytes, format: str = "wav"):
        return cls(url=encode_audio(audio_bytes, format=format))

    @pydantic.model_serializer()
    def serialize_model(self):
        return "<DSPY_AUDIO_START>" + self.url + "<DSPY_AUDIO_END>"

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            audio_type = self.url.split(";")[0].split("/")[-1]
            return f"Audio(url=data:audio/{audio_type};base64,<AUDIO_BASE_64_ENCODED({str(len_base64)})>)"
        return f"Audio(url='{self.url}')"


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def encode_audio(
    audio: Union[str, bytes, dict], download_images: bool = False, format: str = None
) -> str:
    """
    Encode an audio file to a base64 data URI.

    Args:
        audio: The audio to encode. Can be a file path, URL, or data URI.
        download_images: Whether to download audio from URLs.
        format: The audio format when encoding from bytes (e.g., 'wav', 'mp3', etc.)

    Returns:
        str: The data URI of the audio or the URL if download_images is False.

    Raises:
        ValueError: If the audio type is not supported.
    """
    if isinstance(audio, dict) and "url" in audio:
        return audio["url"]
    elif isinstance(audio, str):
        if audio.startswith("data:audio/"):
            # Already a data URI
            return audio
        elif os.path.isfile(audio):
            # File path
            return _encode_audio_from_file(audio)
        elif is_url(audio):
            # URL
            if download_images:
                return _encode_audio_from_url(audio)
            else:
                # Return the URL as is
                return audio
        else:
            print(f"Unsupported audio string: {audio}")
            raise ValueError(f"Unsupported audio string: {audio}")
    elif isinstance(audio, bytes):
        # Raw bytes
        if not format:
            format = "wav"  # Default format
        return _encode_audio_from_bytes(audio, format)
    elif isinstance(audio, Audio):
        return audio.url
    else:
        print(f"Unsupported audio type: {type(audio)}")
        raise ValueError(f"Unsupported audio type: {type(audio)}")


def _encode_audio_from_file(file_path: str) -> str:
    """Encode an audio file from a file path to a base64 data URI."""
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()
    file_extension = _get_file_extension(file_path)
    encoded_audio = base64.b64encode(audio_data).decode("utf-8")
    return f"data:audio/{file_extension};base64,{encoded_audio}"


def _encode_audio_from_url(audio_url: str) -> str:
    """Encode an audio file from a URL to a base64 data URI."""
    response = requests.get(audio_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("audio/"):
        file_extension = content_type.split("/")[-1]
    else:
        file_extension = _get_file_extension(audio_url) or "wav"
    encoded_audio = base64.b64encode(response.content).decode("utf-8")
    return f"data:audio/{file_extension};base64,{encoded_audio}"


def _encode_audio_from_bytes(audio_bytes: bytes, format: str) -> str:
    """Encode audio bytes to a base64 data URI."""
    encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/{format};base64,{encoded_audio}"


def _get_file_extension(path_or_url: str) -> str:
    """Extract the file extension from a file path or URL."""
    extension = os.path.splitext(urlparse(path_or_url).path)[1].lstrip(".").lower()
    return extension or "wav"  # Default to 'wav' if no extension found


def is_audio(obj) -> bool:
    """Check if the object is an audio file or a valid audio reference."""
    if isinstance(obj, str):
        if obj.startswith("data:audio/"):
            return True
        elif os.path.isfile(obj):
            # Could add more specific audio file extension checking here
            return True
        elif is_url(obj):
            return True
    return False
