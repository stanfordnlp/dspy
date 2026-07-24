import base64
import io
import mimetypes
import os
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type

try:
    import soundfile as sf

    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


def _normalize_audio_format(audio_format: str) -> str:
    """Removes 'x-' prefixes from audio format strings."""
    return audio_format.removeprefix("x-")


class Audio(Type):
    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def __init__(self, *args, **data):
        if len(args) > 1:
            raise TypeError(f"Audio expected at most 1 positional argument, received {len(args)}")
        if args:
            if "data" in data:
                raise TypeError("Audio received data as both a positional and keyword argument")
            value = args[0]
            sampling_rate = data.pop("sampling_rate", None)
            audio_format = data.pop("audio_format", None)
            if audio_format is not None and _carries_own_format(value):
                raise TypeError(
                    "Audio received audio_format alongside an input that already carries its format; provide only one"
                )
            if sampling_rate is not None and not hasattr(value, "shape"):
                raise TypeError("Audio received sampling_rate for a non-array input; it only applies to array data")
            normalized = encode_audio(value, sampling_rate=sampling_rate or 16000, format=audio_format or "wav")
            normalized.update(data)
            data = normalized
        super().__init__(**data)

    def format(self) -> list[dict[str, Any]]:
        try:
            data = self.data
        except Exception as e:
            raise ValueError(f"Failed to format audio for DSPy: {e}")
        return [{"type": "input_audio", "input_audio": {"data": data, "format": self.audio_format}}]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        """
        Validate input for Audio, expecting 'data' and 'audio_format' keys in dictionary.
        """
        if isinstance(values, cls):
            return {"data": values.data, "audio_format": values.audio_format}
        return encode_audio(values)

    @classmethod
    def from_url(cls, url: str, verify: bool = True) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.

        Args:
            url: The URL of the audio to download.
            verify: Whether to verify SSL certificates. Set to False for self-signed certs.
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError(f"Audio.from_url requires an HTTP(S) URL, received: {url}")
        response = requests.get(url, verify=verify)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "audio/wav")
        if not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
        audio_format = mime_type.split("/")[1]

        audio_format = _normalize_audio_format(audio_format)

        encoded_data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_file(cls, file_path: str) -> "Audio":
        """
        Read local audio file and encode it as base64.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")

        with open(file_path, "rb") as file:
            file_data = file.read()

        audio_format = mime_type.split("/")[1]

        audio_format = _normalize_audio_format(audio_format)

        encoded_data = base64.b64encode(file_data).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_array(cls, array: Any, sampling_rate: int, format: str = "wav") -> "Audio":
        """
        Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.
        """
        if not SF_AVAILABLE:
            raise ImportError("soundfile is required to process audio arrays.")

        byte_buffer = io.BytesIO()
        sf.write(
            byte_buffer,
            array,
            sampling_rate,
            format=format.upper(),
            subtype="PCM_16",
        )
        encoded_data = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
        return cls(data=encoded_data, audio_format=format)

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, audio_format='{self.audio_format}')"


def _carries_own_format(value: Any) -> bool:
    """Whether an audio input already carries its own format (making audio_format redundant)."""
    if isinstance(value, Audio):
        return True
    if isinstance(value, dict) and "audio_format" in value:
        return True
    return isinstance(value, str) and value.startswith("data:audio/")


def encode_audio(audio: Union[str, bytes, dict, "Audio", Any], sampling_rate: int = 16000, format: str = "wav") -> dict:
    """
    Encode audio to a dict with 'data' and 'audio_format'.

    Accepts in-memory data: data URI, dict, Audio instance, numpy array, or bytes.
    """
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return audio
    elif isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    elif isinstance(audio, str) and audio.startswith("data:audio/"):
        header, separator, b64data = audio.partition(",")
        mime = header.removeprefix("data:").split(";", 1)[0]
        _, format_separator, audio_format = mime.partition("/")
        if not separator or not format_separator or not audio_format:
            raise ValueError("Malformed audio data URI")
        return {"data": b64data, "audio_format": _normalize_audio_format(audio_format)}
    elif isinstance(audio, str):
        raise ValueError(
            "String audio inputs must be data URIs. "
            "Load local files with Audio.from_file() and remote resources with Audio.from_url()."
        )
    elif SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, bytes):
        encoded = base64.b64encode(audio).decode("utf-8")
        return {"data": encoded, "audio_format": format}
    else:
        raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
