import base64
import io
import mimetypes
import os
from typing import Any
from ..types.custom_type import CustomType

import pydantic
import requests

try:
    import soundfile as sf

    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


class Audio(pydantic.BaseModel, CustomType):
    data: str
    format: str

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }

    def _format(self) -> list[dict[str, Any]]:
        try:
            data = self.data
            if isinstance(data, str) and "<DSPY_AUDIO_START>" in data and "<DSPY_AUDIO_END>" in data:
                data = data.split("<DSPY_AUDIO_START>", 1)[-1].split("<DSPY_AUDIO_END>", 1)[0]
        except Exception as e:
            raise ValueError(f"Failed to format audio for DSPy: {e}")
        return [{
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": self.format
            }
        }]


    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        """
        Validate input for Audio, expecting 'data' and 'format' keys in dictionary.
        """
        if isinstance(values, dict) and {"data", "format"} <= set(values.keys()):
            return values
        raise TypeError("Expected dict with keys 'data' and 'format'.")

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.
        """
        response = requests.get(url)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "audio/wav")
        if not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
        audio_format = mime_type.split("/")[1]
        encoded_data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=encoded_data, format=audio_format)

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
        encoded_data = base64.b64encode(file_data).decode("utf-8")
        return cls(data=encoded_data, format=audio_format)

    @classmethod
    def from_array(
        cls, array: Any, sampling_rate: int, format: str = "wav"
    ) -> "Audio":
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
        return cls(data=encoded_data, format=format)

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return f"<DSPY_AUDIO_START>{self.data}<DSPY_AUDIO_END>"

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, format='{self.format}')"
