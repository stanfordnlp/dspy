import base64
import io
import mimetypes
import os
from typing import Any, Union

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
    """A type for representing audio data in DSPy.

    The Audio class provides a standardized way to handle audio inputs for language models
    that support audio processing. Audio data is stored as base64-encoded strings along
    with format metadata.

    Attributes:
        data: Base64-encoded audio data.
        audio_format: The audio format (e.g., "wav", "mp3", "flac").

    Example:
        Basic usage with a local file:

        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-audio-preview"))

        class TranscribeAudio(dspy.Signature):
            audio: dspy.Audio = dspy.InputField()
            transcription: str = dspy.OutputField()

        # Create Audio from a local file
        audio = dspy.Audio.from_file("speech.wav")

        predict = dspy.Predict(TranscribeAudio)
        result = predict(audio=audio)
        ```

    Example:
        Creating Audio from different sources:

        ```python
        import dspy

        # From a URL
        audio = dspy.Audio.from_url("https://example.com/audio.mp3")

        # From a local file path (auto-detected)
        audio = dspy.Audio("path/to/audio.wav")

        # From a numpy array (requires soundfile)
        import numpy as np
        samples = np.random.randn(16000)  # 1 second of audio at 16kHz
        audio = dspy.Audio.from_array(samples, sampling_rate=16000)
        ```
    """

    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def format(self) -> list[dict[str, Any]]:
        """Format the audio data for consumption by language models.

        Returns:
            A list containing the audio block in the format expected by
            audio-enabled language models.

        Raises:
            ValueError: If the audio data cannot be formatted.
        """
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
    def from_url(cls, url: str) -> "Audio":
        """Create an Audio instance by downloading from a URL.

        Downloads the audio file from the specified URL, determines the format
        from the Content-Type header, and encodes the content as base64.

        Args:
            url: The URL of the audio file to download.

        Returns:
            An Audio instance containing the base64-encoded audio data.

        Raises:
            ValueError: If the Content-Type is not an audio MIME type.
            requests.HTTPError: If the HTTP request fails.
        """
        response = requests.get(url)
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
        """Create an Audio instance from a local file.

        Reads the audio file from disk, determines the format from the file
        extension, and encodes the content as base64.

        Args:
            file_path: The path to the local audio file.

        Returns:
            An Audio instance containing the base64-encoded audio data.

        Raises:
            ValueError: If the file does not exist or has an unsupported MIME type.
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
        """Create an Audio instance from a numpy array.

        Converts a numpy-like array of audio samples into an Audio instance
        by encoding it with the specified format and sampling rate.

        Args:
            array: A numpy-like array containing audio samples.
            sampling_rate: The sampling rate in Hz (e.g., 16000 for 16kHz).
            format: The output audio format. Defaults to "wav".

        Returns:
            An Audio instance containing the base64-encoded audio data.

        Raises:
            ImportError: If the soundfile library is not installed.

        Note:
            This method requires the ``soundfile`` library to be installed.
            Install it with ``pip install soundfile``.
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


def encode_audio(audio: Union[str, bytes, dict, "Audio", Any], sampling_rate: int = 16000, format: str = "wav") -> dict:
    """Encode audio from various sources into a standardized dictionary format.

    This function accepts multiple input types and normalizes them into a dictionary
    containing base64-encoded audio data and format information.

    Args:
        audio: The audio input. Supported types include:

            - ``str``: Local file path, HTTP(S) URL, or data URI
            - ``bytes``: Raw audio bytes
            - ``dict``: Dictionary with "data" and "audio_format" keys
            - ``Audio``: An existing Audio instance
            - ``numpy.ndarray``: Audio samples as a numpy array (requires soundfile)

        sampling_rate: The sampling rate in Hz for numpy array inputs. Defaults to 16000.
        format: The audio format for numpy array or bytes inputs. Defaults to "wav".

    Returns:
        A dictionary with "data" (base64-encoded string) and "audio_format" keys.

    Raises:
        ValueError: If the input type is unsupported or the data URI is malformed.
    """
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return audio
    elif isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    elif isinstance(audio, str) and audio.startswith("data:audio/"):
        try:
            header, b64data = audio.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            audio_format = mime.split("/")[1]

            audio_format = _normalize_audio_format(audio_format)

            return {"data": b64data, "audio_format": audio_format}
        except Exception as e:
            raise ValueError(f"Malformed audio data URI: {e}")
    elif isinstance(audio, str) and os.path.isfile(audio):
        a = Audio.from_file(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, str) and audio.startswith("http"):
        a = Audio.from_url(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    elif SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, bytes):
        encoded = base64.b64encode(audio).decode("utf-8")
        return {"data": encoded, "audio_format": format}
    else:
        raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
