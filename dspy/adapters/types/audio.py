import base64
import io
import mimetypes
import os
from typing import Any, Union

import pydantic
import requests

from dspy.adapters.types._url_safety import DEFAULT_DOWNLOAD_TIMEOUT, assert_public_url
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

    def __init__(
        self,
        source: Any = None,
        *,
        download: bool = False,
        verify: bool = True,
        timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
        **data: Any,
    ):
        """Construct an Audio from a file path, data URI, bytes, numpy array,
        dict, or (with ``download=True``) a URL.

        Downloading from a URL is opt-in for security. A bare ``Audio(url)`` or
        a string coerced into an ``Audio``-typed field will NOT fetch; it raises
        and points you at ``Audio.from_url(url)``. Pass ``download=True`` to opt
        in. ``verify`` toggles SSL verification and ``timeout`` bounds the
        request; both flow through to the download.
        """
        if source is not None and "data" not in data:
            encoded = encode_audio(source, download=download, verify=verify, timeout=timeout)
            data = {**data, **encoded}
        super().__init__(**data)

    def format(self) -> list[dict[str, Any]]:
        try:
            data = self.data
        except Exception as e:
            raise ValueError(f"Failed to format audio for DSPy: {e}")
        return [{
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": self.audio_format
            }
        }]


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
    def from_url(cls, url: str, *, verify: bool = True, timeout: float = DEFAULT_DOWNLOAD_TIMEOUT) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.

        The destination is validated against the SSRF guard (rejects loopback,
        private, and link-local addresses), the request is bounded by ``timeout``
        seconds, and ``verify`` toggles SSL certificate verification.
        """
        assert_public_url(url)
        response = requests.get(url, verify=verify, timeout=timeout)
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
        return cls(data=encoded_data, audio_format=format)

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, audio_format='{self.audio_format}')"

def encode_audio(
    audio: Union[str, bytes, dict, "Audio", Any],
    sampling_rate: int = 16000,
    format: str = "wav",
    *,
    download: bool = False,
    verify: bool = True,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> dict:
    """
    Encode audio to a dict with 'data' and 'audio_format'.

    Accepts: local file path, URL, data URI, dict, Audio instance, numpy array, or bytes (with known format).

    Downloading from a URL is opt-in (``download=True``) for security: this
    function runs from ``Audio``'s ``before`` validator, so any string coerced
    into an ``Audio`` field would otherwise trigger an implicit outbound request
    (an SSRF and hanging-host DoS surface). With ``download=False`` (the default)
    an ``http(s)`` string raises and points at ``Audio.from_url(url)``.
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
        if not download:
            raise ValueError(
                "Audio will not download from a URL by default, because this "
                "runs from a validator and any string coerced into an Audio "
                "field would trigger an implicit outbound request (SSRF / "
                "hanging-host DoS). To download intentionally, call "
                "Audio.from_url(url), or opt in with Audio(url, download=True)."
            )
        a = Audio.from_url(audio, verify=verify, timeout=timeout)
        return {"data": a.data, "audio_format": a.audio_format}
    elif SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, bytes):
        encoded = base64.b64encode(audio).decode("utf-8")
        return {"data": encoded, "audio_format": format}
    else:
        raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
