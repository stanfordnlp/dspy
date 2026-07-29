import base64
import binascii
import io
import mimetypes
import os
import warnings
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


def _validate_base64_audio_data(data: str) -> str:
    if not data:
        raise ValueError("Audio data must be non-empty base64 or an audio data URI")
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Audio data must be valid base64 or an audio data URI") from error
    return data


def _parse_audio_data_uri(data_uri: str) -> tuple[str, str]:
    header, separator, data = data_uri.partition(",")
    media_type, *parameters = header.removeprefix("data:").split(";")
    media_category, format_separator, audio_format = media_type.partition("/")
    if (
        not separator
        or media_category.lower() != "audio"
        or not format_separator
        or not audio_format
        or "base64" not in {parameter.lower() for parameter in parameters}
    ):
        raise ValueError("Audio data URI must have the form data:audio/<format>;base64,<base64_data>")
    return _validate_base64_audio_data(data), _normalize_audio_format(audio_format.lower())


class Audio(Type):
    """An audio input type for DSPy.

    Raw base64 is passed as ``Audio(data=..., audio_format=...)``. An audio data URI may instead be
    passed as ``Audio(data=...)``; its format is inferred and its payload is normalized to raw
    base64. Construction and adapter parsing never access the filesystem or network; use
    :meth:`from_path` to read a local file, :meth:`from_url` to download a remote resource,
    :meth:`from_bytes` for audio bytes, or :meth:`from_array` for array data.
    """

    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @pydantic.field_validator("audio_format")
    @classmethod
    def validate_audio_format(cls, value: str) -> str:
        if not value:
            raise ValueError("audio_format must be a non-empty string")
        return _normalize_audio_format(value)

    @pydantic.field_validator("data")
    @classmethod
    def validate_data(cls, value: str) -> str:
        return _validate_base64_audio_data(value)

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
        if isinstance(values, dict):
            normalized_values = dict(values)
            data = normalized_values.get("data")
            if isinstance(data, str) and data.startswith("data:"):
                normalized_data, uri_format = _parse_audio_data_uri(data)
                declared_format = normalized_values.get("audio_format")
                if isinstance(declared_format, str):
                    if _normalize_audio_format(declared_format).lower() != uri_format:
                        raise ValueError(
                            f"audio_format {declared_format!r} does not match data URI format {uri_format!r}"
                        )
                elif declared_format is not None:
                    normalized_values["data"] = normalized_data
                    return normalized_values
                normalized_values["data"] = normalized_data
                normalized_values["audio_format"] = uri_format
            return normalized_values
        raise ValueError("Audio must be constructed from data and audio_format fields or an explicit from_* factory")

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        timeout: float | tuple[float, float] = 30.0,
        verify: bool | str = True,
    ) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.

        This is a synchronous, blocking network operation.

        Security: this performs an explicit, caller-initiated fetch and applies no
        SSRF protection beyond requiring an HTTP(S) scheme. Like ``requests.get``, it
        will reach private, loopback, or cloud-metadata hosts and follow redirects to
        them. When ``url`` is derived from untrusted input, the caller is responsible
        for validating the host against an allowlist before calling this method.

        Args:
            url: The URL of the audio to download.
            timeout: Connection and read timeout in seconds, either as one value or a tuple.
            verify: Whether to verify SSL certificates. Set to False for self-signed certs.
        """
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError(f"Audio.from_url requires an HTTP(S) URL, received: {url}")
        response = requests.get(url, timeout=timeout, verify=verify)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "audio/wav").split(";", 1)[0].strip().lower()
        if not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
        audio_format = _normalize_audio_format(mime_type.split("/", 1)[1])
        return cls.from_bytes(response.content, audio_format=audio_format)

    @classmethod
    def from_path(cls, path: str | os.PathLike[str]) -> "Audio":
        """
        Read local audio file and encode it as base64.
        """
        with open(path, "rb") as file:
            file_data = file.read()

        path_string = os.fspath(path)
        mime_type, _ = mimetypes.guess_type(path_string)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")

        audio_format = _normalize_audio_format(mime_type.split("/", 1)[1])
        return cls.from_bytes(file_data, audio_format=audio_format)

    @classmethod
    def from_bytes(cls, audio_bytes: bytes, *, audio_format: str) -> "Audio":
        """Create Audio from raw bytes with an explicit format."""
        if not audio_format:
            raise ValueError("audio_format must be a non-empty string")
        encoded_data = base64.b64encode(audio_bytes).decode("utf-8")
        return cls(data=encoded_data, audio_format=_normalize_audio_format(audio_format))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "Audio":
        """Deprecated alias for :meth:`from_path`."""
        warnings.warn(
            "Audio.from_file is deprecated and will be removed in 3.4; use Audio.from_path instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_path(path)

    @classmethod
    def from_array(cls, array: Any, sampling_rate: int, *, audio_format: str = "wav") -> "Audio":
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
            format=audio_format.upper(),
            subtype="PCM_16",
        )
        return cls.from_bytes(byte_buffer.getvalue(), audio_format=audio_format)

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, audio_format='{self.audio_format}')"


def encode_audio(
    audio: Union[str, bytes, dict, "Audio", Any],
    *,
    sampling_rate: int | None = None,
    audio_format: str | None = None,
) -> dict:
    """
    Encode audio to a dict with 'data' and 'audio_format'.

    Accepts in-memory data: data URI, dict, Audio instance, numpy array, or bytes.
    """
    if isinstance(audio, dict):
        encoded_audio = Audio(**audio)
        return {"data": encoded_audio.data, "audio_format": encoded_audio.audio_format}
    elif isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    elif isinstance(audio, str) and audio.startswith("data:"):
        encoded_audio = Audio(data=audio)
        return {"data": encoded_audio.data, "audio_format": encoded_audio.audio_format}
    elif isinstance(audio, str):
        raise ValueError(
            "String audio inputs must be data URIs. "
            "Load local files with Audio.from_path() and remote resources with Audio.from_url()."
        )
    elif SF_AVAILABLE and hasattr(audio, "shape"):
        if sampling_rate is None:
            raise ValueError("sampling_rate is required for array audio inputs")
        array_format = "wav" if audio_format is None else audio_format
        a = Audio.from_array(audio, sampling_rate=sampling_rate, audio_format=array_format)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, bytes):
        if audio_format is None:
            raise ValueError("audio_format is required for byte audio inputs")
        a = Audio.from_bytes(audio, audio_format=audio_format)
        return {"data": a.data, "audio_format": a.audio_format}
    else:
        raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
