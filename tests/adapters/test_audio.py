from unittest import mock

import pytest
import requests

from dspy.adapters.types.audio import Audio, _normalize_audio_format


@pytest.mark.parametrize(
    "input_format, expected_format",
    [
        # Case 1: Standard format (no change)
        ("wav", "wav"),
        ("mp3", "mp3"),

        # Case 2: The 'x-' prefix
        ("x-wav", "wav"),
        ("x-mp3", "mp3"),
        ("x-flac", "flac"),

        # Case 3: The edge case
        ("my-x-format", "my-x-format"),
        ("x-my-format", "my-format"),

        # Case 4: Empty string and edge cases
        ("", ""),
        ("x-", ""),
    ],
)
def test_normalize_audio_format(input_format, expected_format):
    """
    Tests that the _normalize_audio_format helper correctly removes 'x-' prefixes.
    This single test covers the logic for from_url, from_file, and encode_audio.
    """
    assert _normalize_audio_format(input_format) == expected_format


def test_from_url_passes_default_timeout():
    """from_url should pass a timeout to requests.get so it can't hang forever."""
    response = mock.Mock()
    response.headers = {"Content-Type": "audio/wav"}
    response.content = b"fake-audio-bytes"

    with mock.patch("dspy.adapters.types.audio.requests.get", return_value=response) as mock_get:
        Audio.from_url("https://example.com/sound.wav")

    assert mock_get.call_args.kwargs["timeout"] == 30.0


def test_from_url_uses_custom_timeout():
    """A caller-provided timeout should be forwarded to requests.get."""
    response = mock.Mock()
    response.headers = {"Content-Type": "audio/wav"}
    response.content = b"fake-audio-bytes"

    with mock.patch("dspy.adapters.types.audio.requests.get", return_value=response) as mock_get:
        Audio.from_url("https://example.com/sound.wav", timeout=5.0)

    assert mock_get.call_args.kwargs["timeout"] == 5.0


def test_from_url_propagates_timeout_error():
    """If the request times out, the error should surface instead of hanging."""
    with mock.patch("dspy.adapters.types.audio.requests.get", side_effect=requests.exceptions.Timeout):
        with pytest.raises(requests.exceptions.Timeout):
            Audio.from_url("https://example.com/sound.wav")
