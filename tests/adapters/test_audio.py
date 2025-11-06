import pytest

from dspy.adapters.types.audio import _normalize_audio_format


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
