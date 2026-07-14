"""Tests for the URL-download safety controls on Audio and Image.

All tests are offline: the SSRF guard and the opt-in gate both raise before any
network request is made, so no real download happens.
"""
import pytest

from dspy.adapters.types._url_safety import assert_public_url
from dspy.adapters.types.audio import Audio, encode_audio
from dspy.adapters.types.image import _encode_image_from_url


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x",                          # loopback
    "http://localhost/x",                          # loopback name
    "http://169.254.169.254/latest/meta-data/",    # cloud metadata endpoint
    "http://10.0.0.5/x",                           # private
    "http://192.168.1.1/x",                        # private
    "http://172.16.0.1/x",                         # private
    "http://[::1]/x",                              # ipv6 loopback
    "http://0.0.0.0/x",                            # unspecified
])
def test_guard_rejects_non_public_addresses(url):
    with pytest.raises(ValueError):
        assert_public_url(url)


@pytest.mark.parametrize("url", [
    "http://8.8.8.8/x",       # public IP (validated, not fetched)
    "https://1.1.1.1/x",
])
def test_guard_allows_public_addresses(url):
    assert_public_url(url) is None  # does not raise


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "gopher://x"])
def test_guard_rejects_non_http_schemes(url):
    with pytest.raises(ValueError, match="scheme"):
        assert_public_url(url)


# ---------------------------------------------------------------------------
# Audio: download is opt-in (secure by default)
# ---------------------------------------------------------------------------

def test_encode_audio_url_requires_optin():
    with pytest.raises(ValueError, match="will not download"):
        encode_audio("http://example.com/a.wav")  # download defaults to False


def test_bare_audio_url_does_not_download():
    # A string coerced/constructed into Audio must not trigger a fetch.
    with pytest.raises(ValueError, match="will not download"):
        Audio("http://example.com/a.wav")


def test_audio_field_coercion_does_not_download():
    # The model_validator path (untrusted coercion) must also refuse.
    with pytest.raises(ValueError, match="will not download"):
        Audio.model_validate("http://example.com/a.wav")


def test_audio_from_url_applies_ssrf_guard():
    # Even the explicit download path validates the destination.
    with pytest.raises(ValueError):
        Audio.from_url("http://127.0.0.1/a.wav")


def test_audio_non_url_paths_still_work():
    # Data URI and explicit data/format construction never touch the network.
    a = Audio("data:audio/wav;base64,QUJD")
    assert a.data == "QUJD" and a.audio_format == "wav"
    b = Audio(data="QUJD", audio_format="mp3")
    assert b.data == "QUJD" and b.audio_format == "mp3"


# ---------------------------------------------------------------------------
# Image: opt-in already existed; guard is the new part
# ---------------------------------------------------------------------------

def test_image_from_url_applies_ssrf_guard():
    with pytest.raises(ValueError):
        _encode_image_from_url("http://169.254.169.254/latest/meta-data/")
