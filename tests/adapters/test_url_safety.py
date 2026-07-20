"""Tests for the URL-download safety controls on Audio and Image.

All tests are offline: the SSRF guard and the opt-in gate both raise before any
network request is made, so no real download happens.
"""
from unittest import mock

import pytest
import requests

from dspy.adapters.types._url_safety import assert_public_url, safe_get
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
    "http://100.64.0.1/x",                         # CGNAT / RFC 6598 (mislabeled global on <3.11)
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


# ---------------------------------------------------------------------------
# Redirect bypass: a public URL that redirects to a private one must be refused
# ---------------------------------------------------------------------------

def _redirect_resp(location, url="http://8.8.8.8/start"):
    r = requests.Response()
    r.status_code = 302
    r.url = url
    r.headers["Location"] = location
    return r


def _ok_resp(url="http://8.8.8.8/x"):
    r = requests.Response()
    r.status_code = 200
    r.url = url
    r._content = b"ok"
    return r


@mock.patch("dspy.adapters.types._url_safety.requests.get")
def test_safe_get_blocks_redirect_to_private(mock_get):
    # First hop is a public literal IP (passes the guard); it 302-redirects to
    # the metadata endpoint, which must be caught before it is followed.
    mock_get.return_value = _redirect_resp("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError):
        safe_get("http://8.8.8.8/start")


@mock.patch("dspy.adapters.types._url_safety.requests.get")
def test_safe_get_follows_public_redirect(mock_get):
    # A redirect to another public address is allowed.
    mock_get.side_effect = [_redirect_resp("http://1.1.1.1/next"), _ok_resp("http://1.1.1.1/next")]
    resp = safe_get("http://8.8.8.8/start")
    assert resp.status_code == 200


@mock.patch("dspy.adapters.types._url_safety.requests.get")
def test_safe_get_caps_redirects(mock_get):
    # An endless public redirect loop terminates instead of hanging.
    mock_get.return_value = _redirect_resp("http://8.8.8.8/loop")
    with pytest.raises(ValueError, match="Too many redirects"):
        safe_get("http://8.8.8.8/start", max_redirects=3)


def test_dns_resolution_is_bounded():
    # assert_public_url must not hang on a dead resolver; a tiny timeout raises.
    with mock.patch("dspy.adapters.types._url_safety.socket.getaddrinfo",
                    side_effect=lambda *a, **k: __import__("time").sleep(5)):
        with pytest.raises(ValueError, match="timed out"):
            assert_public_url("http://example.com/x", dns_timeout=0.2)
