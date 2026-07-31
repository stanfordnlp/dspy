import base64

import pytest
from pydantic import TypeAdapter, ValidationError

import dspy


def _forbid_host_io(monkeypatch):
    """Make any filesystem read or outbound request fail the test."""

    def fail_open(*args, **kwargs):
        pytest.fail("resource handling attempted a filesystem read")

    def fail_request(*args, **kwargs):
        pytest.fail("resource handling attempted a network request")

    monkeypatch.setattr("builtins.open", fail_open)
    monkeypatch.setattr("dspy.adapters.types.image.requests.get", fail_request)
    monkeypatch.setattr("dspy.adapters.types.audio.requests.get", fail_request)


# Locator-shaped inputs an attacker could supply. Neither construction nor the validation/parse
# path may trigger host I/O, whether the call raises (paths, non-data-URI strings) or is retained
# as a reference (Image URLs).
_LOCATOR_INPUTS = [
    (dspy.Image, {"url": "/etc/passwd"}),
    (dspy.Image, "/etc/passwd"),
    (dspy.Image, {"url": "https://evil.example/secret.png"}),
    (dspy.Audio, "/etc/passwd"),
    (dspy.Audio, "https://evil.example/secret.wav"),
    (dspy.File, "/etc/passwd"),
    (dspy.File, "https://evil.example/secret.bin"),
]


@pytest.mark.parametrize(("annotation", "value"), _LOCATOR_INPUTS)
def test_construction_performs_no_host_io(monkeypatch, annotation, value):
    # The constructor is reachable from untrusted application input (e.g. dspy.Image(user_string));
    # it must never dereference a locator.
    _forbid_host_io(monkeypatch)
    try:
        annotation(value)
    except (ValueError, TypeError):
        pass


@pytest.mark.parametrize(("annotation", "value"), _LOCATOR_INPUTS)
def test_validation_performs_no_host_io(monkeypatch, annotation, value):
    # validate_python is the path adapters use to coerce untrusted LM output (and pydantic uses for
    # nested models / deserialization); it must never dereference a locator either.
    _forbid_host_io(monkeypatch)
    try:
        TypeAdapter(annotation).validate_python(value)
    except (ValueError, TypeError, ValidationError):
        pass


def test_image_url_constructor_does_not_download(monkeypatch):
    def fail_request(*args, **kwargs):
        pytest.fail("Image construction attempted to download a remote resource")

    monkeypatch.setattr("dspy.adapters.types.image.requests.get", fail_request)

    image = dspy.Image("https://example.com/image.png")

    assert image.url == "https://example.com/image.png"


def test_image_validation_rejects_download_without_host_io(monkeypatch):
    _forbid_host_io(monkeypatch)

    with pytest.raises((TypeError, ValidationError), match="only valid with a positional image source"):
        TypeAdapter(dspy.Image).validate_python({"url": "http://169.254.169.254/latest/meta-data", "download": True})


def test_image_validation_rejects_download_with_source_keyword_without_host_io(monkeypatch):
    _forbid_host_io(monkeypatch)

    with pytest.raises((TypeError, ValidationError), match="only valid with a positional image source"):
        TypeAdapter(dspy.Image).validate_python({"source": "http://169.254.169.254/latest/meta-data", "download": True})


def test_image_positional_download_compatibility_shim(monkeypatch):
    class Response:
        content = b"remote bytes"
        headers = {"Content-Type": "image/png"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("dspy.adapters.types.image.requests.get", lambda *args, **kwargs: Response())

    with pytest.warns(DeprecationWarning, match="download.*deprecated"):
        image = dspy.Image("https://example.com/image.png", download=True)

    assert base64.b64decode(image.url.split(",", 1)[1]) == b"remote bytes"


def test_explicit_local_resource_factories(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image bytes")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio bytes")
    file_path = tmp_path / "document.txt"
    file_path.write_bytes(b"file bytes")

    assert base64.b64decode(dspy.Image.from_path(str(image_path)).url.split(",", 1)[1]) == b"image bytes"
    assert base64.b64decode(dspy.Audio.from_path(str(audio_path)).data) == b"audio bytes"
    assert base64.b64decode(dspy.File.from_path(str(file_path)).file_data.split(",", 1)[1]) == b"file bytes"


def test_audio_from_file_is_deprecated_alias(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio bytes")

    with pytest.warns(DeprecationWarning, match="Audio.from_file is deprecated"):
        aliased = dspy.Audio.from_file(str(audio_path))

    assert base64.b64decode(aliased.data) == b"audio bytes"


@pytest.mark.parametrize(
    ("factory", "mime_type"), [(dspy.Image.from_url, "image/png"), (dspy.Audio.from_url, "audio/wav")]
)
def test_explicit_remote_resource_factories(monkeypatch, factory, mime_type):
    class Response:
        def __init__(self):
            self.content = b"remote bytes"
            self.headers = {"Content-Type": mime_type}

        def raise_for_status(self):
            return None

    module = "image" if factory == dspy.Image.from_url else "audio"
    monkeypatch.setattr(f"dspy.adapters.types.{module}.requests.get", lambda *args, **kwargs: Response())

    resource = factory(f"https://example.com/resource.{mime_type.split('/', 1)[1]}")
    encoded = resource.url.split(",", 1)[1] if isinstance(resource, dspy.Image) else resource.data
    assert base64.b64decode(encoded) == b"remote bytes"


def test_in_memory_resource_construction():
    image = dspy.Image(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    )
    audio = dspy.Audio(b"audio bytes", audio_format="wav")
    file = dspy.File(b"file bytes")

    assert image.url.startswith("data:image/")
    assert base64.b64decode(audio.data) == b"audio bytes"
    assert file.file_data.startswith("data:application/octet-stream;base64,")


@pytest.mark.parametrize(
    ("source", "match"),
    [("clip.wav", r"Audio\.from_path"), ("https://example.com/a.wav", r"Audio\.from_url")],
)
def test_audio_positional_string_must_be_data_uri_even_with_format(source, match):
    with pytest.raises(ValueError, match=match):
        dspy.Audio(source, audio_format="wav")


@pytest.mark.parametrize(
    "source",
    [
        "data:audio/wav;base64,AA==",
        {"data": "AA==", "audio_format": "wav"},
        dspy.Audio(data="AA==", audio_format="wav"),
    ],
)
def test_audio_rejects_audio_format_for_inputs_that_carry_one(source):
    with pytest.raises(TypeError, match="already carries its format"):
        dspy.Audio(source, audio_format="mp3")


@pytest.mark.parametrize("source", [b"audio bytes", "data:audio/wav;base64,AA=="])
def test_audio_rejects_sampling_rate_for_non_array_inputs(source):
    with pytest.raises(TypeError, match="sampling_rate"):
        dspy.Audio(source, sampling_rate=44100)


def test_audio_from_url_passes_verify(monkeypatch):
    captured = {}

    class Response:
        def __init__(self):
            self.content = b"remote bytes"
            self.headers = {"Content-Type": "audio/wav"}

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("dspy.adapters.types.audio.requests.get", fake_get)

    dspy.Audio.from_url("https://example.com/a.wav", verify=False)

    assert captured["verify"] is False
