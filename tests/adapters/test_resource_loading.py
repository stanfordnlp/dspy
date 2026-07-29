import base64
import inspect

import pytest
from pydantic import TypeAdapter, ValidationError

import dspy


@pytest.mark.parametrize(
    ("resource_type", "field_names"),
    [
        (dspy.Image, {"url"}),
        (dspy.Audio, {"data", "audio_format"}),
        (dspy.File, {"file_data", "file_id", "filename"}),
    ],
)
def test_resource_constructor_signatures_are_keyword_only(resource_type, field_names):
    parameters = inspect.signature(resource_type).parameters

    assert set(parameters) == field_names
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())


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
    (dspy.Image, {"url": "https://evil.example/secret.png"}),
    (dspy.Audio, {"data": "/etc/passwd", "audio_format": "wav"}),
    (dspy.Audio, {"data": "https://evil.example/secret.wav", "audio_format": "wav"}),
    (dspy.File, {"file_data": "/etc/passwd"}),
    (dspy.File, {"file_data": "https://evil.example/secret.bin"}),
]

_REJECTED_LOCATOR_INPUTS = _LOCATOR_INPUTS[2:]


@pytest.mark.parametrize(("annotation", "value"), _LOCATOR_INPUTS)
def test_construction_performs_no_host_io(monkeypatch, annotation, value):
    _forbid_host_io(monkeypatch)
    try:
        annotation(**value)
    except (ValueError, TypeError, ValidationError):
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


@pytest.mark.parametrize(("annotation", "value"), _REJECTED_LOCATOR_INPUTS)
def test_resource_construction_rejects_locator_payloads(annotation, value):
    with pytest.raises(ValidationError):
        annotation(**value)


@pytest.mark.parametrize(("annotation", "value"), _REJECTED_LOCATOR_INPUTS)
def test_resource_validation_rejects_locator_payloads(annotation, value):
    with pytest.raises(ValidationError):
        TypeAdapter(annotation).validate_python(value)


def test_audio_data_uri_is_normalized_and_format_is_inferred():
    audio = dspy.Audio(data="data:audio/x-wav;base64,YXVkaW8=")

    assert audio.data == "YXVkaW8="
    assert audio.audio_format == "wav"


def test_audio_data_uri_rejects_mismatched_format():
    with pytest.raises(ValidationError, match="does not match"):
        dspy.Audio(data="data:audio/wav;base64,YXVkaW8=", audio_format="mp3")


@pytest.mark.parametrize(
    "value",
    [
        {"data": "not base64!", "audio_format": "wav"},
        {"data": "data:audio/wav;base64,not base64!"},
        {"data": "data:text/plain;base64,YXVkaW8="},
    ],
)
def test_audio_rejects_malformed_payloads(value):
    with pytest.raises(ValidationError):
        TypeAdapter(dspy.Audio).validate_python(value)


@pytest.mark.parametrize(
    "file_data",
    [
        "data:text/plain,plain text",
        "data:text/plain;base64,not base64!",
        "data:;base64,",
    ],
)
def test_file_rejects_malformed_data_uris(file_data):
    with pytest.raises(ValidationError):
        TypeAdapter(dspy.File).validate_python({"file_data": file_data})


def test_image_validation_rejects_download_without_host_io(monkeypatch):
    _forbid_host_io(monkeypatch)

    with pytest.raises(ValidationError, match="download"):
        TypeAdapter(dspy.Image).validate_python({"url": "https://evil.example/secret.png", "download": True})


def test_image_url_constructor_does_not_download(monkeypatch):
    def fail_request(*args, **kwargs):
        pytest.fail("Image construction attempted to download a remote resource")

    monkeypatch.setattr("dspy.adapters.types.image.requests.get", fail_request)

    image = dspy.Image(url="https://example.com/image.png")

    assert image.url == "https://example.com/image.png"


def test_explicit_local_resource_factories(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image bytes")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio bytes")
    file_path = tmp_path / "document.txt"
    file_path.write_bytes(b"file bytes")

    assert base64.b64decode(dspy.Image.from_path(image_path).url.split(",", 1)[1]) == b"image bytes"
    assert base64.b64decode(dspy.Audio.from_path(audio_path).data) == b"audio bytes"
    assert base64.b64decode(dspy.File.from_path(file_path).file_data.split(",", 1)[1]) == b"file bytes"


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


def test_in_memory_resource_factories():
    image = dspy.Image.from_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    )
    audio = dspy.Audio.from_bytes(b"audio bytes", audio_format="wav")
    file = dspy.File.from_bytes(b"file bytes")

    assert image.url.startswith("data:image/")
    assert base64.b64decode(audio.data) == b"audio bytes"
    assert file.file_data.startswith("data:application/octet-stream;base64,")


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dspy.Image("https://example.com/image.png"),
        lambda: dspy.Audio(b"audio bytes", audio_format="wav"),
        lambda: dspy.File(b"file bytes"),
    ],
)
def test_resource_constructors_reject_positional_sources(constructor):
    with pytest.raises(TypeError):
        constructor()


def test_audio_from_bytes_requires_nonempty_format():
    with pytest.raises(ValueError, match="non-empty"):
        dspy.Audio.from_bytes(b"audio bytes", audio_format="")


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
    assert captured["timeout"] == 30.0


def test_audio_from_url_strips_content_type_parameters(monkeypatch):
    class Response:
        def __init__(self):
            self.content = b"remote bytes"
            self.headers = {"Content-Type": "audio/x-wav; charset=binary"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr("dspy.adapters.types.audio.requests.get", lambda *args, **kwargs: Response())

    assert dspy.Audio.from_url("https://example.com/a.wav").audio_format == "wav"


@pytest.mark.parametrize("factory", [dspy.Image.from_path, dspy.Audio.from_path, dspy.File.from_path])
def test_from_path_preserves_file_not_found_error(factory, tmp_path):
    with pytest.raises(FileNotFoundError):
        factory(tmp_path / "missing.resource")
