import base64
import json

import pytest

import dspy
from dspy.adapters.json_adapter import JSONAdapter


@pytest.mark.parametrize(
    ("annotation", "payload"),
    [
        (dspy.Image, lambda path: {"url": path}),
        (dspy.Audio, lambda path: path),
        (dspy.File, lambda path: path),
    ],
)
def test_lm_output_cannot_read_local_resources(tmp_path, monkeypatch, annotation, payload):
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"TOP-SECRET")

    class OutputSignature(dspy.Signature):
        resource: annotation = dspy.OutputField()

    def fail_open(*args, **kwargs):
        pytest.fail("LM output parsing attempted to open a local resource")

    monkeypatch.setattr("builtins.open", fail_open)
    completion = '{"resource": ' + json.dumps(payload(str(secret))) + "}"

    with pytest.raises((ValueError, TypeError)):
        JSONAdapter().parse(OutputSignature, completion)


def test_audio_lm_output_cannot_download_url(monkeypatch):
    class OutputSignature(dspy.Signature):
        audio: dspy.Audio = dspy.OutputField()

    def fail_request(*args, **kwargs):
        pytest.fail("LM output parsing attempted to download a remote resource")

    monkeypatch.setattr("dspy.adapters.types.audio.requests.get", fail_request)

    with pytest.raises(ValueError, match=r"Audio\.from_url"):
        JSONAdapter().parse(OutputSignature, '{"audio": "https://example.com/secret.wav"}')


def test_image_url_constructor_does_not_download(monkeypatch):
    def fail_request(*args, **kwargs):
        pytest.fail("Image construction attempted to download a remote resource")

    monkeypatch.setattr("dspy.adapters.types.image.requests.get", fail_request)

    image = dspy.Image("https://example.com/image.png")

    assert image.url == "https://example.com/image.png"


def test_explicit_local_resource_factories(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image bytes")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio bytes")
    file_path = tmp_path / "document.txt"
    file_path.write_bytes(b"file bytes")

    assert base64.b64decode(dspy.Image.from_file(str(image_path)).url.split(",", 1)[1]) == b"image bytes"
    assert base64.b64decode(dspy.Audio.from_file(str(audio_path)).data) == b"audio bytes"
    assert base64.b64decode(dspy.File.from_path(str(file_path)).file_data.split(",", 1)[1]) == b"file bytes"


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
    [("clip.wav", r"Audio\.from_file"), ("https://example.com/a.wav", r"Audio\.from_url")],
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
