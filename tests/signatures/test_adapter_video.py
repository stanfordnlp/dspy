import base64
import os
import tempfile

import pydantic
import pytest

import dspy
from dspy.adapters.types.video import _encode_video_from_file
from dspy.utils.dummies import DummyLM
from tests.signatures.test_adapter_file import count_messages_with_file_pattern


SAMPLE_VIDEO_BYTES = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 50
SAMPLE_VIDEO_B64 = base64.b64encode(SAMPLE_VIDEO_BYTES).decode("utf-8")


@pytest.fixture
def sample_video_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(SAMPLE_VIDEO_BYTES)
        tmp_path = tmp.name
    yield tmp_path
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


@pytest.fixture
def sample_webm_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(SAMPLE_VIDEO_BYTES)
        tmp_path = tmp.name
    yield tmp_path
    try:
        os.unlink(tmp_path)
    except Exception:
        pass


def setup_predictor(signature, expected_output):
    lm = DummyLM([expected_output])
    dspy.settings.configure(lm=lm)
    return dspy.Predict(signature), lm


def test_from_url():
    url = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
    v = dspy.Video.from_url(url)
    assert v.url == url
    assert v.inline_data is None
    assert v.mime_type == "video/mp4"


def test_from_url_custom_mime():
    v = dspy.Video.from_url("https://example.com/video", mime_type="video/webm")
    assert v.mime_type == "video/webm"


def test_from_file(sample_video_file):
    v = dspy.Video.from_file(sample_video_file)
    assert v.inline_data is not None
    assert v.url is None
    assert v.mime_type == "video/mp4"
    assert base64.b64decode(v.inline_data) == SAMPLE_VIDEO_BYTES


def test_from_file_mime_detection(sample_webm_file):
    v = dspy.Video.from_file(sample_webm_file)
    assert v.mime_type == "video/webm"


def test_from_file_custom_mime(sample_video_file):
    v = dspy.Video.from_file(sample_video_file, mime_type="video/webm")
    assert v.mime_type == "video/webm"


def test_from_file_not_found():
    with pytest.raises(ValueError, match="File not found"):
        dspy.Video.from_file("/nonexistent/path/video.mp4")


def test_from_bytes():
    v = dspy.Video.from_bytes(SAMPLE_VIDEO_BYTES)
    assert v.inline_data == SAMPLE_VIDEO_B64
    assert v.mime_type == "video/mp4"


def test_from_bytes_custom_mime():
    v = dspy.Video.from_bytes(SAMPLE_VIDEO_BYTES, mime_type="video/webm")
    assert v.mime_type == "video/webm"


def test_dict_with_url():
    v = dspy.Video(url="https://example.com/video.mp4")
    assert v.url == "https://example.com/video.mp4"
    assert v.inline_data is None


def test_dict_with_inline_data():
    v = dspy.Video(inline_data=SAMPLE_VIDEO_B64, mime_type="video/webm")
    assert v.inline_data == SAMPLE_VIDEO_B64
    assert v.mime_type == "video/webm"


def test_both_url_and_inline_data_raises():
    with pytest.raises(pydantic.ValidationError, match="Exactly one"):
        dspy.Video(url="https://example.com/video.mp4", inline_data=SAMPLE_VIDEO_B64)


def test_neither_url_nor_inline_data_raises():
    with pytest.raises(pydantic.ValidationError, match="url.*inline_data"):
        dspy.Video(mime_type="video/mp4")


def test_invalid_dict_raises():
    with pytest.raises(pydantic.ValidationError):
        dspy.Video(**{"bad_key": "value"})


def test_frozen():
    v = dspy.Video(url="https://example.com/video.mp4")
    with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
        v.url = "https://other.com/video.mp4"


def test_repr_url():
    v = dspy.Video(url="https://example.com/video.mp4")
    assert repr(v) == "Video(url='https://example.com/video.mp4')"


def test_repr_inline():
    v = dspy.Video(inline_data=SAMPLE_VIDEO_B64)
    assert f"base64:{len(SAMPLE_VIDEO_B64)}" in repr(v)


def test_str_contains_custom_type_markers():
    v = dspy.Video(url="https://example.com/video.mp4")
    s = str(v)
    assert "<<CUSTOM-TYPE-START-IDENTIFIER>>" in s
    assert "<<CUSTOM-TYPE-END-IDENTIFIER>>" in s


# --- Integration with DSPy signature ---


def test_video_in_signature():
    predictor, lm = setup_predictor(
        "video: dspy.Video -> description: str",
        {"description": "A video about dogs"},
    )
    result = predictor(video=dspy.Video(url="https://example.com/video.mp4"))
    assert result.description == "A video about dogs"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1


def test_video_inline_in_signature():
    predictor, lm = setup_predictor(
        "video: dspy.Video -> description: str",
        {"description": "A test video"},
    )
    result = predictor(video=dspy.Video.from_bytes(SAMPLE_VIDEO_BYTES))
    assert result.description == "A test video"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1


def test_video_list_in_signature():
    class VideoListSignature(dspy.Signature):
        videos: list[dspy.Video] = dspy.InputField()
        summary: str = dspy.OutputField()

    predictor, lm = setup_predictor(VideoListSignature, {"summary": "Multiple videos"})
    videos = [
        dspy.Video(url="https://example.com/video1.mp4"),
        dspy.Video.from_bytes(SAMPLE_VIDEO_BYTES),
    ]
    result = predictor(videos=videos)
    assert result.summary == "Multiple videos"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2


def test_optional_video_field():
    class OptionalVideoSignature(dspy.Signature):
        video: dspy.Video | None = dspy.InputField()
        output: str = dspy.OutputField()

    predictor, lm = setup_predictor(OptionalVideoSignature, {"output": "No video"})
    result = predictor(video=None)
    assert result.output == "No video"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 0


def test_save_load_video_signature():
    signature = "video: dspy.Video -> description: str"
    video = dspy.Video(url="https://example.com/video.mp4")
    examples = [dspy.Example(video=video, description="A dog video")]

    predictor, lm = setup_predictor(signature, {"description": "A video"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as tmp:
        compiled.save(tmp.name)
        loaded = dspy.Predict(signature)
        loaded.load(tmp.name)

    loaded(video=dspy.Video(url="https://example.com/new.mp4"))
    # 2 = one from the few-shot demo + one from the new input
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2


def test_encode_video_from_file(sample_video_file):
    result = _encode_video_from_file(sample_video_file)
    assert result["inline_data"] == SAMPLE_VIDEO_B64
    assert result["mime_type"] == "video/mp4"


def test_encode_video_from_file_not_found():
    with pytest.raises(ValueError, match="File not found"):
        _encode_video_from_file("/nonexistent/video.mp4")
