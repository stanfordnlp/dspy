import base64
import os
import tempfile

import pytest

from dspy.adapters.types.video import Video, encode_video


@pytest.fixture
def sample_video_bytes():
    """Return some dummy bytes to simulate video content."""
    return b"\x00\x00\x00\x1cftypisom" + b"\x00" * 50


@pytest.fixture
def sample_video_file(sample_video_bytes):
    """Create a temporary .mp4 file with dummy content."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(sample_video_bytes)
        path = f.name
    yield path
    os.unlink(path)


class TestVideoFromDict:
    def test_create_from_dict(self):
        data = base64.b64encode(b"fake video data").decode("utf-8")
        video = Video(data=data, video_format="mp4")
        assert video.data == data
        assert video.video_format == "mp4"

    def test_passthrough_dict_in_validator(self):
        data = base64.b64encode(b"fake video data").decode("utf-8")
        result = encode_video({"data": data, "video_format": "mp4"})
        assert result["data"] == data
        assert result["video_format"] == "mp4"


class TestVideoFromFile:
    def test_from_file(self, sample_video_file):
        video = Video.from_file(sample_video_file)
        assert video.video_format == "mp4"
        assert len(video.data) > 0
        # Verify it's valid base64
        base64.b64decode(video.data)

    def test_from_file_not_found(self):
        with pytest.raises(ValueError, match="File not found"):
            Video.from_file("/nonexistent/path/video.mp4")

    def test_encode_video_from_file_path(self, sample_video_file):
        result = encode_video(sample_video_file)
        assert "data" in result
        assert result["video_format"] == "mp4"


class TestVideoFromDataURI:
    def test_encode_video_from_data_uri(self):
        b64 = base64.b64encode(b"fake video").decode("utf-8")
        data_uri = f"data:video/mp4;base64,{b64}"
        result = encode_video(data_uri)
        assert result["data"] == b64
        assert result["video_format"] == "mp4"

    def test_encode_video_from_webm_data_uri(self):
        b64 = base64.b64encode(b"fake video").decode("utf-8")
        data_uri = f"data:video/webm;base64,{b64}"
        result = encode_video(data_uri)
        assert result["data"] == b64
        assert result["video_format"] == "webm"


class TestVideoFromBytes:
    def test_encode_video_from_bytes(self, sample_video_bytes):
        result = encode_video(sample_video_bytes)
        assert result["video_format"] == "mp4"
        decoded = base64.b64decode(result["data"])
        assert decoded == sample_video_bytes


class TestVideoFormat:
    def test_format_output(self):
        data = base64.b64encode(b"test").decode("utf-8")
        video = Video(data=data, video_format="mp4")
        formatted = video.format()
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "input_video"
        assert formatted[0]["input_video"]["data"] == data
        assert formatted[0]["input_video"]["format"] == "mp4"


class TestVideoSerialization:
    def test_str_roundtrip(self):
        data = base64.b64encode(b"test video").decode("utf-8")
        video = Video(data=data, video_format="mp4")
        s = str(video)
        assert "input_video" in s

    def test_repr(self):
        data = base64.b64encode(b"test video").decode("utf-8")
        video = Video(data=data, video_format="mp4")
        r = repr(video)
        assert "VIDEO_BASE_64_ENCODED" in r
        assert "mp4" in r


class TestEncodeVideoUnsupported:
    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            encode_video(12345)
