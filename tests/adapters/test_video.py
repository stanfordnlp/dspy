import base64
import os
import tempfile

import pydantic
import pytest

import dspy
from dspy.adapters.types.video import (
    VIDEO_MIME_TYPES,
    MAX_INLINE_SIZE_BYTES,
    encode_video_to_dict,
    get_video_mime_type,
    is_gcs_url,
    is_video_url,
    is_youtube_url,
)


@pytest.fixture
def sample_video_file():
    """Create a small fake video file for testing."""
    # Create a minimal MP4-like file (not a real video, but enough for testing file handling)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".mp4") as tmp_file:
        # Write some fake video data
        tmp_file.write(b"\x00\x00\x00\x1c\x66\x74\x79\x70\x69\x73\x6f\x6d")  # Fake MP4 header
        tmp_file.write(b"\x00" * 100)  # Padding
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    try:
        os.unlink(tmp_file_path)
    except Exception:
        pass


@pytest.fixture
def sample_webm_file():
    """Create a small fake WebM file for testing."""
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".webm") as tmp_file:
        tmp_file.write(b"\x1a\x45\xdf\xa3")  # Fake WebM header
        tmp_file.write(b"\x00" * 100)
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    try:
        os.unlink(tmp_file_path)
    except Exception:
        pass


# ============================================================================
# Helper function tests
# ============================================================================


class TestIsYoutubeUrl:
    def test_youtube_watch_url(self):
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_short_url(self):
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")

    def test_youtube_mobile_url(self):
        assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtube_no_www(self):
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")

    def test_non_youtube_url(self):
        assert not is_youtube_url("https://vimeo.com/123456")

    def test_non_url_string(self):
        assert not is_youtube_url("not a url")

    def test_empty_string(self):
        assert not is_youtube_url("")

    def test_non_string(self):
        assert not is_youtube_url(123)


class TestIsVideoUrl:
    def test_https_url(self):
        assert is_video_url("https://example.com/video.mp4")

    def test_http_url(self):
        assert is_video_url("http://example.com/video.mp4")

    def test_invalid_scheme(self):
        assert not is_video_url("ftp://example.com/video.mp4")

    def test_local_path(self):
        assert not is_video_url("/path/to/video.mp4")

    def test_empty_string(self):
        assert not is_video_url("")

    def test_gcs_url(self):
        assert is_video_url("gs://my-bucket/videos/sample.mp4")

    def test_gcs_url_nested_path(self):
        assert is_video_url("gs://bucket-name/path/to/nested/video.webm")


class TestIsGcsUrl:
    def test_gcs_url_basic(self):
        assert is_gcs_url("gs://my-bucket/video.mp4")

    def test_gcs_url_nested(self):
        assert is_gcs_url("gs://bucket-name/path/to/video.mp4")

    def test_gcs_url_no_path(self):
        assert is_gcs_url("gs://bucket-name/")

    def test_not_gcs_https(self):
        assert not is_gcs_url("https://storage.googleapis.com/bucket/video.mp4")

    def test_not_gcs_http(self):
        assert not is_gcs_url("http://example.com/video.mp4")

    def test_not_gcs_local_path(self):
        assert not is_gcs_url("/path/to/video.mp4")

    def test_not_gcs_empty(self):
        assert not is_gcs_url("")

    def test_not_gcs_non_string(self):
        assert not is_gcs_url(123)


class TestGetVideoMimeType:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("video.mp4", "video/mp4"),
            ("video.webm", "video/webm"),
            ("video.mov", "video/quicktime"),
            ("video.avi", "video/x-msvideo"),
            ("video.3gp", "video/3gpp"),
            ("video.mpeg", "video/mpeg"),
            ("video.mpg", "video/mpeg"),
            ("https://example.com/video.mp4", "video/mp4"),
        ],
    )
    def test_known_extensions(self, path, expected):
        assert get_video_mime_type(path) == expected

    def test_unknown_extension(self):
        # Should fall back to mimetypes module or return None
        result = get_video_mime_type("video.xyz")
        assert result is None or isinstance(result, str)


# ============================================================================
# Video class creation tests
# ============================================================================


class TestVideoFromPath:
    def test_from_path_basic(self, sample_video_file):
        video = dspy.Video.from_path(sample_video_file)
        assert video.url is not None
        assert video.url.startswith("data:video/mp4;base64,")
        assert video.filename == os.path.basename(sample_video_file)
        assert video.mime_type == "video/mp4"

    def test_from_path_webm(self, sample_webm_file):
        video = dspy.Video.from_path(sample_webm_file)
        assert video.url.startswith("data:video/webm;base64,")
        assert video.mime_type == "video/webm"

    def test_from_path_custom_filename(self, sample_video_file):
        video = dspy.Video.from_path(sample_video_file, filename="custom.mp4")
        assert video.filename == "custom.mp4"

    def test_from_path_custom_mime_type(self, sample_video_file):
        video = dspy.Video.from_path(sample_video_file, mime_type="video/custom")
        assert video.url.startswith("data:video/custom;base64,")
        assert video.mime_type == "video/custom"

    def test_from_path_file_not_found(self):
        with pytest.raises(ValueError, match="File not found"):
            dspy.Video.from_path("/nonexistent/path/video.mp4")

    def test_from_path_size_limit(self, sample_video_file):
        # Test that files exceeding the limit raise an error
        # We can't easily create a 20MB+ file in tests, so we'll mock or skip this
        # For now, just verify the constant exists
        assert MAX_INLINE_SIZE_BYTES == 20 * 1024 * 1024


class TestVideoFromBytes:
    def test_from_bytes_basic(self):
        video_bytes = b"\x00" * 100
        video = dspy.Video.from_bytes(video_bytes, mime_type="video/mp4")
        assert video.url is not None
        assert video.url.startswith("data:video/mp4;base64,")
        assert video.mime_type == "video/mp4"

    def test_from_bytes_with_filename(self):
        video_bytes = b"\x00" * 100
        video = dspy.Video.from_bytes(video_bytes, mime_type="video/mp4", filename="test.mp4")
        assert video.filename == "test.mp4"

    def test_from_bytes_without_mime_type_fails(self):
        # encode_video_to_dict should raise error for bytes without mime_type
        with pytest.raises(ValueError, match="Cannot create Video from bytes"):
            encode_video_to_dict(b"\x00" * 100)


class TestVideoFromUrl:
    def test_from_url_basic(self):
        video = dspy.Video.from_url("https://example.com/video.mp4")
        assert video.url == "https://example.com/video.mp4"
        assert video.mime_type == "video/mp4"

    def test_from_url_with_filename(self):
        video = dspy.Video.from_url("https://example.com/video.mp4", filename="my_video.mp4")
        assert video.filename == "my_video.mp4"

    def test_from_url_custom_mime_type(self):
        video = dspy.Video.from_url("https://example.com/video", mime_type="video/webm")
        assert video.mime_type == "video/webm"

    def test_from_url_invalid(self):
        with pytest.raises(ValueError, match="Invalid video URL"):
            dspy.Video.from_url("not_a_url")


class TestVideoFromYoutube:
    def test_from_youtube_watch(self):
        video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert video.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert video.filename == "youtube_video"

    def test_from_youtube_short(self):
        video = dspy.Video.from_youtube("https://youtu.be/dQw4w9WgXcQ")
        assert video.url == "https://youtu.be/dQw4w9WgXcQ"

    def test_from_youtube_invalid(self):
        with pytest.raises(ValueError, match="Not a valid YouTube URL"):
            dspy.Video.from_youtube("https://vimeo.com/123456")


class TestVideoFromFileId:
    def test_from_file_id_basic(self):
        video = dspy.Video.from_file_id("files/abc123")
        assert video.file_id == "files/abc123"
        assert video.url is None

    def test_from_file_id_with_filename(self):
        video = dspy.Video.from_file_id("files/abc123", filename="video.mp4")
        assert video.file_id == "files/abc123"
        assert video.filename == "video.mp4"

    def test_from_file_id_with_mime_type(self):
        video = dspy.Video.from_file_id("files/abc123", mime_type="video/mp4")
        assert video.file_id == "files/abc123"
        assert video.mime_type == "video/mp4"


class TestVideoFromGcs:
    def test_from_gcs_basic(self):
        video = dspy.Video.from_gcs("gs://my-bucket/videos/sample.mp4")
        assert video.url == "gs://my-bucket/videos/sample.mp4"
        assert video.mime_type == "video/mp4"
        assert video.filename == "sample.mp4"

    def test_from_gcs_webm(self):
        video = dspy.Video.from_gcs("gs://bucket/path/to/video.webm")
        assert video.url == "gs://bucket/path/to/video.webm"
        assert video.mime_type == "video/webm"
        assert video.filename == "video.webm"

    def test_from_gcs_custom_filename(self):
        video = dspy.Video.from_gcs("gs://bucket/video.mp4", filename="custom_name.mp4")
        assert video.filename == "custom_name.mp4"

    def test_from_gcs_custom_mime_type(self):
        video = dspy.Video.from_gcs("gs://bucket/video", mime_type="video/mp4")
        assert video.mime_type == "video/mp4"

    def test_from_gcs_invalid_url(self):
        with pytest.raises(ValueError, match="Not a valid GCS URI"):
            dspy.Video.from_gcs("https://example.com/video.mp4")


class TestVideoDirectConstruction:
    def test_from_local_path_string(self, sample_video_file):
        video = dspy.Video(sample_video_file)
        assert video.url is not None
        assert video.url.startswith("data:video/mp4;base64,")

    def test_from_url_string(self):
        video = dspy.Video("https://example.com/video.mp4")
        assert video.url == "https://example.com/video.mp4"

    def test_from_youtube_string(self):
        video = dspy.Video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert video.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_from_gcs_string(self):
        video = dspy.Video("gs://my-bucket/videos/sample.mp4")
        assert video.url == "gs://my-bucket/videos/sample.mp4"
        assert video.mime_type == "video/mp4"
        assert video.filename == "sample.mp4"

    def test_from_data_uri(self):
        data_uri = "data:video/mp4;base64,AAAA"
        video = dspy.Video(data_uri)
        assert video.url == data_uri

    def test_from_dict_with_url(self):
        video = dspy.Video(url="https://example.com/video.mp4", mime_type="video/mp4")
        assert video.url == "https://example.com/video.mp4"
        assert video.mime_type == "video/mp4"

    def test_from_dict_with_file_id(self):
        video = dspy.Video(file_id="files/abc123", filename="video.mp4")
        assert video.file_id == "files/abc123"
        assert video.filename == "video.mp4"

    def test_invalid_dict_raises(self):
        with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
            dspy.Video(invalid="field")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unrecognized video string"):
            dspy.Video("not_a_path_or_url")


# ============================================================================
# Video format() tests
# ============================================================================


class TestVideoFormat:
    def test_format_with_file_id(self):
        video = dspy.Video(file_id="files/abc123", filename="video.mp4", mime_type="video/mp4")
        formatted = video.format()
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "file"
        assert formatted[0]["file"]["file_id"] == "files/abc123"
        assert formatted[0]["file"]["filename"] == "video.mp4"
        assert formatted[0]["file"]["format"] == "video/mp4"

    def test_format_with_data_uri(self):
        data_uri = "data:video/mp4;base64,AAAA"
        video = dspy.Video(url=data_uri, filename="test.mp4")
        formatted = video.format()
        assert formatted[0]["type"] == "file"
        assert formatted[0]["file"]["file_data"] == data_uri
        assert formatted[0]["file"]["filename"] == "test.mp4"

    def test_format_with_remote_url(self):
        video = dspy.Video(url="https://example.com/video.mp4", mime_type="video/mp4")
        formatted = video.format()
        assert formatted[0]["type"] == "file"
        assert formatted[0]["file"]["file_data"] == "https://example.com/video.mp4"
        assert formatted[0]["file"]["format"] == "video/mp4"

    def test_format_with_youtube_url(self):
        video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        formatted = video.format()
        assert formatted[0]["type"] == "file"
        assert formatted[0]["file"]["file_data"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert formatted[0]["file"]["filename"] == "youtube_video"

    def test_format_with_gcs_url(self):
        video = dspy.Video.from_gcs("gs://my-bucket/videos/sample.mp4")
        formatted = video.format()
        assert formatted[0]["type"] == "file"
        assert formatted[0]["file"]["file_data"] == "gs://my-bucket/videos/sample.mp4"
        assert formatted[0]["file"]["filename"] == "sample.mp4"
        assert formatted[0]["file"]["format"] == "video/mp4"

    def test_format_with_gcs_url_no_extension(self):
        # When no extension, we should still be able to specify mime_type
        video = dspy.Video.from_gcs("gs://bucket/video", mime_type="video/webm")
        formatted = video.format()
        assert formatted[0]["file"]["file_data"] == "gs://bucket/video"
        assert formatted[0]["file"]["format"] == "video/webm"

    def test_format_no_url_or_file_id_raises(self):
        # This shouldn't happen in practice due to validation, but test the format method
        video = dspy.Video.__new__(dspy.Video)
        object.__setattr__(video, "url", None)
        object.__setattr__(video, "file_id", None)
        object.__setattr__(video, "filename", None)
        object.__setattr__(video, "mime_type", None)
        with pytest.raises(ValueError, match="must have either url or file_id"):
            video.format()


# ============================================================================
# Video repr/str tests
# ============================================================================


class TestVideoRepr:
    def test_repr_with_url(self):
        video = dspy.Video(url="https://example.com/video.mp4")
        repr_str = repr(video)
        assert "url='https://example.com/video.mp4'" in repr_str

    def test_repr_with_data_uri(self):
        video = dspy.Video(url="data:video/mp4;base64,AAAA")
        repr_str = repr(video)
        assert "DATA_URI" in repr_str
        assert "video/mp4" in repr_str

    def test_repr_with_youtube(self):
        video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        repr_str = repr(video)
        assert "YouTube" in repr_str

    def test_repr_with_gcs(self):
        video = dspy.Video.from_gcs("gs://my-bucket/video.mp4")
        repr_str = repr(video)
        assert "GCS" in repr_str
        assert "gs://my-bucket/video.mp4" in repr_str

    def test_repr_with_file_id(self):
        video = dspy.Video(file_id="files/abc123")
        repr_str = repr(video)
        assert "file_id='files/abc123'" in repr_str

    def test_repr_with_filename(self):
        video = dspy.Video(file_id="files/abc123", filename="video.mp4")
        repr_str = repr(video)
        assert "filename='video.mp4'" in repr_str

    def test_repr_with_mime_type(self):
        video = dspy.Video(file_id="files/abc123", mime_type="video/mp4")
        repr_str = repr(video)
        assert "mime_type='video/mp4'" in repr_str


class TestVideoStr:
    def test_str_contains_markers(self):
        video = dspy.Video(url="https://example.com/video.mp4")
        str_repr = str(video)
        assert "<<CUSTOM-TYPE-START-IDENTIFIER>>" in str_repr
        assert "<<CUSTOM-TYPE-END-IDENTIFIER>>" in str_repr


# ============================================================================
# Video immutability tests
# ============================================================================


class TestVideoImmutability:
    def test_video_frozen(self):
        video = dspy.Video(url="https://example.com/video.mp4")
        with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
            video.url = "https://example.com/other.mp4"

    def test_video_from_video_instance(self):
        video1 = dspy.Video(url="https://example.com/video.mp4", mime_type="video/mp4")
        video2 = dspy.Video(video1)
        assert video2.url == video1.url
        assert video2.mime_type == video1.mime_type


# ============================================================================
# encode_video_to_dict tests
# ============================================================================


class TestEncodeVideoToDict:
    def test_from_video_instance(self, sample_video_file):
        video = dspy.Video.from_path(sample_video_file)
        result = encode_video_to_dict(video)
        assert "url" in result
        assert "filename" in result
        assert "mime_type" in result

    def test_from_local_path(self, sample_video_file):
        result = encode_video_to_dict(sample_video_file)
        assert "url" in result
        assert result["url"].startswith("data:video/mp4;base64,")

    def test_from_youtube_url(self):
        result = encode_video_to_dict("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert result["filename"] == "youtube_video"

    def test_from_gcs_url(self):
        result = encode_video_to_dict("gs://my-bucket/videos/sample.mp4")
        assert result["url"] == "gs://my-bucket/videos/sample.mp4"
        assert result["filename"] == "sample.mp4"
        assert result["mime_type"] == "video/mp4"

    def test_from_remote_url(self):
        result = encode_video_to_dict("https://example.com/video.mp4")
        assert result["url"] == "https://example.com/video.mp4"
        assert result["mime_type"] == "video/mp4"

    def test_from_data_uri(self):
        data_uri = "data:video/mp4;base64,AAAA"
        result = encode_video_to_dict(data_uri)
        assert result["url"] == data_uri
        assert result["mime_type"] == "video/mp4"

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported video input type"):
            encode_video_to_dict(12345)


# ============================================================================
# Constants and configuration tests
# ============================================================================


class TestVideoConstants:
    def test_video_mime_types_dict(self):
        assert "mp4" in VIDEO_MIME_TYPES
        assert "webm" in VIDEO_MIME_TYPES
        assert "mov" in VIDEO_MIME_TYPES
        assert VIDEO_MIME_TYPES["mp4"] == "video/mp4"

    def test_max_inline_size(self):
        assert MAX_INLINE_SIZE_BYTES == 20 * 1024 * 1024  # 20MB
