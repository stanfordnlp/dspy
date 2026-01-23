import base64
import mimetypes
import os
from typing import Any, Union
from urllib.parse import urlparse

import pydantic

from dspy.adapters.types.base_type import Type

# Supported video MIME types for Gemini
VIDEO_MIME_TYPES = {
    "mp4": "video/mp4",
    "mpeg": "video/mpeg",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "flv": "video/x-flv",
    "mpg": "video/mpeg",
    "webm": "video/webm",
    "wmv": "video/x-ms-wmv",
    "3gp": "video/3gpp",
    "3gpp": "video/3gpp",
}

# Maximum size for inline base64 encoding (20MB)
MAX_INLINE_SIZE_BYTES = 20 * 1024 * 1024


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube video URL."""
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    youtube_domains = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}
    return parsed.netloc in youtube_domains


def is_gcs_url(url: str) -> bool:
    """Check if a URL is a Google Cloud Storage URL (gs://)."""
    if not isinstance(url, str):
        return False
    return url.startswith("gs://")


def is_video_url(url: str) -> bool:
    """Check if a string is a valid video URL (HTTP, HTTPS, or GCS)."""
    try:
        # Check for GCS URLs first
        if is_gcs_url(url):
            return True
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False


def get_video_mime_type(file_path_or_url: str) -> str | None:
    """Get MIME type for a video file from its extension."""
    ext = os.path.splitext(urlparse(file_path_or_url).path)[1].lstrip(".").lower()
    if ext in VIDEO_MIME_TYPES:
        return VIDEO_MIME_TYPES[ext]
    mime_type, _ = mimetypes.guess_type(file_path_or_url)
    return mime_type


class Video(Type):
    """A video input type for DSPy, with support for Gemini video understanding.

    The Video type supports multiple input sources:
    - Local file paths (encoded as base64 for files <20MB)
    - Remote HTTP(S) URLs
    - YouTube URLs (native Gemini support)
    - Google Cloud Storage URIs (gs://)
    - Raw bytes
    - Pre-uploaded file IDs from Gemini Files API

    For videos larger than 20MB, use the `upload()` method to upload to Gemini's
    Files API first, then use the returned file_id.

    Example:
        ```python
        import dspy

        # From local file
        video = dspy.Video("./my_video.mp4")

        # From URL
        video = dspy.Video("https://example.com/video.mp4")

        # From YouTube
        video = dspy.Video("https://www.youtube.com/watch?v=VIDEO_ID")

        # From Google Cloud Storage
        video = dspy.Video("gs://my-bucket/videos/sample.mp4")

        # From pre-uploaded file ID
        video = dspy.Video(file_id="files/abc123", mime_type="video/mp4")

        # Upload large video first
        video = dspy.Video("./large_video.mp4")
        uploaded_video = video.upload()  # Returns new Video with file_id

        # Use in a signature
        class VideoQA(dspy.Signature):
            video: dspy.Video = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        qa = dspy.Predict(VideoQA)
        result = qa(video=dspy.Video("./clip.mp4"), question="What happens in this video?")
        ```

    Note:
        Video understanding is currently best supported by Gemini models.
        This uses LiteLLM's beta Files API support for Gemini.
    """

    url: str | None = None
    file_id: str | None = None
    filename: str | None = None
    mime_type: str | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, source: Any = None, **data):
        """Create a Video.

        Parameters
        ----------
        source:
            The video source. Supported values include:

            - ``str``: Local file path, HTTP(S) URL, YouTube URL, or data URI
            - ``dict`` with video fields (url, file_id, filename, mime_type)

        Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.
        """
        if source is not None and not any(k in data for k in ("url", "file_id", "filename", "mime_type")):
            # Process positional argument
            if isinstance(source, Video):
                data["url"] = source.url
                data["file_id"] = source.file_id
                data["filename"] = source.filename
                data["mime_type"] = source.mime_type
            elif isinstance(source, str):
                # Encode string input (path, URL, data URI, YouTube)
                encoded = encode_video_to_dict(source)
                data.update(encoded)
            elif isinstance(source, dict):
                data.update(source)
            else:
                raise ValueError(f"Unsupported video source type: {type(source)}")

        super().__init__(**data)

    def format(self) -> list[dict[str, Any]]:
        """Format the video for LLM API consumption.

        Returns OpenAI-compatible content blocks that LiteLLM translates for Gemini.
        """
        try:
            if self.file_id:
                # Pre-uploaded file via Gemini Files API
                file_dict = {"file_id": self.file_id}
                if self.filename:
                    file_dict["filename"] = self.filename
                if self.mime_type:
                    file_dict["format"] = self.mime_type
                return [{"type": "file", "file": file_dict}]

            if self.url:
                if is_youtube_url(self.url):
                    # YouTube URLs - pass through for Gemini native handling
                    # Must include format so LiteLLM uses file_uri instead of trying to download
                    return [{"type": "file", "file": {"file_data": self.url, "filename": self.filename or "youtube_video", "format": "video/mp4"}}]

                if is_gcs_url(self.url):
                    # GCS URLs (gs://) - Gemini natively supports these
                    # LiteLLM extracts MIME type from extension, but we pass it if available
                    file_dict = {"file_data": self.url}
                    if self.filename:
                        file_dict["filename"] = self.filename
                    if self.mime_type:
                        file_dict["format"] = self.mime_type
                    else:
                        # Try to detect MIME type from the GCS path extension
                        detected_mime = get_video_mime_type(self.url)
                        if detected_mime:
                            file_dict["format"] = detected_mime
                    return [{"type": "file", "file": file_dict}]

                if self.url.startswith("data:"):
                    # Already a data URI
                    file_dict = {"file_data": self.url}
                    if self.filename:
                        file_dict["filename"] = self.filename
                    return [{"type": "file", "file": file_dict}]

                # Remote URL - pass through
                file_dict = {"file_data": self.url}
                if self.filename:
                    file_dict["filename"] = self.filename
                if self.mime_type:
                    file_dict["format"] = self.mime_type
                return [{"type": "file", "file": file_dict}]

            raise ValueError("Video must have either url or file_id")

        except Exception as e:
            raise ValueError(f"Failed to format video for DSPy: {e}")

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        parts = []
        if self.url is not None:
            if self.url.startswith("data:"):
                mime = self.url.split(";")[0].split(":")[1] if ":" in self.url else "unknown"
                len_data = len(self.url.split("base64,")[1]) if "base64," in self.url else len(self.url)
                parts.append(f"url=<DATA_URI({mime}, {len_data} chars)>")
            elif is_youtube_url(self.url):
                parts.append(f"url='{self.url}' (YouTube)")
            elif is_gcs_url(self.url):
                parts.append(f"url='{self.url}' (GCS)")
            else:
                parts.append(f"url='{self.url}'")
        if self.file_id is not None:
            parts.append(f"file_id='{self.file_id}'")
        if self.filename is not None:
            parts.append(f"filename='{self.filename}'")
        if self.mime_type is not None:
            parts.append(f"mime_type='{self.mime_type}'")
        return f"Video({', '.join(parts)})"

    @classmethod
    def from_path(cls, file_path: str, filename: str | None = None, mime_type: str | None = None) -> "Video":
        """Create a Video from a local file path.

        Args:
            file_path: Path to the video file
            filename: Optional filename to use (defaults to basename of path)
            mime_type: Optional MIME type (defaults to auto-detection)

        Returns:
            Video instance with base64-encoded data URI

        Raises:
            ValueError: If file not found or exceeds inline size limit
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > MAX_INLINE_SIZE_BYTES:
            raise ValueError(
                f"Video file ({file_size / 1024 / 1024:.1f}MB) exceeds {MAX_INLINE_SIZE_BYTES / 1024 / 1024:.0f}MB "
                f"inline limit. Use Video.upload_from_path('{file_path}') to upload via Gemini Files API."
            )

        with open(file_path, "rb") as f:
            video_bytes = f.read()

        if filename is None:
            filename = os.path.basename(file_path)

        if mime_type is None:
            mime_type = get_video_mime_type(file_path)
            if mime_type is None:
                raise ValueError(f"Could not determine MIME type for file: {file_path}")

        encoded_data = base64.b64encode(video_bytes).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded_data}"

        return cls(url=url, filename=filename, mime_type=mime_type)

    @classmethod
    def from_bytes(
        cls,
        video_bytes: bytes,
        mime_type: str,
        filename: str | None = None,
    ) -> "Video":
        """Create a Video from raw bytes.

        Args:
            video_bytes: Raw video bytes
            mime_type: MIME type (e.g., 'video/mp4')
            filename: Optional filename

        Returns:
            Video instance with base64-encoded data URI
        """
        if len(video_bytes) > MAX_INLINE_SIZE_BYTES:
            raise ValueError(
                f"Video data ({len(video_bytes) / 1024 / 1024:.1f}MB) exceeds "
                f"{MAX_INLINE_SIZE_BYTES / 1024 / 1024:.0f}MB inline limit. "
                f"Upload via Gemini Files API first."
            )

        encoded_data = base64.b64encode(video_bytes).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded_data}"
        return cls(url=url, filename=filename, mime_type=mime_type)

    @classmethod
    def from_url(cls, url: str, filename: str | None = None, mime_type: str | None = None) -> "Video":
        """Create a Video from a remote URL.

        Args:
            url: HTTP(S) URL to the video, or YouTube URL
            filename: Optional filename
            mime_type: Optional MIME type (auto-detected from URL if not provided)

        Returns:
            Video instance referencing the URL
        """
        if not is_video_url(url) and not is_youtube_url(url):
            raise ValueError(f"Invalid video URL: {url}")

        if mime_type is None and not is_youtube_url(url):
            mime_type = get_video_mime_type(url)

        return cls(url=url, filename=filename, mime_type=mime_type)

    @classmethod
    def from_youtube(cls, url: str) -> "Video":
        """Create a Video from a YouTube URL.

        Args:
            url: YouTube video URL (youtube.com/watch or youtu.be format)

        Returns:
            Video instance for YouTube URL

        Note:
            Gemini natively supports YouTube URLs. Free tier allows up to 8 hours
            of YouTube video per day; paid tiers have no restrictions.
        """
        if not is_youtube_url(url):
            raise ValueError(f"Not a valid YouTube URL: {url}")
        return cls(url=url, filename="youtube_video")

    @classmethod
    def from_file_id(cls, file_id: str, filename: str | None = None, mime_type: str | None = None) -> "Video":
        """Create a Video from a pre-uploaded Gemini file ID.

        Args:
            file_id: The file ID returned from Gemini Files API upload
            filename: Optional filename
            mime_type: Optional MIME type

        Returns:
            Video instance referencing the uploaded file
        """
        return cls(file_id=file_id, filename=filename, mime_type=mime_type)

    @classmethod
    def from_gcs(cls, gcs_uri: str, filename: str | None = None, mime_type: str | None = None) -> "Video":
        """Create a Video from a Google Cloud Storage URI.

        Args:
            gcs_uri: GCS URI in format gs://bucket-name/path/to/video.mp4
            filename: Optional filename (defaults to basename of path)
            mime_type: Optional MIME type (auto-detected from URI extension if not provided)

        Returns:
            Video instance referencing the GCS URI

        Note:
            The GCS bucket must be accessible to the Gemini API. For Vertex AI,
            this typically means the bucket should be in the same project or
            have appropriate IAM permissions.

        Example:
            ```python
            video = dspy.Video.from_gcs("gs://my-bucket/videos/sample.mp4")
            result = qa(video=video, question="What is in this video?")
            ```
        """
        if not is_gcs_url(gcs_uri):
            raise ValueError(f"Not a valid GCS URI (must start with gs://): {gcs_uri}")

        if mime_type is None:
            mime_type = get_video_mime_type(gcs_uri)

        if filename is None:
            # Extract filename from GCS path
            filename = os.path.basename(gcs_uri)

        return cls(url=gcs_uri, filename=filename, mime_type=mime_type)

    @classmethod
    def upload_from_path(cls, file_path: str, api_key: str | None = None, mime_type: str | None = None) -> "Video":
        """Upload a video file directly to Gemini Files API.

        Use this method for videos of any size, especially those larger than 20MB
        that cannot be inlined. This bypasses the inline size limit entirely.

        Args:
            file_path: Path to the video file to upload
            api_key: Optional Gemini API key. If not provided, uses GEMINI_API_KEY
                    environment variable.
            mime_type: Optional MIME type (auto-detected from file extension if not provided)

        Returns:
            Video instance with file_id set (ready to use with Gemini)

        Raises:
            ValueError: If file not found or no API key available
            ImportError: If litellm doesn't support create_file

        Example:
            ```python
            # Upload a large video directly
            video = dspy.Video.upload_from_path("./large_video.mp4")

            # Use it in your signature
            result = qa(video=video, question="What happens in this video?")
            ```
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        try:
            from litellm import create_file
        except ImportError:
            raise ImportError(
                "litellm.create_file is required for video uploads. "
                "Please update litellm: pip install --upgrade litellm"
            )

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Detect MIME type
        if mime_type is None:
            mime_type = get_video_mime_type(file_path)
            if mime_type is None:
                mime_type = "video/mp4"  # Default fallback

        # Read and upload the file
        with open(file_path, "rb") as f:
            video_data = f.read()

        filename = os.path.basename(file_path)

        # Pass file as tuple (filename, data, content_type) to include MIME type
        response = create_file(
            file=(filename, video_data, mime_type),
            purpose="user_data",
            custom_llm_provider="gemini",
            api_key=api_key,
        )

        return cls(
            file_id=response.id,
            filename=filename,
            mime_type=mime_type,
        )

    def upload(self, api_key: str | None = None) -> "Video":
        """Upload video to Gemini Files API and return a new Video with file_id.

        This method uploads the video data to Google's servers using LiteLLM's
        Files API support. Use this for videos that are already loaded as data URIs.

        For large files (>20MB), use `Video.upload_from_path()` instead.

        Args:
            api_key: Optional Gemini API key. If not provided, uses GEMINI_API_KEY
                    environment variable.

        Returns:
            New Video instance with file_id set

        Raises:
            ValueError: If video has no uploadable data
            ImportError: If litellm doesn't support create_file

        Example:
            ```python
            video = dspy.Video("./small_video.mp4")  # Must be <20MB
            uploaded = video.upload()
            # Use uploaded.file_id in your requests
            ```
        """
        try:
            from litellm import create_file
        except ImportError:
            raise ImportError(
                "litellm.create_file is required for video uploads. "
                "Please update litellm: pip install --upgrade litellm"
            )

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "No API key provided. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Get video data to upload
        if self.url and self.url.startswith("data:"):
            # Extract bytes from data URI
            header, b64data = self.url.split(",", 1)
            video_data = base64.b64decode(b64data)
        elif self.url and os.path.isfile(self.url):
            # Read from file path
            with open(self.url, "rb") as f:
                video_data = f.read()
        elif self.url and is_video_url(self.url) and not is_youtube_url(self.url):
            # Download from URL
            import requests
            response = requests.get(self.url)
            response.raise_for_status()
            video_data = response.content
        else:
            raise ValueError(
                "Cannot upload: Video must have a data URI, local file path, or remote URL. "
                "YouTube URLs and file_ids don't need uploading."
            )

        # Upload to Gemini - pass file as tuple (filename, data, content_type) to include MIME type
        upload_filename = self.filename or "video"
        upload_mime_type = self.mime_type or "video/mp4"
        response = create_file(
            file=(upload_filename, video_data, upload_mime_type),
            purpose="user_data",
            custom_llm_provider="gemini",
            api_key=api_key,
        )

        return Video(
            file_id=response.id,
            filename=self.filename,
            mime_type=self.mime_type,
        )


def encode_video_to_dict(video_input: Any) -> dict:
    """Encode various video inputs to a dict with url, file_id, filename, and/or mime_type.

    Args:
        video_input: Can be a file path (str), URL (str), bytes, or Video instance.

    Returns:
        dict: A dictionary with video fields.
    """
    if isinstance(video_input, Video):
        result = {}
        if video_input.url is not None:
            result["url"] = video_input.url
        if video_input.file_id is not None:
            result["file_id"] = video_input.file_id
        if video_input.filename is not None:
            result["filename"] = video_input.filename
        if video_input.mime_type is not None:
            result["mime_type"] = video_input.mime_type
        return result

    elif isinstance(video_input, str):
        # Check if it's a data URI
        if video_input.startswith("data:video/"):
            mime_match = video_input.split(";")[0].split(":")[1] if ":" in video_input else None
            return {"url": video_input, "mime_type": mime_match}

        # Check if it's a YouTube URL
        if is_youtube_url(video_input):
            return {"url": video_input, "filename": "youtube_video"}

        # Check if it's a GCS URL
        if is_gcs_url(video_input):
            mime_type = get_video_mime_type(video_input)
            filename = os.path.basename(video_input)
            return {"url": video_input, "filename": filename, "mime_type": mime_type}

        # Check if it's a local file
        if os.path.isfile(video_input):
            video_obj = Video.from_path(video_input)
            return {
                "url": video_obj.url,
                "filename": video_obj.filename,
                "mime_type": video_obj.mime_type,
            }

        # Check if it's a URL
        if is_video_url(video_input):
            mime_type = get_video_mime_type(video_input)
            return {"url": video_input, "mime_type": mime_type}

        raise ValueError(
            f"Unrecognized video string: {video_input}; must be a valid file path, URL, or YouTube URL"
        )

    elif isinstance(video_input, bytes):
        raise ValueError(
            "Cannot create Video from bytes without mime_type. "
            "Use Video.from_bytes(data, mime_type='video/mp4') instead."
        )

    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")
