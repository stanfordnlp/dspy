import base64
import mimetypes
import os
from typing import Any, Union

import pydantic
import requests

from dspy.adapters.types.base_type import Type


class Video(Type):
    data: str
    video_format: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def format(self) -> list[dict[str, Any]]:
        try:
            data = self.data
        except Exception as e:
            raise ValueError(f"Failed to format video for DSPy: {e}")
        return [{
            "type": "input_video",
            "input_video": {
                "data": data,
                "format": self.video_format,
            },
        }]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        """Validate input for Video, expecting 'data' and 'video_format' keys in dictionary."""
        if isinstance(values, cls):
            return {"data": values.data, "video_format": values.video_format}
        return encode_video(values)

    @classmethod
    def from_url(cls, url: str) -> "Video":
        """Download a video file from URL and encode it as base64."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "video/mp4")
        # Strip parameters (e.g., "video/mp4; charset=binary" → "video/mp4")
        mime_type = mime_type.split(";")[0].strip()
        if not mime_type.startswith("video/"):
            raise ValueError(f"Unsupported MIME type for video: {mime_type}")
        video_format = mime_type.split("/")[1]
        encoded_data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=encoded_data, video_format=video_format)

    @classmethod
    def from_file(cls, file_path: str) -> "Video":
        """Read local video file and encode it as base64."""
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("video/"):
            raise ValueError(f"Unsupported MIME type for video: {mime_type}")

        with open(file_path, "rb") as file:
            file_data = file.read()

        video_format = mime_type.split("/")[1]
        encoded_data = base64.b64encode(file_data).decode("utf-8")
        return cls(data=encoded_data, video_format=video_format)

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Video(data=<VIDEO_BASE_64_ENCODED({length})>, video_format='{self.video_format}')"


def encode_video(video: Union[str, bytes, dict, "Video", Any]) -> dict:
    """Encode video to a dict with 'data' and 'video_format'.

    Accepts: local file path, URL, data URI, dict, Video instance, or bytes.
    """
    if isinstance(video, dict) and "data" in video and "video_format" in video:
        return video
    elif isinstance(video, Video):
        return {"data": video.data, "video_format": video.video_format}
    elif isinstance(video, str) and video.startswith("data:video/"):
        try:
            header, b64data = video.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            video_format = mime.split("/")[1]
            return {"data": b64data, "video_format": video_format}
        except Exception as e:
            raise ValueError(f"Malformed video data URI: {e}")
    elif isinstance(video, str) and os.path.isfile(video):
        v = Video.from_file(video)
        return {"data": v.data, "video_format": v.video_format}
    elif isinstance(video, str) and video.startswith("http"):
        v = Video.from_url(video)
        return {"data": v.data, "video_format": v.video_format}
    elif isinstance(video, bytes):
        encoded = base64.b64encode(video).decode("utf-8")
        return {"data": encoded, "video_format": "mp4"}
    else:
        raise ValueError(f"Unsupported type for encode_video: {type(video)}")
