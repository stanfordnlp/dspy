from dspy.clients.openai_format import part_to_openai_blocks
from dspy.core.types import LMBinaryPart, LMMessage, LMVideoPart


def test_binary_part_emits_format_detail_and_video_metadata():
    part = LMBinaryPart(
        file_id="https://generativelanguage.googleapis.com/v1beta/files/abc123",
        media_type="video/webm",
        detail="high",
        video_metadata={"fps": 1.0},
    )

    assert part_to_openai_blocks(part) == [
        {
            "type": "file",
            "file": {
                "file_id": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
                "format": "video/webm",
                "detail": "high",
                "video_metadata": {"fps": 1.0},
            },
        }
    ]


def test_binary_part_omits_format_for_default_media_type():
    part = LMBinaryPart(file_id="file-abc123", filename="document.bin")

    assert part_to_openai_blocks(part) == [
        {"type": "file", "file": {"file_id": "file-abc123", "filename": "document.bin"}}
    ]


def test_file_content_block_round_trips_through_openai_blocks():
    block = {
        "type": "file",
        "file": {
            "file_id": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
            "format": "video/webm",
            "detail": "high",
            "video_metadata": {"fps": 1.0},
        },
    }
    part = LMMessage(role="user", content=[block]).parts[0]

    assert part_to_openai_blocks(part) == [block]


def test_video_part_keeps_media_type_as_file_format():
    part = LMVideoPart(
        file_id="https://generativelanguage.googleapis.com/v1beta/files/abc123",
        media_type="video/webm",
    )

    assert part_to_openai_blocks(part) == [
        {
            "type": "file",
            "file": {
                "file_id": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
                "format": "video/webm",
            },
        }
    ]


def test_video_content_block_keeps_media_type_when_emitted_to_openai():
    block = {"type": "video", "video": {"file_id": "files/abc123", "media_type": "video/webm"}}
    part = LMMessage(role="user", content=[block]).parts[0]

    assert part_to_openai_blocks(part) == [
        {"type": "file", "file": {"file_id": "files/abc123", "format": "video/webm"}}
    ]
