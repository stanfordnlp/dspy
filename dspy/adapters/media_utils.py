import re

from typing import Any, Dict, List, Union


def try_expand_media_tags(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Try to expand media tags (audio or image) in the messages."""
    for message in messages:
        if "content" in message and isinstance(message["content"], str):
            content = message["content"]

            # Check for audio tags first (since they also use image_url type)
            if "<DSPY_AUDIO_START>" in content:
                message["content"] = expand_media_content(content, media_type="audio")
            # Only check for image tags if no audio tags were found
            elif "<DSPY_IMAGE_START>" in content:
                message["content"] = expand_media_content(content, media_type="image")

    return messages


def expand_media_content(
    text: str, media_type: str
) -> Union[str, List[Dict[str, Any]]]:
    """Expand media tags in the text into a content list with text and media URLs.

    Args:
        text: The text content that may contain media tags
        media_type: Either "audio" or "image"
    """
    tag_start = f"<DSPY_{media_type.upper()}_START>"
    tag_end = f"<DSPY_{media_type.upper()}_END>"
    tag_regex = rf'"?{tag_start}(.*?){tag_end}"?'

    # If no media tags, return original text
    if not re.search(tag_regex, text):
        return text

    final_list = []
    remaining_text = text

    while remaining_text:
        match = re.search(tag_regex, remaining_text)
        if not match:
            if remaining_text.strip():
                final_list.append({"type": "text", "text": remaining_text.strip()})
            break

        # Get text before the media tag
        prefix = remaining_text[: match.start()].strip()
        if prefix:
            final_list.append({"type": "text", "text": prefix})

        # Add the media URL
        media_url = match.group(1)
        mime_prefix = f"data:{media_type}/"
        if media_url.startswith(mime_prefix):
            final_list.append({"type": "image_url", "image_url": {"url": media_url}})

        # Update remaining text
        remaining_text = remaining_text[match.end() :].strip()

    return final_list
