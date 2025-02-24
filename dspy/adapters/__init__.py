from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Image, History, encode_image, is_image
from dspy.adapters.audio_utils import Audio, encode_audio, is_audio
from dspy.adapters.media_utils import try_expand_media_tags

__all__ = [
    'Adapter',
    'ChatAdapter',
    'JSONAdapter',
    'Image',
    'Audio',
    'encode_image',
    'encode_audio',
    'is_image',
    'is_audio',
    'try_expand_media_tags',
    "History",
]
