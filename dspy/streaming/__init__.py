from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StreamResponse
from dspy.streaming.streaming_listener import StreamListener

__all__ = [
    "StatusMessage",
    "StatusMessageProvider",
    "streamify",
    "StreamListener",
    "StreamResponse",
    "streaming_response",
    "apply_sync_streaming",
]


def __getattr__(name):
    if name in ("streamify", "streaming_response", "apply_sync_streaming"):
        from dspy.streaming.streamify import apply_sync_streaming, streamify, streaming_response

        _map = {
            "streamify": streamify,
            "streaming_response": streaming_response,
            "apply_sync_streaming": apply_sync_streaming,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
