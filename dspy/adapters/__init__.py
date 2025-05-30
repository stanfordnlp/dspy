from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.retry_adapter import RetryAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.types import History, Image, Audio, BaseType

DEFAULT_ADAPTER = RetryAdapter(main_adapter=ChatAdapter(), fallback_adapter=JSONAdapter())

__all__ = [
    "Adapter",
    "ChatAdapter",
    "RetryAdapter",
    "BaseType",
    "History",
    "Image",
    "Audio",
    "JSONAdapter",
    "TwoStepAdapter",
]
