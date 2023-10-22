from typing import Any

from dspy.callbacks.base_handler import BaseCallbackHandler


def arize_phoenix_callback_handler(**kwargs: Any) -> BaseCallbackHandler:
    try:
        from phoenix.trace.exporter import HttpExporter
        from dspy.callbacks.trace import OpenInferenceTraceCallbackHandler
    except ImportError:
        raise ImportError(
            "Please install Arize Phoenix with `pip install -q arize-phoenix`"
        )
    if "exporter" not in kwargs:
        kwargs = {"exporter": HttpExporter(), **kwargs}
    return OpenInferenceTraceCallbackHandler(**kwargs)
