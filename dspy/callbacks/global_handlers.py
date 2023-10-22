from typing import Any
from dspy.callbacks.arize_phoenix_callback import arize_phoenix_callback_handler

from dspy.callbacks.base_handler import BaseCallbackHandler
from dspy.callbacks.open_inference_callback import OpenInferenceCallbackHandler


def set_global_handler(eval_mode: str, **eval_params: Any) -> None:
    """Set global eval handlers."""
    import dspy

    dspy.global_handler = create_global_handler(eval_mode, **eval_params)


def create_global_handler(eval_mode: str, **eval_params: Any) -> BaseCallbackHandler:
    """Get global eval handler."""
    if eval_mode == "openinference":
        handler = OpenInferenceCallbackHandler(**eval_params)
    elif eval_mode == "arize_phoenix":
        handler = arize_phoenix_callback_handler(**eval_params)
    # elif eval_mode == "simple":
    #     handler = SimpleLLMHandler(**eval_params)
    else:
        raise ValueError(f"Eval mode {eval_mode} not supported.")

    return handler