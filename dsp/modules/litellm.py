from typing import Any, Optional, Literal, cast

import litellm
from litellm.types.utils import Choices, ModelResponse, Usage

from dspy.utils.logging import logger
from dsp.modules.schemas import (
    ChatMessage,
    DSPyModelResponse,
    LLMParams
)

from dsp.modules.lm import LM
from dsp.utils.settings import settings

class LLM(LM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        api_provider: str,
        api_base: str,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        