# Currently, LiteLLM logs way to much to the console in Info
# I am setting this to WARNING to silence this
# at some point, we will need to manage this appropriately with the new dspy logger
import logging
import typing as t

from litellm import ModelResponse, completion
from pydantic import Field

from .base import BaseLM

logger = logging.getLogger("LiteLLM")
logger.setLevel(logging.WARNING)


class LiteLLM(BaseLM):
    STANDARD_PARAMS: dict[str, t.Any] = {
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "num_retries": 3,
    }

    model: str
    default_params: dict[str, t.Any] = Field(default_factory=dict)

    def generate(
        self,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""

        options = {**self.STANDARD_PARAMS, **self.default_params, **kwargs}

        # We are not streaming this content in, therefore we can assume it'll always be a litellm ModelResponse
        response = completion(
            model=self.model,
            **options,
        )

        if type(response) != ModelResponse:
            raise AssertionError("Response from completion incorrect type/format")

        # TODO: Move this to proper logging
        if len([c for c in response.choices if c["finish_reason"] == "length"]) > 0:
            print("Some of the generations are being limited by 'max_tokens', you may want to raise this value.")

        return [c["message"]["content"] for c in response.choices if c["finish_reason"] != "length"]
