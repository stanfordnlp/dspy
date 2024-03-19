import typing as t

from litellm import ModelResponse, completion, token_counter
from pydantic import Field

from .base import BaseLM, GeneratedContent


class LiteLM(BaseLM):
    STANDARD_PARAMS: dict[str, t.Any] = {
        "temperature": 0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    model: str
    default_params: dict[str, t.Any] = Field(default_factory=dict)

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> list[GeneratedContent]:
        """Generates `n` predictions for the signature output."""
        options = {**self.STANDARD_PARAMS, **self.default_params, **kwargs}
        # We are not streaming this content in, therefore we can assume it'll always be a litellm ModelResponse
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )

        if type(response) != ModelResponse:
            raise AssertionError("Response from completion incorrect type/format")

        return [dict(c) for c in response.choices if c["finish_reason"] != "length"]

    def count_tokens(self, prompt: str) -> int:
        """Counts the number of tokens for a specific prompt."""
        return token_counter(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
