import typing as t

from litellm import completion
from pydantic import Field


from .base import BaseLM


class LiteLLM(BaseLM):
    STANDARD_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    model: str
    default_params: dict[str, t.Any] = Field(default_factory=dict)

    def __call__(
        self,
        prompt: str,
        **kwargs,
    ) -> list[str]:
        """Generates `n` predictions for the signature output."""
        options = {**self.STANDARD_PARAMS, **self.default_params, **kwargs}
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
        choices = [c for c in response["choices"] if c["finish_reason"] != "length"]
        return [c["message"]["content"] for c in choices]
