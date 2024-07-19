from typing import Any, Optional

import backoff

from dsp.modules.lm import LM
from dsp.utils.settings import settings

try:
    from unify.clients import Unify as UnifyClient
    from unify.exceptions import AuthenticationError, RateLimitError

    unify_api_error = (AuthenticationError, RateLimitError)
except ImportError:
    unify_api_error = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class Unify(LM):
    """A class to interact with the Unify AI API."""

    def __init__(
        self,
        model: str = "llama-3-8b-chat@together-ai",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        self.client = UnifyClient(api_key=api_key, endpoint=model)
        super().__init__(model=model)
        self.provider = "unify"
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1,
            "n": 1,
            **kwargs,
        }
        self.history: list[dict[str, Any]] = []

    def basic_request(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]
        if "system_prompt" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})

        raw_response = self.client.generate(
            messages=messages,
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )
        response_content = (
            raw_response if isinstance(raw_response, str) else "".join(raw_response)
        )  # to handle stream=True or False output format
        formated_response: dict = {"choices": [{"message": {"content": response_content}}]}

        history = {
            "prompt": prompt,
            "response": formated_response,
            "kwargs": kwargs,
        }
        self.history.append(history)

        return formated_response

    @backoff.on_exception(
        backoff.expo,
        unify_api_error,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Unfify whilst handling API errors."""
        if "model_type" in kwargs:
            del kwargs["model_type"]
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: Optional[str],
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Request completions from the Unify API."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n: int = kwargs.pop("n", 1)
        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response["choices"][0]["message"]["content"])

        return completions
