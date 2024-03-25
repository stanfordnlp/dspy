from typing import Any, Optional

import backoff

from dsp.modules.lm import LM

try:
    import mistralai
    from mistralai.client import MistralClient
    from mistralai.exceptions import MistralAPIException
    from mistralai.models.chat_completion import ChatCompletionResponse, ChatMessage
    mistralai_api_error = MistralAPIException
except ImportError:
    mistralai_api_error = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


class Mistral(LM):
    """Wrapper around Mistral AI's API.

    Currently supported models include `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`.
    """

    def __init__(
        self,
        model: str = "mistral-medium-latest",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Mistral AI to use?
            Choices are [`mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`]
        api_key : str
            The API key for Mistral AI.
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)

        if mistralai_api_error == Exception:
            raise ImportError("Not loading Mistral AI because it is not installed. Install it with `pip install mistralai`.")

        self.client = MistralClient(api_key=api_key)

        self.provider = "mistral"
        self.kwargs = {
            "model": model,
            "temperature": 0.17,
            "max_tokens": 150,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        """Basic request to Mistral AI's API."""
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            "messages": [ChatMessage(role="user", content=prompt)],
            **kwargs,
        }

        # Mistral disallows "n" arguments
        n = kwargs.pop("n", None)
        if n is not None and n > 1 and kwargs['temperature'] == 0.0:
            kwargs['temperature'] = 0.7

        response = self.client.chat(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (mistralai_api_error),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Mistral AI whilst handling API errors."""
        prompt = prompt + "Follow the format only once !"
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.choices[0].message.content)

        return completions
