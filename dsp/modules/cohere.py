from typing import Any, Optional

import backoff

from dsp.modules.lm import LM
from dsp.utils.settings import settings

try:
    import cohere
    cohere_api_error = cohere.errors.UnauthorizedError
except ImportError:
    cohere_api_error = Exception
    # print("Not loading Cohere because it is not installed.")
except AttributeError:
    cohere_api_error = Exception


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


class Cohere(LM):
    """Wrapper around Cohere's API.

    Currently supported models include `command-r-plus`, `command-r`, `command`, `command-nightly`, `command-light`, `command-light-nightly`.
    """

    def __init__(
        self,
        model: str = "command-r",
        api_key: Optional[str] = None,
        stop_sequences: list[str] = [],
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Cohere to use?
            Choices are [`command-r-plus`, `command-r`, `command`, `command-nightly`, `command-light`, `command-light-nightly`]
        api_key : str
            The API key for Cohere.
            It can be obtained from https://dashboard.cohere.ai/register.
        stop_sequences : list of str
            Additional stop tokens to end generation.
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)
        self.co = cohere.Client(api_key, client_name='dspy')
        self.provider = "cohere"
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 2000,
            "p": 1,
            "num_generations": 1,
            **kwargs,
        }
        self.stop_sequences = stop_sequences
        self.max_num_generations = 5

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            "stop_sequences": self.stop_sequences,
            "chat_history": [],
            "message": prompt,
            **kwargs,
        }
        kwargs.pop("num_generations")
        if "n" in kwargs.keys():
            kwargs.pop("n")
        response = self.co.chat(**kwargs)

        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": kwargs,
                "raw_kwargs": raw_kwargs,
            },
        )

        return response

    @backoff.on_exception(
        backoff.expo,
        (cohere_api_error),
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Cohere whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        **kwargs,
    ):
        response = self.request(prompt, **kwargs)
        return [response.text]
