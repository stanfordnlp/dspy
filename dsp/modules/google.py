import math
from typing import Any, Optional
import backoff

from dsp.modules.lm import LM

try:
    import google.generativeai as genai
except ImportError:
    google_api_error = Exception
    # print("Not loading Google because it is not installed.")


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details)
    )


def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


class Google(LM):
    """Wrapper around Google's API.

    Currently supported models include `gemini-pro-1.0`.
    """

    def __init__(
        self, model: str = "gemini-pro-1.0", api_key: Optional[str] = None, **kwargs
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Google to use?
            Choices are [`gemini-pro-1.0`]
        api_key : str
            The API key for Google.
            It can be obtained from https://cloud.google.com/generative-ai-studio
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)
        self.google = genai.configure(api_key=api_key)
        self.provider = "google"
        self.kwargs = {
            "model_name": model,
            "temperature": 0.0
            if "temperature" not in kwargs
            else kwargs["temperature"],
            "max_output_tokens": 2048,
            "top_p": 1,
            "top_k": 1,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            "prompt": prompt,
            **kwargs,
        }
        response = self.co.generate(**kwargs)

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
        (Exception),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Google whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        return self.request(prompt, **kwargs)
