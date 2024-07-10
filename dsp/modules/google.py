import os
from collections.abc import Iterable
from typing import Any, Optional

import backoff

from dsp.modules.lm import LM
from dsp.utils.settings import settings

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError
    google_api_error = GoogleAPICallError
except ImportError:
    google_api_error = Exception
    # print("Not loading Google because it is not installed.")


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


BLOCK_ONLY_HIGH = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
]


class Google(LM):
    """Wrapper around Google's API.

    Currently supported models include `gemini-pro-1.0`.
    """

    def __init__(
        self,
        model: str = "models/gemini-1.0-pro",
        api_key: Optional[str] = None,
        safety_settings: Optional[Iterable] = BLOCK_ONLY_HIGH,
        **kwargs,
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
        api_key = os.environ.get("GOOGLE_API_KEY") if api_key is None else api_key
        genai.configure(api_key=api_key)

        # Google API uses "candidate_count" instead of "n" or "num_generations"
        # For now, google API only supports 1 generation at a time. Raises an error if candidate_count > 1
        num_generations = kwargs.pop("n", kwargs.pop("num_generations", 1))

        self.provider = "google"
        kwargs = {
            "candidate_count": 1,
            "temperature": 0.0 if "temperature" not in kwargs else kwargs["temperature"],
            "max_output_tokens": 2048,
            "top_p": 1,
            "top_k": 1,
            **kwargs,
        }

        self.config = genai.GenerationConfig(**kwargs)
        self.llm = genai.GenerativeModel(model_name=model,
                                         generation_config=self.config,
                                         safety_settings=safety_settings)

        self.kwargs = {
            "n": num_generations,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            **kwargs,
        }

        # Google disallows "n" arguments
        n = kwargs.pop("n", None)
        if n is not None and n > 1 and kwargs['temperature'] == 0.0:
            kwargs['temperature'] = 0.7

        response = self.llm.generate_content(prompt, generation_config=kwargs)

        history = {
            "prompt": prompt,
            "response": [response],
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (google_api_error),
        max_time=settings.backoff_time,
        max_tries=8,
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
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.parts[0].text)

        return completions
