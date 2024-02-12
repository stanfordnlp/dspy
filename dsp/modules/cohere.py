import math
from typing import Any, Optional
import backoff

from dsp.modules.lm import LM

try:
    import cohere
    cohere_api_error = cohere.CohereAPIError
except ImportError:
    cohere_api_error = Exception
    # print("Not loading Cohere because it is not installed.")


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


class Cohere(LM):
    """Wrapper around Cohere's API.

    Currently supported models include `command`, `command-nightly`, `command-light`, `command-light-nightly`.
    """

    def __init__(
        self,
        model: str = "command-nightly",
        api_key: Optional[str] = None,
        stop_sequences: list[str] = [],
        **kwargs
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Cohere to use?
            Choices are [`command`, `command-nightly`, `command-light`, `command-light-nightly`]
        api_key : str
            The API key for Cohere.
            It can be obtained from https://dashboard.cohere.ai/register.
        stop_sequences : list of str
            Additional stop tokens to end generation.
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)
        self.co = cohere.Client(api_key)
        self.provider = "cohere"
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "num_generations": 1,
            **kwargs
        }
        self.stop_sequences = stop_sequences
        self.max_num_generations = 5

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            "stop_sequences": self.stop_sequences,
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
        (cohere_api_error),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Cohere whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs
    ):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # Cohere uses 'num_generations' whereas dsp.generate() uses 'n'
        n = kwargs.pop("n", 1)

        # Cohere can generate upto self.max_num_generations completions at a time
        choices = []
        num_iters = math.ceil(n / self.max_num_generations)
        remainder = n % self.max_num_generations
        for i in range(num_iters):
            if i == (num_iters - 1):
                kwargs["num_generations"] = (
                    remainder if remainder != 0 else self.max_num_generations
                )
            else:
                kwargs["num_generations"] = self.max_num_generations
            response = self.request(prompt, **kwargs)
            choices.extend(response.generations)
        completions = [c.text for c in choices]

        if return_sorted and kwargs.get("num_generations", 1) > 1:
            scored_completions = []

            for c in choices:
                scored_completions.append((c.likelihood, c.text))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions
