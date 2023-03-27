import math
from typing import Any, Optional

from dsp.modules.lm import LM

try:
    import cohere
except ImportError:
    print("Not loading Cohere because it is not installed.")


class Cohere(LM):
    """Wrapper around Cohere's API.

    Currently supported models include `medium-20221108`, `xlarge-20221108`, `command-medium-nightly`, and `command-xlarge-nightly`.
    """

    def __init__(
        self,
        model: str = "command-xlarge-nightly",
        api_key: Optional[str] = None,
        stop_sequences: list[str] = [],
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Cohere to use?
            Choices are ['medium-20221108', 'xlarge-20221108', 'command-medium-nightly', 'command-xlarge-nightly']
        api_key : str
            The API key for Cohere.
            It can be obtained from https://dashboard.cohere.ai/register.
        stop_sequences : list of str
            Additional stop tokens to end generation.
        """
        super().__init__(model)
        self.co = cohere.Client(api_key)

        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "num_generations": 1,
            "return_likelihoods": "GENERATION",
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
            response = self.basic_request(prompt, **kwargs)
            choices.extend(response.generations)
        completions = [c.text for c in choices]

        if return_sorted and kwargs.get("num_generations", 1) > 1:
            scored_completions = []

            for c in choices:
                scored_completions.append((c.likelihood, c.text))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions
