import logging
from typing import Any

import backoff

try:
    import groq
    from groq import Groq

    groq_api_error = (groq.APIError, groq.RateLimitError)
except ImportError:
    groq_api_error = Exception


from dsp.modules.lm import LM
from dsp.utils.settings import settings


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class GroqLM(LM):
    """Wrapper around groq's API.

    Args:
        model (str, optional): groq supported LLM model to use. Defaults to "mixtral-8x7b-32768".
        api_key (Optional[str], optional): API provider Authentication token. use Defaults to None.
        **kwargs: Additional arguments to pass to the API provider.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "mixtral-8x7b-32768",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "groq"
        if api_key:
            self.api_key = api_key
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError("api_key is required for groq")

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }
        models = self.client.models.list().data
        if models is not None:
            if model in [m.id for m in models]:
                self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def log_usage(self, response):
        """Log the total tokens from the Groq API response."""
        usage_data = response.usage  # Directly accessing the 'usage' attribute
        if usage_data:
            total_tokens = usage_data.total_tokens
            logging.debug(f"Groq Total Tokens Response Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}

        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = self.chat_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }

        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        groq_api_error,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retreival of model completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice) -> str:
        return choice.message.content

    def chat_request(self, **kwargs):
        """Handles retreival of model completions whilst handling rate limiting and caching."""
        response = self.client.chat.completions.create(**kwargs)
        return response

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from model.

        Args:
            prompt (str): prompt to send to model
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        choices = response.choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions
