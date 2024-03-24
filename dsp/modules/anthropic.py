import logging
import os
from typing import Any, Optional

import backoff

from dsp.modules.lm import LM

try:
    import anthropic
    anthropic_rate_limit = anthropic.RateLimitError
except ImportError:
    anthropic_rate_limit = Exception


logger = logging.getLogger(__name__)

BASE_URL = "https://api.anthropic.com/v1/messages"


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/."""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def giveup_hdlr(details):
    """Wrapper function that decides when to give up on retry."""
    if "rate limits" in details.message:
        return False
    return True


class Claude(LM):
    """Wrapper around anthropic's API. Supports both the Anthropic and Azure APIs."""
    def __init__(
            self,
            model: str = "claude-instant-1.2",
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(model)

        try:
            from anthropic import Anthropic
        except ImportError as err:
            raise ImportError("Claude requires `pip install anthropic`.") from err

        self.provider = "anthropic"
        self.api_key = api_key = os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key
        self.api_base = BASE_URL if api_base is None else api_base

        self.kwargs = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": min(kwargs.get("max_tokens", 4096), 4096),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 1),
            "n": kwargs.pop("n", kwargs.pop("num_generations", 1)),
            **kwargs,
        }
        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        self.client = Anthropic(api_key=api_key)

    def log_usage(self, response):
        """Log the total tokens from the Anthropic API response."""
        usage_data = response.usage
        if usage_data:
            total_tokens = usage_data.input_tokens + usage_data.output_tokens
            logger.info(f'{total_tokens}')

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        # caching mechanism requires hashable kwargs
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs.pop("n")
        response = self.client.messages.create(**kwargs)

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
        (anthropic_rate_limit),
        max_time=1000,
        max_tries=8,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Anthropic whilst handling API errors."""
        return self.basic_request(prompt, **kwargs)

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """Retrieves completions from Anthropic.

        Args:
            prompt (str): prompt to send to Anthropic
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[str]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"


        # per eg here: https://docs.anthropic.com/claude/reference/messages-examples
        # max tokens can be used as a proxy to return smaller responses
        # so this cannot be a proper indicator for incomplete response unless it isnt the user-intent.

        n = kwargs.pop("n", 1)
        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            # TODO: Log llm usage instead of hardcoded openai usage
            # if dsp.settings.log_openai_usage:
            #     self.log_usage(response)
            if only_completed and response.stop_reason == "max_tokens":
                continue
            completions = [c.text for c in response.content]
        return completions
