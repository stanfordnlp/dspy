import logging
import os
from typing import Any, Literal, Optional

from dsp.modules.lm import LM


class Unify(LM):
    """A class to interact with the Unify AI API."""

    API_BASE = "https://api.unify.ai/v0"

    def __init__(
        self,
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03",
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        api_key=None,
        **kwargs,  # Added to accept additional keyword arguments
    ):
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.model = endpoint

        super().__init__(model=self.model)
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1,
            "top_k": 20,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "num_ctx": 1024,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}

        settings_dict = {
            "model": self.model,
            "options": {k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]},
            "stream": False,
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt

        return self._call_generate(settings_dict)

    def request(self, prompt: str, **kwargs) -> Any:
        """Handles retreival of model completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _call_generate(self, settings_dict) -> Any:
        """Call the generate method from the unify client."""
        try:
            return Unify.generate(settings=settings_dict, api_key=self.api_key)
        except Exception as e:
            logging.error(f"An error occurred while calling the generate method: {e}")
            return None

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Request completions from the Unify API."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for _ in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.choices[0].message.content)

        return completions
