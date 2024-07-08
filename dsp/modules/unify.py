import logging
from typing import Any, Literal, Optional

from unify.clients import Unify as UnifyClient

from dsp.modules.lm import LM


class Unify(LM):
    """A class to interact with the Unify AI API."""

    def __init__(
        self,
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03",
        # model: Optional[str] = None,
        # provider: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        stream: Optional[bool] = False,
        base_url="https://api.unify.ai/v0",
        system_prompt: Optional[str] = None,
        n: int = 1,
        api_key=None,
        **kwargs,
    ):
        self.api_key = api_key
        # self.model = model
        # self.provider = provider
        self.endpoint = endpoint
        self.stream = stream
        self.client = UnifyClient(api_key=self.api_key, endpoint=self.endpoint)

        super().__init__(model=self.endpoint)

        self.system_prompt = system_prompt
        self.model_type = model_type
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": n,
            **kwargs,
        }
        self.kwargs["endpoint"] = endpoint
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}
        settings_dict = {
            "endpoint": self.endpoint,
            "stream": self.stream,
        }
        if self.model_type == "chat":
            messages = [{"role": "user", "content": prompt}]
            settings_dict["messages"] = messages
            if self.system_prompt:
                settings_dict["messages"].insert(0, {"role": "system", "content": self.system_prompt})
        else:
            settings_dict["prompt"] = prompt

        logging.debug(f"Settings Dict: {settings_dict}")

        if "messages" in settings_dict:
            response = self.client.generate(
                messages=settings_dict["messages"],
                stream=settings_dict["stream"],
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
            )
        else:
            response = self.client.generate(
                prompt=settings_dict["prompt"],
                stream=settings_dict["stream"],
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
            )

        response = {"choices": [{"message": {"content": response}}]}  # response with choices

        if not response:
            logging.error("Unexpected response format, not response")
        elif "choices" not in response:
            logging.error(f"no choices in response: {response}")

        return response

    def request(self, prompt: str, **kwargs) -> Any:
        """Handles retreival of model completions whilst handling rate limiting and caching."""
        if "model_type" in kwargs:
            del kwargs["model_type"]
        return self.basic_request(prompt, **kwargs)

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

            if isinstance(response, dict) and "choices" in response:
                completions.append(response["choices"][0]["message"]["content"])
            else:
                raise ValueError("Unexpected response format")

        return completions
