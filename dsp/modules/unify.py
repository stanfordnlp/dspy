import logging
from typing import Any, Optional

from unify.clients import Unify as UnifyClient

from dsp.modules.lm import LM


class Unify(LM, UnifyClient):
    """A class to interact with the Unify AI API."""

    def __init__(
        self,
        endpoint: str = "router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key=None,
        stream: Optional[bool] = False,
        system_prompt: Optional[str] = None,
        base_url: str = "https://api.unify.ai/v0",
        n: int = 1,
        **kwargs,
    ):
        self.base_url = base_url
        self.stream = stream
        LM.__init__(self, model)
        UnifyClient.__init__(self, endpoint=endpoint, model=model, provider=provider, api_key=api_key)
        # super().__init__(model)
        self.system_prompt = system_prompt
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
        self._dspy_integration_provider = "unify"

    @property
    def provider(self) -> Optional[str]:
        return self._dspy_integration_provider

    @provider.setter
    def provider(self, value: str) -> None:
        self._dspy_integration_provider = value

    @property
    def model_provider(self) -> Optional[str]:
        return UnifyClient.provider(self)

    @model_provider.setter
    def model_provider(self, value: str) -> None:
        if value != "default":
            self.set_provider(value)

    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}
        settings_dict = {
            "endpoint": self.endpoint,
            "stream": self.stream,
        }
        messages = [{"role": "user", "content": prompt}]
        settings_dict["messages"] = messages
        if self.system_prompt:
            settings_dict["messages"].insert(0, {"role": "system", "content": self.system_prompt})

        logging.debug(f"Settings Dict: {settings_dict}")

        response = self.generate(
            messages=settings_dict["messages"],
            stream=settings_dict["stream"],
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )

        response = {"choices": [{"message": {"content": response}}]}  # response with choices

        if not response:
            logging.error("Unexpected response format, not response")
        elif "choices" not in response:
            logging.error(f"no choices in response: {response}")
        if isinstance(response, dict) and "choices" in response:
            self.history.append({"prompt": prompt, "response": response})
        else:
            raise ValueError("Unexpected response format")
        return response

    def __call__(
        self,
        prompt: Optional[str],
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Request completions from the Unify API."""
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        n: int = kwargs.get("n") or 1
        skip: int = kwargs.get("skip") or 0
        self.request(prompt, **kwargs)
        return self.inspect_history(n=n, skip=skip)
