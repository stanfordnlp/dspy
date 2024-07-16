from typing import Any, Optional

from unify.clients import Unify as UnifyClient

from dsp.modules.lm import LM


class Unify(LM):
    """A class to interact with the Unify AI API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        model_provider: Optional[str] = None,  # refearing to provider of the model
        stream: Optional[bool] = False,  # referring to unify router
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        n: int = 1,
        base_url: str = "https://api.unify.ai/v0",
        **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.model_provider = model_provider
        self.endpoint = endpoint
        self.stream = stream
        self.client = UnifyClient(
            api_key=self.api_key, endpoint=self.endpoint, model=self.model, provider=self.model_provider
        )
        super().__init__(model=self.endpoint)

        self.provider = "unify"
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
        self.kwargs["endpoint"] = self.endpoint
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Basic request to the Unify's API."""
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]

        settings_dict = {
            "endpoint": self.endpoint,
            "stream": self.stream,
            "messages": messages,
        }
        if self.system_prompt:
            settings_dict["messages"].insert(0, {"role": "system", "content": self.system_prompt})

        raw_response = self.client.generate(
            messages=settings_dict["messages"],
            stream=settings_dict["stream"],
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )

        # condition to handle both outputs stream =True or False from Unify generate method
        if isinstance(raw_response, str):
            response_content = raw_response
        else:
            response_content = "".join(chunk for chunk in raw_response)
        formated_response: dict = {"choices": [{"message": {"content": response_content}}]}  # response with choices

        history = {
            "prompt": prompt,
            "response": formated_response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }

        self.history.append(history)

        return formated_response

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
            completions.append(response["choices"][0]["message"]["content"])

        return completions
