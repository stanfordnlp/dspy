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
        **kwargs,
    ):
        self.api_key = api_key
        if endpoint:
            self.endpoint = endpoint
        elif model and model_provider:
            self.endpoint = "@".join([model, model_provider])

        self.client = UnifyClient(
            api_key=self.api_key,
            endpoint=self.endpoint,
        )

        super().__init__(model=self.endpoint)

        self.provider = "unify"
        self.stream = stream
        self.system_prompt = system_prompt
        self.kwargs = {
            "model": self.endpoint,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "n": n,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(
        self,
        prompt: str,
        **kwargs,
    ) -> Any:
        """Basic request to the Unify's API."""
        kwargs = {**self.kwargs, **kwargs}
        messages = [{"role": "user", "content": prompt}]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        raw_response = self.client.generate(
            messages=messages,
            stream=self.stream,
            temperature=kwargs["temperature"],
            max_tokens=kwargs["max_tokens"],
        )

        # condition to handle both outputs stream =True or False from Unify generate method
        response_content = raw_response if isinstance(raw_response, str) else "".join(raw_response)
        formated_response: dict = {"choices": [{"message": {"content": response_content}}]}  # response with choices

        history = {
            "prompt": prompt,
            "response": formated_response,
            "kwargs": kwargs,
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
