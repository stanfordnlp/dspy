import os
from collections.abc import Generator
from typing import Any, Literal, Optional

import openai
import unify
from unify import ChatBot, Unify
from unify.exceptions import status_error_map

from dsp.modules.lm import LM

try:
    import openai.error
    from openai.openai_object import OpenAIObject

    ERRORS = (openai.error.RateLimitError,)
except Exception:
    ERRORS = (openai.RateLimitError,)
    OpenAIObject = dict


class ChatBot(ChatBot):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        inp: str,
    ) -> Generator[str, None, None]:
        """Processes the user input to generate AI response."""
        self._update_message_history(role="user", content=inp)
        return self._client.generate(
            messages=self._message_history,
            stream=True,
        )


class UnifyClient(Unify):
    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        endpoint: str = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = 1024,
        stream: Optional[bool] = True,
        **kwargs,
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.endpoint if self.endpoint else endpoint,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            stream=stream,
            name="Unify-generation",
            metadata={
                "model": self.model,
                "provider": self.provider,  # todo: update trace metadata after call (see set_provider)
                "endpoint": self.endpoint,
            },
            **kwargs,
        )

    def _generate_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            chat_completion = self.generate_completion(
                messages=messages,
                endpoint=endpoint,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            for chunk in chat_completion:
                content = chunk.choices[0].delta.content  # type: ignore[union-attr]
                self.set_provider(chunk.model.split("@")[-1])  # type: ignore[union-attr]
                if content is not None:
                    yield content
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None

    def _generate_non_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        print(f"_GENERATE_NON_STREAM: {messages}")  # type: ignore # noqa: T201
        try:
            chat_completion = self.generate_completion(
                messages=messages,
                endpoint=endpoint,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )

            self.set_provider(
                chat_completion.model.split(  # type: ignore[union-attr]
                    "@",
                )[-1],
            )

            return chat_completion.choices[0].message.content.strip(" ")  # type: ignore # noqa: E501, WPS219
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None


class UnifyDSP(LM):
    def __init__(
        self,
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        api_key=None,
        **kwargs,  # Added to accept additional keyword arguments
    ):
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_provider: Literal["unify"] = "unify"
        self.api_base = "https://api.unify.ai/v0"
        self.model = model
        self.provider = provider
        super().__init__(model=self.endpoint)
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

    def __call__(
        self,
        prompt: Optional[str] = "",
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from the model called by unify.

        Args:
            prompt (str): prompt to send to unify
            only_completed (bool, optional): return only completed responses
                and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using
                the returned probabilities. Defaults to False.
            **kwargs (Any): metadata passed to the model.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        assert prompt, "for now"
        assert kwargs, "for now"

        return self.generate(user_prompt=prompt, system_prompt=self.system_prompt, **self.kwargs)

    def generate(
        self,
        prompt: Optional[str] = "",
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from the model called by unify.

        Args:
            prompt (str): prompt to send to unify
            only_completed (bool, optional): return only completed responses
                and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using
                the returned probabilities. Defaults to False.
            **kwargs (Any): metadata passed to the model.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        assert prompt, "for now"
        assert kwargs, "for now"
        unify_client = UnifyClient(
            endpoint=self.endpoint,
            api_key=self.api_key,
        )
        print(type(prompt), type(self.system_prompt))  # type: ignore # noqa: T201
        print(f"KWARGS: {kwargs}, {type(kwargs)}")  # type: ignore # noqa: T201
        return unify_client.generate(user_prompt=prompt, system_prompt=self.system_prompt, **kwargs)

    def get_credit_balance(self) -> Any:
        return UnifyClient(api_key=self.api_key, endpoint=self.endpoint).get_credit_balance()

    @classmethod
    def list_available_models(cls) -> list:
        return unify.list_models()

    def basic_request(self, prompt: str, **kwargs) -> Any:
        """Send request to the Unify AI API. This method is required by the LM base class."""
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

        # Call the generate method
        return self._call_generate(settings_dict)

    def _call_generate(self, settings_dict):
        """Call the generate method from the unify client."""
        try:
            unify_client = UnifyClient(endpoint=self.endpoint, api_key=self.api_key)
            return unify_client.generate_completion(settings=settings_dict, api_key=self.api_key)
        except Exception as e:
            return f"An error occurred while calling the generate method: {e}"


# Usage example
if __name__ == "__main__":
    # Initialize the UnifyAI instance with a specific model and fallback
    unify_lm = UnifyClient(endpoint="llama-3-8b-chat@fireworks-ai->gpt-3.5-turbo@openai")

    # Check credit balance
    credit_balance = unify_lm.get_credit_balance()
    print(f"Current credit balance: {credit_balance}")  # type: ignore # noqa: T201

    # List available models
    print("Available models:")  # type: ignore # noqa: T201
    models = unify_lm.list_available_models()
    for model in models:
        print(f"- {model}")  # type: ignore # noqa: T201

    # Generate a response
    prompt = "Translate 'Hello, world!' to French."
    print(f"\nGenerating response for prompt: '{prompt}'")  # type: ignore # noqa: T201
    responses = unify_lm.generate(prompt, max_tokens=50, temperature=0.7, n=1)

    if responses:
        print("Generated response:")  # type: ignore # noqa: T201
        for response in responses:
            print(response)  # type: ignore # noqa: T201
    else:
        print("Failed to generate any responses.")  # type: ignore # noqa: T201

    # Example with router
    router_lm = UnifyClient(endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03")
    print("\nUsing router for generation:")  # type: ignore # noqa: T201
    router_responses = router_lm.generate("What is the capital of France?", max_tokens=50)
    if router_responses:
        print("Router-generated response:")  # type: ignore # noqa: T201
        for response in router_responses:
            print(response)  # type: ignore # noqa: T201
    else:
        print("Router failed to generate any responses.")  # type: ignore # noqa: T201
