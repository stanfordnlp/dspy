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
    """
    Extends the Unify ChatBot to handle message generation and interaction.

    This class provides methods to process user input and generate AI responses
    using the Unify client.

    Methods:
        __init__(endpoint: Optional[str] = None, model: Optional[str] = None,
                 provider: Optional[str] = None, api_key: Optional[str] = None) -> None:
            Initializes the ChatBot instance with the specified parameters.

        generate_completion(inp: str) -> Generator[str, None, None]:
            Processes the user input to generate an AI response.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the ChatBot instance with the specified parameters.

        Args:
            endpoint (Optional[str]): The API endpoint to connect to.
            model (Optional[str]): The model to use for generating responses.
            provider (Optional[str]): The provider for the AI model.
            api_key (Optional[str]): The API key for authenticating with the service.
        """
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        inp: str,
    ) -> Generator[str, None, None]:
        """
        Processes the user input to generate an AI response.

        Args:
            inp (str): The input from the user.

        Yields:
            str: Generated AI responses as a generator.
        """
        self._update_message_history(role="user", content=inp)
        return self._client.generate(
            messages=self._message_history,
            stream=True,
        )



class UnifyClient(Unify):
    """
    Extends the Unify class to manage interactions with Unify AI services.

    This class provides methods for generating AI completions and handling
    communication with the Unify AI platform.

    Methods:
        __init__(endpoint: Optional[str] = None, model: Optional[str] = None,
                 provider: Optional[str] = None, api_key: Optional[str] = None) -> None:
            Initializes the UnifyClient instance with the specified parameters.

        generate_completion(endpoint: str = None, messages: Optional[list[dict[str, str]]] = None,
                            max_tokens: Optional[int] = 1024, stream: Optional[bool] = True,
                            **kwargs) -> Any:
            Generates AI completions using Unify's API.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the UnifyClient instance with the specified parameters.

        Args:
            endpoint (Optional[str]): The API endpoint to connect to.
            model (Optional[str]): The model to use for generating responses.
            provider (Optional[str]): The provider for the AI model.
            api_key (Optional[str]): The API key for authenticating with the service.
        """
        super().__init__(endpoint, model, provider, api_key)

    def generate_completion(
        self,
        endpoint: str = None,
        messages: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = 1024,
        stream: Optional[bool] = True,
        **kwargs,
    ) -> Any:
        """
        Generates AI completions using Unify's API.

        Args:
            endpoint (str): The API endpoint to use for generating completions.
            messages (Optional[list[dict[str, str]]]): A list of message dictionaries
                representing the conversation history.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the response.
            **kwargs: Additional keyword arguments for the API request.

        Returns:
            Any: The generated completion response from the Unify API.
        """
        return self.client.chat.completions.create(
            model=self.endpoint if self.endpoint else endpoint,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )


    def _generate_stream(
        self,
        messages: list[dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
    Generates AI completions in stream mode using Unify's API.

    This method processes the provided messages and streams the AI-generated
    responses in chunks.

    Args:
        messages (list[dict[str, str]]): A list of message dictionaries representing
            the conversation history.
        endpoint (str): The API endpoint to use for generating completions.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        **kwargs: Additional keyword arguments for the API request.

    Yields:
        str: Chunks of the generated AI response.

    Raises:
        openai.APIStatusError: If an API status error occurs, it raises the appropriate
            status error mapped from Unify's API.
    """
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
        """
    Generates AI completions in non-stream mode using Unify's API.

    This method processes the provided messages and returns the complete AI-generated
    response as a single string.

    Args:
        messages (list[dict[str, str]]): A list of message dictionaries representing
            the conversation history.
        endpoint (str): The API endpoint to use for generating completions.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
        **kwargs: Additional keyword arguments for the API request.

    Returns:
        str: The complete generated AI response.

    Raises:
        openai.APIStatusError: If an API status error occurs, it raises the appropriate
            status error mapped from Unify's API.
    """
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
    """
    UnifyDSP class integrates Unify AI's capabilities with a custom LM (Language Model) implementation.

    This class is designed to interface with Unify AI's models and provides methods
    for generating AI completions, managing conversation history, and configuring model parameters.

    Attributes:
        endpoint (str): The API endpoint for the Unify AI model.
        api_key (str): The API key for authenticating with the Unify AI service.
        api_provider (Literal["unify"]): The API provider, which is set to "unify".
        api_base (str): The base URL for the Unify AI API.
        model (Optional[str]): The model to use for generating responses.
        provider (Optional[str]): The provider for the AI model.
        system_prompt (Optional[str]): The system prompt to use for the AI model.
        model_type (Literal["chat", "text"]): The type of model to use ("chat" or "text").
        kwargs (dict): Additional keyword arguments for the model configuration.
        history (list[dict[str, Any]]): The conversation history.
    """
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
        """
        Initializes the UnifyDSP instance with the specified parameters.

        Args:
            endpoint (str): The API endpoint for the Unify AI model. Defaults to a specific router endpoint.
            model (Optional[str]): The model to use for generating responses.
            provider (Optional[str]): The provider for the AI model.
            model_type (Literal["chat", "text"]): The type of model to use ("chat" or "text"). Defaults to "chat".
            system_prompt (Optional[str]): The system prompt to use for the AI model.
            api_key (Optional[str]): The API key for authenticating with the Unify AI service. If not provided, it will be retrieved from the environment variable "UNIFY_API_KEY".
            **kwargs: Additional keyword arguments for the model configuration.
        """
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
        return unify_client.generate(user_prompt=prompt, system_prompt=self.system_prompt, **kwargs)

    def get_credit_balance(self) -> Any:
        """
    Retrieves the current credit balance for the Unify AI account.

    This method creates a UnifyClient instance with the stored API key and endpoint
    and calls its get_credit_balance method.

    Returns:
        Any: The current credit balance.
    """
        return UnifyClient(api_key=self.api_key, endpoint=self.endpoint).get_credit_balance()

    @classmethod
    def list_available_models(cls) -> list:
        """
    Lists all available models from Unify AI.

    This method calls the Unify API to retrieve a list of available models.

    Returns:
        list: A list of available models.
    """
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
