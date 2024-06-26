


from typing import AsyncGenerator, Dict, Generator, List, Optional, Union

import openai
import requests
from unify.exceptions import BadRequestError, UnifyError, status_error_map
from unify.utils import (  # noqa:WPS450
    _available_dynamic_modes,
    _validate_api_key,
    _validate_endpoint,
)


class Unify:

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:  # noqa: DAR101, DAR401
    
        """Initialize the Unify client.

            Args:
                endpoint (str, optional): Endpoint name in OpenAI API format:
                    <uploaded_by>/<model_name>@<provider_name>
                    Defaults to None.

                model (str, optional): Name of the model. If None,
                endpoint must be provided.

                provider (str, optional): Name of the provider. If None,
                endpoint must be provided.

                api_key (str, optional): API key for accessing the Unify API.
                    If None, it attempts to retrieve the API key from the
                    environment variable UNIFY_KEY.
                    Defaults to None.

            Raises:
                UnifyError: If the API key is missing.
        """
        self._api_key = _validate_api_key(api_key)
        self._endpoint, self._model, self._provider = _validate_endpoint(  # noqa:WPS414
            endpoint,
            model,
            provider,
        )
        try:
            self.client = openai.OpenAI(
                base_url="https://api.unify.ai/v0/",
                api_key=self._api_key,
            )
        except openai.OpenAIError as e:
            raise UnifyError(f"Failed to initialize Unify client: {str(e)}")

    @property
    def model(self) -> str:
        """
        Get the model name.  # noqa: DAR201.

        Returns:
            str: The model name.
        """
        return self._model

    def set_model(self, value: str) -> None:
        """
        Set the model name.  # noqa: DAR101.

        Args:
            value (str): The model name.
        """
        self._model = value
        if self._provider:
            self._endpoint = "@".join([value, self._provider])
        else:
            mode = self._endpoint.split("@")[1]
            self._endpoint = "@".join([value, mode])

    @property
    def provider(self) -> Optional[str]:
        """
        Get the provider name.  # noqa :DAR201.

        Returns:
            str: The provider name.
        """
        return self._provider

    def set_provider(self, value: str) -> None:
        """
        Set the provider name.  # noqa: DAR101.

        Args:
            value (str): The provider name.
        """
        self._provider = value
        self._endpoint = "@".join([self._model, value])

    @property
    def endpoint(self) -> str:
        """
        Get the endpoint name.  # noqa: DAR201.

        Returns:
            str: The endpoint name.
        """
        return self._endpoint

    def set_endpoint(self, value: str) -> None:
        """
        Set the model name.  # noqa: DAR101.

        Args:
            value (str): The endpoint name.
        """
        self._endpoint = value
        self._model, self._provider = value.split("@")  # noqa: WPS414
        if self._provider in _available_dynamic_modes:
            self._provider = None

    def generate(  # noqa: WPS234, WPS211
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[Generator[str, None, None], str]:  # noqa: DAR101, DAR201, DAR401
        """Generate content using the Unify API.

        Args:
            user_prompt (Optional[str]): A string containing the user prompt.
            If provided, messages must be None.

            system_prompt (Optional[str]): An optional string containing the
            system prompt.

            messages (List[Dict[str, str]]): A list of dictionaries containing the
            conversation history. If provided, user_prompt must be None.

            max_tokens (Optional[int]): The max number of output tokens.
            Defaults to the provider's default max_tokens when the value is None.

            temperature (Optional[float]):  What sampling temperature to use, between 0 and 2. 
            Higher values like 0.8 will make the output more random, 
            while lower values like 0.2 will make it more focused and deterministic.
            Defaults to the provider's default max_tokens when the value is None.

            stop (Optional[List[str]]): Up to 4 sequences where the API will stop generating further tokens.

            stream (bool): If True, generates content as a stream.
            If False, generates content as a single response.
            Defaults to False.

        Returns:
            Union[Generator[str, None, None], str]: If stream is True,
             returns a generator yielding chunks of content.
             If stream is False, returns a single string response.

        Raises:
            UnifyError: If an error occurs during content generation.
        """
        contents = []
        if system_prompt:
            contents.append({"role": "system", "content": system_prompt})
        if user_prompt:
            contents.append({"role": "user", "content": user_prompt})
        elif messages:
            contents.extend(messages)
        else:
            raise UnifyError("You must provider either the user_prompt or messages!")

        if stream:
            return self._generate_stream(contents, self._endpoint,
                                          max_tokens=max_tokens,
                                          temperature=temperature,
                                          stop=stop)
        return self._generate_non_stream(contents, self._endpoint,
                                          max_tokens=max_tokens,
                                          temperature=temperature,
                                          stop=stop)

    def get_credit_balance(self) -> float:
        # noqa: DAR201, DAR401
        """
        Get the remaining credits left on your account.

        Returns:
            int or None: The remaining credits on the account
            if successful, otherwise None.
        Raises:
            BadRequestError: If there was an HTTP error.
            ValueError: If there was an error parsing the JSON response.
        """
        url = "https://api.unify.ai/v0/get_credits"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()["credits"]
        except requests.RequestException as e:
            raise BadRequestError("There was an error with the request.") from e
        except (KeyError, ValueError) as e:
            raise ValueError("Error parsing JSON response.") from e

    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        try:
            chat_completion = self.client.chat.completions.create(
                model=endpoint,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
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
        messages: List[Dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                model=endpoint,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=False,
            )
            self.set_provider(
                chat_completion.model.split(  # type: ignore[union-attr]
                    "@",
                )[-1]
            )

            return chat_completion.choices[0].message.content.strip(" ")  # type: ignore # noqa: E501, WPS219
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None


class AsyncUnify:
    """Class for interacting asynchronously with the Unify API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:  # noqa:DAR101, DAR401
        """Initialize the AsyncUnify client.

        Args:
            endpoint (str, optional): Endpoint name in OpenAI API format:
            <uploaded_by>/<model_name>@<provider_name>
            Defaults to None.

            model (str, optional): Name of the model. If None,
            endpoint must be provided.

            provider (str, optional): Name of the provider. If None,
            endpoint must be provided.

            api_key (str, optional): API key for accessing the Unify API.
            If None, it attempts to retrieve the API key from
            the environment variable UNIFY_KEY.
            Defaults to None.

        Raises:
            UnifyError: If the API key is missing.
        """
        self._api_key = _validate_api_key(api_key)
        self._endpoint, self._model, self._provider = (  # noqa: WPS414
            _validate_endpoint(
                endpoint,
                model,
                provider,
            )
        )
        try:
            self.client = openai.AsyncOpenAI(
                base_url="https://api.unify.ai/v0/",
                api_key=self._api_key,
            )
        except openai.APIStatusError as e:
            raise UnifyError(f"Failed to initialize Unify client: {str(e)}")

    @property
    def model(self) -> str:
        """
        Get the model name.  # noqa: DAR201.

        Returns:
            str: The model name.
        """
        return self._model

    def set_model(self, value: str) -> None:
        """
        Set the model name.  # noqa: DAR101.

        Args:
            value (str): The model name.
        """
        self._model = value
        if self._provider:
            self._endpoint = "@".join([value, self._provider])
        else:
            mode = self._endpoint.split("@")[1]
            self._endpoint = "@".join([value, mode])

    @property
    def provider(self) -> Optional[str]:
        """
        Get the provider name.  # noqa :DAR201.

        Returns:
            str: The provider name.
        """
        return self._provider

    def set_provider(self, value: str) -> None:
        """
        Set the provider name.  # noqa: DAR101.

        Args:
            value (str): The provider name.
        """
        self._provider = value
        self._endpoint = "@".join([self._model, value])

    @property
    def endpoint(self) -> str:
        """
        Get the endpoint name.  # noqa: DAR201.

        Returns:
            str: The endpoint name.
        """
        return self._endpoint

    def set_endpoint(self, value: str) -> None:
        """
        Set the model name.  # noqa: DAR101.

        Args:
            value (str): The endpoint name.
        """
        self._endpoint = value
        self._model, self._provider = value.split("@")  # noqa: WPS414
        if self._provider in _available_dynamic_modes:
            self._provider = None

    def get_credit_balance(self) -> Optional[int]:
        # noqa: DAR201, DAR401
        """
        Get the remaining credits left on your account.

        Returns:
            int or None: The remaining credits on the account
            if successful, otherwise None.

        Raises:
            BadRequestError: If there was an HTTP error.
            ValueError: If there was an error parsing the JSON response.
        """
        url = "https://api.unify.ai/v0/get_credits"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()["credits"]
        except requests.RequestException as e:
            raise BadRequestError("There was an error with the request.") from e
        except (KeyError, ValueError) as e:
            raise ValueError("Error parsing JSON response.") from e

    async def generate(  # noqa: WPS234, WPS211
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[AsyncGenerator[str, None], str]:  # noqa: DAR101, DAR201, DAR401
        """Generate content asynchronously using the Unify API.

        Args:
            user_prompt (Optional[str]): A string containing the user prompt.
            If provided, messages must be None.

            system_prompt (Optional[str]): An optional string containing the
            system prompt.

            messages (List[Dict[str, str]]): A list of dictionaries containing the
            conversation history. If provided, user_prompt must be None.

            max_tokens (Optional[int]): The max number of output tokens, defaults
            to the provider's default max_tokens when the value is None.

            temperature (Optional[float]):  What sampling temperature to use, between 0 and 2. 
            Higher values like 0.8 will make the output more random, 
            while lower values like 0.2 will make it more focused and deterministic.
            Defaults to the provider's default max_tokens when the value is None.

            stop (Optional[List[str]]): Up to 4 sequences where the API will stop generating further tokens.

            stream (bool): If True, generates content as a stream.
            If False, generates content as a single response.
            Defaults to False.

        Returns:
            Union[AsyncGenerator[str, None], List[str]]: If stream is True,
            returns an asynchronous generator yielding chunks of content.
            If stream is False, returns a list of string responses.

        Raises:
            UnifyError: If an error occurs during content generation.
        """
        contents = []
        if system_prompt:
            contents.append({"role": "system", "content": system_prompt})

        if user_prompt:
            contents.append({"role": "user", "content": user_prompt})
        elif messages:
            contents.extend(messages)
        else:
            raise UnifyError("You must provide either the user_prompt or messages!")

        if stream:
            return self._generate_stream(contents, self._endpoint, max_tokens=max_tokens, stop=stop, temperature=temperature)
        return await self._generate_non_stream(contents, self._endpoint, max_tokens=max_tokens, stop=stop, temperature=temperature)

    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            async_stream = await self.client.chat.completions.create(
                model=endpoint,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=True,
            )
            async for chunk in async_stream:  # type: ignore[union-attr]
                self.set_provider(chunk.model.split("@")[-1])
                yield chunk.choices[0].delta.content or ""
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None

    async def _generate_non_stream(
        self,
        messages: List[Dict[str, str]],
        endpoint: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        try:
            async_response = await self.client.chat.completions.create(
                model=endpoint,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=False,
            )
            self.set_provider(async_response.model.split("@")[-1])  # type: ignore
            return async_response.choices[0].message.content.strip(" ")  # type: ignore # noqa: E501, WPS219
        except openai.APIStatusError as e:
            raise status_error_map[e.status_code](e.message) from None
            