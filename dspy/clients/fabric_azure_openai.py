"""Microsoft Fabric Azure OpenAI integration for DSPy.

This module provides a custom LM class for using Azure OpenAI models within
Microsoft Fabric notebooks. It handles authentication and endpoint configuration
automatically using Fabric's built-in service discovery and token utilities.

Note: This class only works within a Microsoft Fabric environment.
"""

from typing import Any, ClassVar

import requests

from dspy.clients.base_lm import BaseLM


class FabricAzureOpenAI(BaseLM):
    """Language model client for Azure OpenAI in Microsoft Fabric.

    This class provides integration with Azure OpenAI models deployed in Microsoft Fabric.
    It automatically handles authentication and endpoint configuration using Fabric's
    service discovery and token utilities.

    Note:
        This class requires the following packages available in Microsoft Fabric:
        - synapse.ml.fabric.service_discovery
        - synapse.ml.fabric.token_utils

    Supported models:
        - gpt-5 (reasoning model)
        - gpt-4.1
        - gpt-4.1-mini

    Args:
        deployment_name: The name of the Azure OpenAI deployment to use.
            Must be one of: gpt-5, gpt-4.1, gpt-4.1-mini
        model_type: The type of the model. Defaults to "chat".
        temperature: The sampling temperature. Defaults to 0.0.
        max_tokens: Maximum number of tokens to generate. Defaults to 4000.
        cache: Whether to cache responses. Defaults to True.
        **kwargs: Additional arguments passed to the base class.

    Example:
        ```python
        import dspy
        from dspy.clients import FabricAzureOpenAI

        # In a Microsoft Fabric notebook
        lm = FabricAzureOpenAI(deployment_name="gpt-5")
        dspy.configure(lm=lm)

        # Use with DSPy modules
        predictor = dspy.Predict("question -> answer")
        result = predictor(question="What is DSPy?")
        ```
    """

    # Supported models in Microsoft Fabric
    SUPPORTED_MODELS: ClassVar[set[str]] = {"gpt-5", "gpt-4.1", "gpt-4.1-mini"}
    REASONING_MODELS: ClassVar[set[str]] = {"gpt-5"}

    def __init__(
        self,
        deployment_name: str = "gpt-5",
        model_type: str = "chat",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        cache: bool = True,
        **kwargs,
    ):
        """Initialize the FabricAzureOpenAI client.

        Args:
            deployment_name: The Azure OpenAI deployment name.
            model_type: The type of model ("chat" or "text").
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            cache: Whether to enable caching.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If deployment_name is not a supported model.
        """
        # Validate model support
        if deployment_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{deployment_name}' is not supported in Microsoft Fabric. "
                f"Supported models are: {', '.join(sorted(self.SUPPORTED_MODELS))}. "
                f"For more information, see: "
                f"https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview"
            )

        self.deployment_name = deployment_name
        super().__init__(
            model=deployment_name,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        # Check if this is a reasoning model
        self.is_reasoning_model = deployment_name in self.REASONING_MODELS

    def _get_fabric_config(self):
        """Get Fabric environment configuration and auth header.

        Returns:
            tuple: (fabric_env_config, auth_header)

        Raises:
            ImportError: If Fabric SDK packages are not available.
        """
        try:
            from synapse.ml.fabric.service_discovery import get_fabric_env_config
            from synapse.ml.fabric.token_utils import TokenUtils
        except ImportError as e:
            raise ImportError(
                "Microsoft Fabric SDK packages are required to use FabricAzureOpenAI. "
                "These packages are only available in Microsoft Fabric notebooks. "
                "Please ensure you are running in a Fabric environment."
            ) from e

        fabric_env_config = get_fabric_env_config().fabric_env_config
        auth_header = TokenUtils().get_openai_auth_header()
        return fabric_env_config, auth_header

    def _prepare_messages(self, prompt: str | list | None, messages: list[dict[str, Any]] | None) -> list[dict]:
        """Prepare messages for the API request.

        Args:
            prompt: A string prompt or list of messages.
            messages: A list of message dictionaries.

        Returns:
            list: Formatted messages for the API.
        """
        if messages is not None:
            return messages
        elif prompt is not None:
            if isinstance(prompt, str):
                return [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                return prompt
            else:
                return [{"role": "user", "content": str(prompt)}]
        else:
            return []

    def _build_payload(self, messages: list[dict], **kwargs) -> dict:
        """Build the request payload based on model type.

        Args:
            messages: The formatted messages.
            **kwargs: Additional parameters like max_tokens, temperature, etc.

        Returns:
            dict: The request payload.
        """
        max_tokens_value = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 4000))

        # Build payload based on model type
        payload = {"messages": messages}

        if self.is_reasoning_model:
            # Reasoning models use max_completion_tokens and don't support temperature
            payload["max_completion_tokens"] = max_tokens_value
            # Don't include temperature or n for reasoning models
        else:
            # Standard models use max_tokens and support temperature
            payload["max_tokens"] = max_tokens_value
            payload["temperature"] = kwargs.get("temperature", self.kwargs.get("temperature", 0.0))
            payload["n"] = kwargs.get("n", 1)

        return payload

    def _make_request(self, payload: dict) -> list[str]:
        """Make the API request to Azure OpenAI.

        Args:
            payload: The request payload.

        Returns:
            list: List of response contents.

        Raises:
            Exception: If the API call fails.
        """
        fabric_env_config, auth_header = self._get_fabric_config()

        url = (
            f"{fabric_env_config.ml_workload_endpoint}cognitive/openai/openai/deployments/"
            f"{self.deployment_name}/chat/completions?api-version=2025-04-01-preview"
        )
        headers = {"Authorization": auth_header, "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            response_data = response.json()
            return [choice["message"]["content"] for choice in response_data.get("choices", [])]
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")

    def basic_request(self, prompt: str | list | None = None, **kwargs) -> list[str]:
        """Make a basic request to the Azure OpenAI API.

        Args:
            prompt: The prompt string or list of messages.
            **kwargs: Additional parameters for the request.

        Returns:
            list: List of generated response strings.
        """
        messages = self._prepare_messages(prompt, None)
        payload = self._build_payload(messages, **kwargs)
        return self._make_request(payload)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Forward pass for the language model.

        This method is required by BaseLM and must return a response in OpenAI format.

        Args:
            prompt: Optional string prompt.
            messages: Optional list of message dictionaries.
            **kwargs: Additional parameters.

        Returns:
            A response object compatible with OpenAI's response format.

        Raises:
            ValueError: If neither prompt nor messages is provided.
        """
        if prompt is None and messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        # Prepare messages
        formatted_messages = self._prepare_messages(prompt, messages)

        # Build payload
        payload = self._build_payload(formatted_messages, **kwargs)

        # Make request
        response_contents = self._make_request(payload)

        # Convert to OpenAI-compatible format
        # We need to return a response object that looks like OpenAI's ChatCompletion
        from types import SimpleNamespace

        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content, role="assistant"),
                finish_reason="stop",
                index=i,
            )
            for i, content in enumerate(response_contents)
        ]

        # Create a minimal response object
        response = SimpleNamespace(
            choices=choices,
            model=self.deployment_name,
            usage=SimpleNamespace(
                prompt_tokens=0,  # Fabric API doesn't return token counts
                completion_tokens=0,
                total_tokens=0,
            ),
        )

        return response

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[str]:
        """Call the language model.

        This method provides a simpler interface that returns just the text outputs.

        Args:
            prompt: Optional string prompt.
            messages: Optional list of message dictionaries.
            **kwargs: Additional parameters.

        Returns:
            list: List of generated response strings.

        Raises:
            ValueError: If neither prompt nor messages is provided.
        """
        if messages is not None:
            return self.basic_request(messages, **kwargs)
        elif prompt is not None:
            return self.basic_request(prompt, **kwargs)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
