import os
from typing import Any, Optional, Union

import dspy

# Pydantic data model for the LLM class
import litellm
from litellm.types.utils import Choices, ModelResponse, Usage
from litellm.caching import Cache

from dspy.utils.logging import logger
from dsp.modules.schemas import (
    ChatMessage,
    DSPyModelResponse,
    LLMModelParams,
    LiteLLMParams,
)
from dsp.modules.lm_helper import LLMHelper

litellm.cache = Cache(disk_cache_dir=".dspy_cache", type="disk")

# Use this for development testing
litellm.set_verbose = False

class LLM(LLMHelper):
    """Wrapper around the LLM API.

    Usage:
    ```python
    import dspy
    from dspy import LLMParams

    openai_llm = dspy.LLM(model="gpt-4o-mini", provider="openai")
    openai_llm("Hello, how are you?")
    ```
    """

    llm_params: LLMModelParams
    request_params: LiteLLMParams
    history: list[dict[str, Any]] = []

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        n: int = 1,
        stop: Optional[Union[str, list[str]]] = None,
        safety_settings: Optional[dict[str, Any]] = None,
        logit_bias: Optional[dict[str, float]] = None,
        user: Optional[str] = None,
        response_format: Optional[str] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = False,
        top_logprobs: Optional[int] = None,
        extra_headers: Optional[dict[str, str]] = None,
        # Vertex specific params
        vertex_credentials: Optional[str] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        # AWS specific params
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        # Cloudflare specific params
        cloudflare_api_key: Optional[str] = None,
        cloudflare_account_id: Optional[str] = None,
        # Anyscale specific params
        anyscale_api_key: Optional[str] = None,
        # TogetherAI specific params
        togetherai_api_key: Optional[str] = None,
        # Azure OpenAI specific params
        project_id: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        # Litellm specific params
        num_retries: Optional[int] = None,
        context_window_fallback_dict: Optional[dict[str, Any]] = None,
        fallbacks: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        # All model specific optimization parameters
        self.llm_params = LLMModelParams(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=n,
            stop=stop,
            safety_settings=safety_settings,
            logit_bias=logit_bias,
            user=user,
            response_format=response_format,
            seed=seed,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        # Required to make an API call with the provider
        self.request_params = LiteLLMParams(
            api_key=api_key,
            provider=provider,
            extra_headers=extra_headers,
            num_retries=num_retries,
            context_window_fallback_dict=context_window_fallback_dict,
            fallbacks=fallbacks,
            metadata=metadata,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
            cloudflare_api_key=cloudflare_api_key,
            cloudflare_account_id=cloudflare_account_id,
            anyscale_api_key=anyscale_api_key,
            togetherai_api_key=togetherai_api_key,
            project_id=project_id,
            api_base=api_base,
            api_version=api_version,
        )


    def basic_request(self, model_params: dict[str, Any], request_params: dict[str, Any], **kwargs) -> ModelResponse:

        response: ModelResponse = litellm.completion(**model_params, **request_params, **kwargs, drop_params=True)

        self.history.append(
            {
                "prompt": model_params["messages"][-1]["content"],
                "response": response.to_dict(),
                "raw_kwargs": kwargs,
                "kwargs": model_params,
            }
        )

        return response

    def request(self, model_params: dict[str, Any], request_params: dict[str, Any], **kwargs) -> ModelResponse:
        enable_cache = dspy.settings.enable_cache
        if "enable_cache" in kwargs:
            # caching passed in by request takes precedence over settings
            enable_cache = kwargs["cache"]
        if enable_cache:
            kwargs['cache'] = {"no-cache": False, "no-store": False}
        else:
            kwargs['cache'] = {"no-cache": True, "no-store": True}
        return self.basic_request(model_params, request_params, **kwargs)

    def log_usage(self, response: ModelResponse):
        """Log the total tokens from the OpenAI API response."""
        usage_data: Usage = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logger.debug("OpenAI Response Token Usage: %s", total_tokens)

    def filter_only_completed(self, choices: list[Choices]) -> list[Choices]:
        """Filters out incomplete completions by checking if the finish_reason is not 'length'.
        Returns the filtered list of choices only if there are any, otherwise returns the original list.
        """
        # TODO: This seems provider specific:
        # Calude uses response.stop_reason == "max_tokens", refer previous code.
        filtered_choices = [c for c in choices if c.finish_reason != "length"]
        if len(filtered_choices):
            return filtered_choices
        return choices

    def get_text_from_choice(self, choice: Choices) -> str:
        """Returns the text from the choice."""
        return choice.message.content

    def transform_choices_to_dspy_model_response(
        self, choices: list[Choices], add_logprobs: bool = False
    ) -> list[DSPyModelResponse]:
        """Transforms the choices to DSPyModelResponse."""
        dspy_choices: list[DSPyModelResponse] = []
        for choice in choices:
            # TODO: ideally we should return the choice object itself, which contains more information.
            dspy_choices.append(
                {
                    "text": self.get_text_from_choice(choice),
                }
            )
            if add_logprobs:
                # TODO: check if we can strong type this.
                dspy_choices[-1]["logprobs"] = choice.logprobs if choice.logprobs else None

        return dspy_choices

    def _update_messages_with_prompt(self, prompt: str):
        """Updates the messages with the prompt."""
        self.llm_params.prompt = prompt
        if not self.llm_params.messages:
            self.llm_params.messages = []
        self.llm_params.messages.append(ChatMessage(role="user", content=prompt))

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        **kwargs,
    ) -> list[DSPyModelResponse]:
        """Retrieves completions from LLM."""

        self.llm_params.only_completed = only_completed

        assert self.llm_params.only_completed, "for now"
        assert self.llm_params.return_sorted is False, "for now"

        # TODO: I don't really like that the prompt is string instead of messages. We should refactor this.
        # Option: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chat-markup-language
        self._update_messages_with_prompt(prompt)

        response = self.request(self.llm_params.get_only_valid_model_params(), self.request_params.to_json(), **kwargs)

        self.log_usage(response)

        choices = self.filter_only_completed(response.choices) if self.llm_params.only_completed else response.choices

        choices = self.transform_choices_to_dspy_model_response(choices, self.llm_params.logprobs)

        return choices

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.llm_params.to_json(exclude_none=True), **self.request_params.to_json(exclude_none=True), **kwargs}

        return self.__class__(**kwargs)
