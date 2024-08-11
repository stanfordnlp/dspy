from typing import Any

# Pydantic data model for the LLM class
import litellm
from litellm.types.utils import Choices, ModelResponse, Usage

from dspy.utils.logging import logger
from dsp.modules.schemas import (
    ChatMessage,
    DSPyModelResponse,
    LLMParams,
)
from dsp.modules.lm_helper import LLMHelper

# Use this for development testing
# litellm.set_verbose = True


class LLM(LLMHelper):
    """Wrapper around the LLM API.

    Usage:
    ```python
    import dspy
    from dspy import LLMModelParams

    llm_params = LLMModelParams(
        model="gpt-4o-mini", api_key=OPENAI_APIKEY
    )
    openai_llm = dspy.LLM(llm_params)
    openai_llm("Hello, how are you?")
    ```
    """

    llm_params: LLMParams
    history: list[dict[str, Any]] = []

    def __init__(
        self,
        llm_params: LLMParams,
    ):
        super().__init__()
        self.llm_params = llm_params

    def basic_request(self, prompt: str, **kwargs) -> ModelResponse:
        self.update_messages_with_prompt(prompt)

        response: ModelResponse = litellm.completion(
            **self.llm_params.to_json(), **kwargs
        )

        self.history.append(
            {
                "prompt": prompt,
                "response": response.to_dict(),
                "raw_kwargs": kwargs,
                "kwargs": self.llm_params.to_json(ignore_sensitive=True),
            }
        )

        return response

    # TODO: enable caching
    def request(self, prompt: str, **kwargs) -> ModelResponse:
        return self.basic_request(prompt, **kwargs)

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
                dspy_choices[-1]["logprobs"] = (
                    choice.logprobs if choice.logprobs else None
                )

        return dspy_choices

    def update_messages_with_prompt(self, prompt: str):
        """Updates the messages with the prompt."""
        self.llm_params.prompt = prompt
        if not self.llm_params.messages:
            self.llm_params.messages = []
        self.llm_params.messages.append(
            ChatMessage(role="user", content=prompt)
        )

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
        response = self.request(prompt, **kwargs)

        self.log_usage(response)

        choices = (
            self.filter_only_completed(response.choices)
            if self.llm_params.only_completed
            else response.choices
        )

        choices = self.transform_choices_to_dspy_model_response(
            choices, self.llm_params.logprobs
        )

        return choices

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.llm_params.to_json(), **kwargs}

        return self.__class__(llm_params=LLMParams(**kwargs), **kwargs)
