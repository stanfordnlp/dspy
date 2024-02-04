"""
A generalized AWS LLM.
"""

from abc import abstractmethod
import logging
from typing import Any, Literal
import json
from dsp.modules.lm import LM

# Heuristic translating number of chars to tokens
# ~4 chars = 1 token
CHARS2TOKENS: int = 4


class AWSLM(LM):
    """
    This class adds support for an AWS model
    """

    def __init__(
        self,
        model: str,
        region_name: str,
        service_name: str,
        max_new_tokens: int,
        truncate_long_prompts: bool = False,
        input_output_ratio: int = 3,
    ) -> None:
        """_summary_

        Args:

            service_name (str): Used in context of invoking the boto3 API.
            region_name (str, optional): The AWS region where this LM is hosted.
            model (str, optional): An LM name, e.g., a bedrock name or an AWS endpoint.
            max_new_tokens (int, optional): The maximum number of tokens to be sampled from the LM.
            input_output_ratio (int, optional): The rough size of the number of input tokens to output tokens in the worst case. Defaults to 3.
            temperature (float, optional): _description_. Defaults to 0.0.
            truncate_long_prompts (bool, optional): If True, remove extremely long inputs to context. Defaults to False.
        """
        super().__init__(model=model)
        # AWS doesn't have an equivalent of max_tokens so let's clarify
        # that the expected input is going to be about 2x as long as the output
        self.kwargs["max_tokens"] = max_new_tokens * input_output_ratio
        self._max_new_tokens: int = max_new_tokens
        self._model_name: str = model
        self._truncate_long_prompt_prompts: bool = truncate_long_prompts

        import boto3
        
        self.predictor = boto3.client(service_name, region_name=region_name)

    @abstractmethod
    def _create_body(self, prompt: str, **kwargs) -> dict[str, str | float]:
        pass

    def _sanitize_kwargs(self, query_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Ensure that input kwargs can be used by Bedrock or Sagemaker."""
        base_args: dict[str, Any] = {"temperature": self.kwargs["temperature"]}

        for k, v in base_args.items():
            if k not in query_kwargs:
                query_kwargs[k] = v
        if query_kwargs["temperature"] > 1.0:
            query_kwargs["temperature"] = 0.99
        if query_kwargs["temperature"] < 0.01:
            query_kwargs["temperature"] = 0.01

        return query_kwargs

    @abstractmethod
    def _call_model(self, body: str) -> str:
        """Call model, get generated input without the formatted prompt"""
        pass

    @abstractmethod
    def _extract_input_parameters(
        self, body: dict[Any, Any]
    ) -> dict[str, str | float | int]:
        pass

    def basic_request(self, prompt, **kwargs) -> str:
        """Query the endpoint."""

        # Remove any texts that are too long
        formatted_prompt: str
        if self._truncate_long_prompt_prompts:
            truncated_prompt: str = self._truncate_prompt(prompt)
            formatted_prompt = self._format_prompt(truncated_prompt)
        else:
            formatted_prompt = self._format_prompt((prompt))
        body: dict[str, str | float] = self._create_body(formatted_prompt, **kwargs)
        json_body: str = json.dumps(body)

        generated: str = self._call_model(json_body)

        self.history.append(
            {"prompt": formatted_prompt, "response": generated, "kwargs": body}
        )

        return generated.replace(formatted_prompt, "")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) * CHARS2TOKENS

    @abstractmethod
    def _format_prompt(self, raw_prompt: str) -> str:
        pass

    def _truncate_prompt(
        self,
        input_text: str,
        remove_beginning_or_ending: Literal["beginning", "ending"] = "beginning",
        max_input_tokens: int = 2500,
    ) -> str:
        """Reformat inputs such that they do not overflow context size limitation."""
        token_count = self._estimate_tokens(input_text)
        if token_count > self.kwargs["max_tokens"]:
            logging.info("Excessive prompt found in llm input")
            logging.info("Truncating texts to avoid error")
            max_chars: int = CHARS2TOKENS * max_input_tokens
            truncated_text: str
            if remove_beginning_or_ending == "ending":
                truncated_text = input_text[0:max_chars]
            else:
                truncated_text = input_text[-max_chars:]
                return truncated_text
        return input_text

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Query the AWS LLM.

        There is only support for only_completed=True and return_sorted=False
        right now.
        """
        if not only_completed:
            raise NotImplementedError("Error, only_completed not yet supported!")
        if return_sorted:
            raise NotImplementedError("Error, return_sorted not yet supported!")
        generated = self.basic_request(prompt, **kwargs)
        return [generated]
