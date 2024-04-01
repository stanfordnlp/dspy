"""AWS models for LMs."""

from __future__ import annotations

import json
import logging
from abc import abstractmethod
from typing import Any

from dsp.modules.aws_providers import AWSProvider, Bedrock, Sagemaker
from dsp.modules.lm import LM

# Heuristic translating number of chars to tokens
# ~4 chars = 1 token
CHARS2TOKENS: int = 4


class AWSModel(LM):
    """This class adds support for an AWS model."""

    def __init__(
        self,
        model: str,
        max_context_size: int,
        max_new_tokens: int,
    ) -> None:
        """_summary_.

        Args:
            model (str, optional): An LM name, e.g., a bedrock name or an AWS endpoint.
            max_context_size (int): The maximum context size in tokens.
            max_new_tokens (int): The maximum number of tokens to be sampled from the LM.
        """
        super().__init__(model=model)
        self._model_name: str = model
        self._max_context_size: int = max_context_size
        self._max_new_tokens: int = max_new_tokens

    @abstractmethod
    def _create_body(self, prompt: str, **kwargs):
        pass

    @abstractmethod
    def _call_model(self, body: str) -> str | list[str]:
        """Call model, get generated input without the formatted prompt."""

    def _estimate_tokens(self, text: str) -> int:
        return len(text) * CHARS2TOKENS

    def _extract_input_parameters(
        self,
        body: dict[Any, Any],
    ) -> dict[str, str | float | int]:
        return body

    def _format_prompt(self, raw_prompt: str) -> str:
        return "\n\nHuman: " + raw_prompt + "\n\nAssistant:"

    def _sanitize_kwargs(self, query_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Ensure that input kwargs can be used by Bedrock or Sagemaker."""
        if "temperature" in query_kwargs:
            if query_kwargs["temperature"] > 1.0:
                query_kwargs["temperature"] = 0.99
            if query_kwargs["temperature"] < 0.01:
                query_kwargs["temperature"] = 0.01

        return query_kwargs

    def _simple_api_call(self, formatted_prompt: str, **kwargs) -> str | list[str]:
        body = self._create_body(formatted_prompt, **kwargs)
        json_body = json.dumps(body)
        llm_out: str | list[str] = self._call_model(json_body)
        if isinstance(llm_out, str):
            llm_out = llm_out.replace(formatted_prompt, "")
        else:
            llm_out = [generated.replace(formatted_prompt, "") for generated in llm_out]
        self.history.append(
            {"prompt": formatted_prompt, "response": llm_out, "kwargs": body},
        )
        return llm_out

    def basic_request(self, prompt, **kwargs) -> str | list[str]:
        """Query the endpoint."""
        token_count = self._estimate_tokens(prompt)
        if token_count > self._max_context_size:
            logging.info("Error - input tokens %s exceeds max context %s", token_count, self._max_context_size)
            raise ValueError(
                f"Error - input tokens {token_count} exceeds max context {self._max_context_size}",
            )

        formatted_prompt: str = self._format_prompt(prompt)
        return self._simple_api_call(formatted_prompt=formatted_prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """Query the AWS LLM.

        There is only support for only_completed=True and return_sorted=False
        right now.
        """
        if not only_completed:
            raise NotImplementedError("Error, only_completed not yet supported!")
        if return_sorted:
            raise NotImplementedError("Error, return_sorted not yet supported!")

        generated = self.basic_request(prompt, **kwargs)
        return [generated]


class AWSMistral(AWSModel):
    """Mistral family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 4096,
        max_new_tokens: int = 1500,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
        )
        self.provider = aws_provider

    def _create_body(self, prompt: str, **kwargs) -> dict[str, str | float]:
        base_args: dict[str, Any] = {}

        base_args["max_tokens"] = self._max_new_tokens

        for k, v in kwargs.items():
            base_args[k] = v

        query_args: dict[str, Any] = {}

        if isinstance(self.provider, Bedrock):
            query_args = self._sanitize_kwargs(base_args)
            query_args["prompt"] = prompt
        elif isinstance(self.provider, Sagemaker):
            query_args["parameters"] = self._sanitize_kwargs(base_args)
            query_args["inputs"] = prompt
        else:
            raise ValueError("Error - provider not recognized")

        return query_args

    def _call_model(self, body: str) -> str:
        response = self.provider.call_model(
            model_id=self._model_name,
            body=body,
        )
        if isinstance(self.provider, Bedrock):
            response_body = json.loads(response["body"].read())
            completion = response_body["outputs"][0]["text"]
        elif isinstance(self.provider, Sagemaker):
            response_body = json.loads(response["Body"].read())
            completion = response_body[0]["generated_text"]
        else:
            raise ValueError("Error - provider not recognized")

        return completion


class AWSAnthropic(AWSModel):
    """Anthropic family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 4096,
        max_new_tokens: int = 1500,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
        )
        self.provider = aws_provider

    def _create_body(self, prompt: str, **kwargs) -> dict[str, str | float]:
        base_args: dict[str, Any] = {}

        base_args["max_tokens"] = self._max_new_tokens

        for k, v in kwargs.items():
            base_args[k] = v

        query_args: dict[str, Any] = self._sanitize_kwargs(base_args)

        query_args["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        query_args["anthropic_version"] = "bedrock-2023-05-31"
        return query_args

    def _call_model(self, body: str) -> str:
        response = self.provider.predictor.invoke_model(
            modelId=self._model_name,
            body=body,
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]


class AWSLlama2(AWSModel):
    """Llama2 family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 4096,
        max_new_tokens: int = 1500,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
        )
        self.provider = aws_provider

    def _create_body(self, prompt: str, **kwargs) -> dict[str, str | float]:
        base_args: dict[str, Any] = {}

        base_args["max_gen_len"] = self._max_new_tokens

        for k, v in kwargs.items():
            base_args[k] = v

        query_args: dict[str, Any] = self._sanitize_kwargs(base_args)
        query_args["prompt"] = prompt
        return query_args

    def _call_model(self, body: str) -> str:
        response = self.provider.predictor.invoke_model(
            modelId=self._model_name,
            body=body,
        )
        response_body = json.loads(response["body"].read())
        return response_body["generation"]
