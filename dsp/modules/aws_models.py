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
    """This class adds support for an AWS model.

    It is an abstract class and should not be instantiated directly.
    Instead, use one of the subclasses - AWSMistral, AWSAnthropic, or AWSMeta.
    The subclasses implement the abstract methods _create_body and _call_model
    and work in conjunction with the AWSProvider classes Bedrock and Sagemaker.
    Usage Example:
        bedrock = dspy.Bedrock(region_name="us-west-2")
        bedrock_mixtral = dspy.AWSMistral(bedrock, "mistral.mixtral-8x7b-instruct-v0:1", **kwargs)
        bedrock_haiku = dspy.AWSAnthropic(bedrock, "anthropic.claude-3-haiku-20240307-v1:0", **kwargs)
        bedrock_llama2 = dspy.AWSMeta(bedrock, "meta.llama2-13b-chat-v1", **kwargs)

        sagemaker = dspy.Sagemaker(region_name="us-west-2")
        sagemaker_model = dspy.AWSMistral(sagemaker, "<YOUR_ENDPOINT_NAME>", **kwargs)
    """

    def __init__(
        self,
        model: str,
        max_context_size: int,
        max_new_tokens: int,
        **kwargs,
    ) -> None:
        """_summary_.

        Args:
            model (str, optional): An LM name, e.g., a bedrock name or an AWS endpoint.
            max_context_size (int): The maximum context size in tokens.
            max_new_tokens (int): The maximum number of tokens to be sampled from the LM.
            **kwargs: Additional arguments.
        """
        super().__init__(model=model)
        self._model_name: str = model
        self._max_context_size: int = max_context_size
        self._max_new_tokens: int = max_new_tokens

        # make it consistent with equivalent LM::max_token
        self.kwargs["max_tokens"] = max_new_tokens

        self.kwargs = {
            **self.kwargs,
            **kwargs,
        }

    @abstractmethod
    def _create_body(self, prompt: str, **kwargs) -> tuple[int, dict[str, str | float]]:
        pass

    @abstractmethod
    def _call_model(self, body: str) -> str | list[str]:
        """Call model, get generated input without the formatted prompt."""

    def _estimate_tokens(self, text: str) -> int:
        return len(text) / CHARS2TOKENS

    def _extract_input_parameters(
        self,
        body: dict[Any, Any],
    ) -> dict[str, str | float | int]:
        return body

    def _format_prompt(self, raw_prompt: str) -> str:
        return "\n\nHuman: " + raw_prompt + "\n\nAssistant:"

    def _simple_api_call(self, formatted_prompt: str, **kwargs) -> str | list[str]:
        n, body = self._create_body(formatted_prompt, **kwargs)
        json_body = json.dumps(body)

        if n > 1:
            llm_out = [self._call_model(json_body) for _ in range(n)]
            llm_out = [generated.replace(formatted_prompt, "") for generated in llm_out]
        else:
            llm_out = self._call_model(json_body)
            llm_out = llm_out.replace(formatted_prompt, "")

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
            raise ValueError("only_completed must be True for now")
        if return_sorted:
            raise ValueError("return_sorted must be False for now")

        generated = self.basic_request(prompt, **kwargs)
        return [generated]


class AWSMistral(AWSModel):
    """Mistral family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 32768,
        max_new_tokens: int = 1500,
        **kwargs,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self.aws_provider = aws_provider
        self.provider = aws_provider.get_provider_name()

        self.kwargs["stop"] = "\n\n---"

    def _format_prompt(self, raw_prompt: str) -> str:
        return "<s> [INST] Human: " + raw_prompt + " [/INST] Assistant: "

    def _create_body(self, prompt: str, **kwargs) -> tuple[int, dict[str, str | float]]:
        base_args: dict[str, Any] = self.kwargs
        for k, v in kwargs.items():
            base_args[k] = v

        n, base_args = self.aws_provider.sanitize_kwargs(base_args)

        query_args: dict[str, str | float] = {}
        if isinstance(self.aws_provider, Bedrock):
            query_args["prompt"] = prompt
        elif isinstance(self.aws_provider, Sagemaker):
            query_args["parameters"] = base_args
            query_args["inputs"] = prompt
        else:
            raise ValueError("Error - provider not recognized")

        return (n, query_args)

    def _call_model(self, body: str) -> str:
        response = self.aws_provider.call_model(
            model_id=self._model_name,
            body=body,
        )
        if isinstance(self.aws_provider, Bedrock):
            response_body = json.loads(response["body"].read())
            completion = response_body["outputs"][0]["text"]
        elif isinstance(self.aws_provider, Sagemaker):
            response_body = json.loads(response["Body"].read())
            completion = response_body[0]["generated_text"]
        else:
            raise ValueError("Error - provider not recognized")

        return completion.split(self.kwargs["stop"])[0]


class AWSAnthropic(AWSModel):
    """Anthropic family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 200000,
        max_new_tokens: int = 1500,
        **kwargs,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self.aws_provider = aws_provider
        self.provider = aws_provider.get_provider_name()

        if isinstance(self.aws_provider, Bedrock):
            self.kwargs["anthropic_version"] = "bedrock-2023-05-31"

        for k, v in kwargs.items():
            self.kwargs[k] = v

    def _create_body(self, prompt: str, **kwargs) -> tuple[int, dict[str, str | float]]:
        base_args: dict[str, Any] = self.kwargs
        for k, v in kwargs.items():
            base_args[k] = v

        n, query_args = self.aws_provider.sanitize_kwargs(base_args)

        # Anthropic models do not support the following parameters
        query_args.pop("frequency_penalty", None)
        query_args.pop("num_generations", None)
        query_args.pop("presence_penalty", None)
        query_args.pop("model", None)

        # we are using the Claude messages API
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
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
        return (n, query_args)

    def _call_model(self, body: str) -> str:
        response = self.aws_provider.predictor.invoke_model(
            modelId=self._model_name,
            body=body,
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]


class AWSMeta(AWSModel):
    """Llama3 family of models."""

    def __init__(
        self,
        aws_provider: AWSProvider,
        model: str,
        max_context_size: int = 4096,
        max_new_tokens: int = 1500,
        **kwargs,
    ) -> None:
        """NOTE: Configure your AWS credentials with the AWS CLI before using this model!"""
        super().__init__(
            model=model,
            max_context_size=max_context_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        self.aws_provider = aws_provider
        self.provider = aws_provider.get_provider_name()

        self.kwargs["stop"] = ["<|eot_id|>"]

        for k, v in kwargs.items():
            self.kwargs[k] = v

    def _format_prompt(self, raw_prompt: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            + raw_prompt
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    def _create_body(self, prompt: str, **kwargs) -> tuple[int, dict[str, str | float]]:
        base_args: dict[str, Any] = self.kwargs.copy()
        for k, v in kwargs.items():
            base_args[k] = v

        n, base_args = self.aws_provider.sanitize_kwargs(base_args)

        # Meta models do not support the following parameters
        base_args.pop("frequency_penalty", None)
        base_args.pop("num_generations", None)
        base_args.pop("presence_penalty", None)
        base_args.pop("model", None)

        max_tokens = base_args.pop("max_tokens", None)
        
        query_args: dict[str, str | float] = {}
        if isinstance(self.aws_provider, Bedrock):
            if max_tokens:
                base_args["max_gen_len"] = max_tokens
            query_args = base_args
            query_args["prompt"] = prompt
        elif isinstance(self.aws_provider, Sagemaker):
            if max_tokens:
                base_args["max_new_tokens"] = max_tokens
            query_args["parameters"] = base_args
            query_args["inputs"] = prompt
        else:
            raise ValueError("Error - provider not recognized")

        return (n, query_args)

    def _call_model(self, body: str) -> str:
        response = self.aws_provider.call_model(
            model_id=self._model_name,
            body=body,
        )
        if isinstance(self.aws_provider, Bedrock):
            response_body = json.loads(response["body"].read())
            completion = response_body["generation"]
        elif isinstance(self.aws_provider, Sagemaker):
            response_body = json.loads(response["Body"].read())
            completion = response_body["generated_text"]
        else:
            raise ValueError("Error - provider not recognized")
        return completion
