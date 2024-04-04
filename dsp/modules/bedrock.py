from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dsp.modules.aws_lm import AWSLM


@dataclass
class ChatMessage:
    role: str
    content: str


class Bedrock(AWSLM):
    def __init__(
            self,
            region_name: str,
            model: str,
            profile_name: str | None = None,
            input_output_ratio: int = 3,
            max_new_tokens: int = 1500,
    ) -> None:
        """Use an AWS Bedrock language model.
        NOTE: You must first configure your AWS credentials with the AWS CLI before using this model!

        Args:
            region_name (str, optional): The AWS region where this LM is hosted.
            model (str, optional): An AWS Bedrock LM name. You can find available models with the AWS CLI as follows: aws bedrock list-foundation-models --query "modelSummaries[*].modelId".
            temperature (float, optional): Default temperature for LM. Defaults to 0.
            input_output_ratio (int, optional): The rough size of the number of input tokens to output tokens in the worst case. Defaults to 3.
            max_new_tokens (int, optional): The maximum number of tokens to be sampled from the LM.
        """
        super().__init__(
            model=model,
            service_name="bedrock-runtime",
            region_name=region_name,
            profile_name=profile_name,
            truncate_long_prompts=False,
            input_output_ratio=input_output_ratio,
            max_new_tokens=max_new_tokens,
            batch_n=True,  # Bedrock does not support the `n` parameter
        )
        self._validate_model(model)
        self.provider = "claude" if "claude" in model.lower() else "bedrock"
        self.use_messages = "claude-3" in model.lower()

    def _validate_model(self, model: str) -> None:
        if "claude" not in model.lower():
            raise NotImplementedError("Only claude models are supported as of now")

    def _create_body(self, prompt: str, system_prompt: str | None = None, **kwargs) -> dict[str, Any]:
        base_args: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
        }
        for k, v in kwargs.items():
            base_args[k] = v

        query_args: dict[str, Any] = self._sanitize_kwargs(base_args)

        if self.use_messages:
            messages = [ChatMessage(role="user", content=prompt)]
            if system_prompt:
                messages.insert(0, ChatMessage(role="system", content=system_prompt))
            else:
                messages.insert(0, ChatMessage(role="system", content="You are a helpful AI assistant."))
            serialized_messages = [vars(m) for m in messages if m.role != "system"]
            system_message = next(m["content"] for m in [vars(m) for m in messages if m.role == "system"])
            query_args["messages"] = serialized_messages
            query_args["system"] = system_message
            query_args["max_tokens"] = self._max_new_tokens
        else:
            query_args["prompt"] = self._format_prompt(prompt)
            query_args["max_tokens_to_sample"] = self._max_new_tokens

        return query_args

    def _call_model(self, body: str) -> str:
        response = self.predictor.invoke_model(
            modelId=self._model_name,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())

        if self.use_messages:  # Claude-3 model
            try:
                completion = response_body['content'][0]['text']
            except (KeyError, IndexError):
                raise ValueError("Unexpected response format from the Claude-3 model.")
        else:  # Other models
            expected_keys = ["completion", "text"]
            found_key = next((key for key in expected_keys if key in response_body), None)

            if found_key:
                completion = response_body[found_key]
            else:
                raise ValueError(
                    f"Unexpected response format from the model. Expected one of {', '.join(expected_keys)} keys.")

        return completion

    def _extract_input_parameters(
            self, body: dict[Any, Any],
    ) -> dict[str, str | float | int]:
        return body

    def _format_prompt(self, raw_prompt: str) -> str:
        return "\n\nHuman: " + raw_prompt + "\n\nAssistant:"