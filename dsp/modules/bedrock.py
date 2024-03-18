from __future__ import annotations

import json
from typing import Any, Optional

from dsp.modules.aws_lm import AWSLM


class Bedrock(AWSLM):
    def __init__(
        self,
        region_name: str,
        model: str,
        profile_name: Optional[str] = None,
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

    def _validate_model(self, model: str) -> None:
        if "claude" not in model.lower():
            raise NotImplementedError("Only claude models are supported as of now")

    def _create_body(self, prompt: str, **kwargs) -> dict[str, str | float]:
        base_args: dict[str, Any] = {
            "max_tokens_to_sample": self._max_new_tokens,
        }
        for k, v in kwargs.items():
            base_args[k] = v
        query_args: dict[str, Any] = self._sanitize_kwargs(base_args)
        query_args["prompt"] = prompt
        # AWS Bedrock forbids these keys
        if "max_tokens" in query_args:
            max_tokens: int = query_args["max_tokens"]
            input_tokens: int = self._estimate_tokens(prompt)
            max_tokens_to_sample: int = max_tokens - input_tokens
            del query_args["max_tokens"]
            query_args["max_tokens_to_sample"] = max_tokens_to_sample
        return query_args

    def _call_model(self, body: str) -> str:
        response = self.predictor.invoke_model(
            modelId=self._model_name,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        completion = response_body["completion"]
        return completion

    def _extract_input_parameters(
        self, body: dict[Any, Any],
    ) -> dict[str, str | float | int]:
        return body

    def _format_prompt(self, raw_prompt: str) -> str:
        return "\n\nHuman: " + raw_prompt + "\n\nAssistant:"
