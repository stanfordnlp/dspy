import logging
import os
from typing import Any, Optional

import backoff
import requests
from pydantic import BaseModel, ValidationError

from dsp.modules.lm import LM
from dsp.utils.settings import settings


def backoff_hdlr(details) -> None:
    """Log backoff details when retries occur."""
    logging.warning(
        f"Backing off {details['wait']:0.1f} seconds afters {details['tries']} tries "
        f"calling function {details['target']} with args {details['args']} and kwargs {details['kwargs']}",
    )


def giveup_hdlr(details) -> bool:
    """Decide whether to give up on retries based on the exception."""
    logging.error(
        "Giving up: After {tries} tries, calling {target} failed due to {value}".format(
            tries=details["tries"],
            target=details["target"],
            value=details.get("value", "Unknown Error"),
        ),
    )
    return False  # Always returns False to not give up


class LLMResponse(BaseModel):
    response: str


class CloudflareAIResponse(BaseModel):
    result: LLMResponse
    success: bool
    errors: list
    messages: list


class CloudflareAI(LM):
    """Wrapper around Cloudflare Workers AI API."""

    def __init__(
        self,
        model: str = "@hf/meta-llama/meta-llama-3-8b-instruct",
        account_id: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "cloudflare"
        self.model = model
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID") if account_id is None else account_id
        self.api_key = os.environ.get("CLOUDFLARE_API_KEY") if api_key is None else api_key
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.system_prompt = system_prompt
        self.kwargs = {
            "temperature": 0.0,  # Cloudflare Workers AI does not support temperature
            "max_tokens": kwargs.get("max_tokens", 256),
            **kwargs,
        }
        self.history: list[dict[str, Any]] = []

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
        on_giveup=giveup_hdlr,
    )
    def basic_request(self, prompt: str, **kwargs):  # noqa: ANN201 - Other LM implementations don't have a return type
        messages = [{"role": "user", "content": prompt}]

        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        json_payload = {"messages": messages, "max_tokens": self.kwargs["max_tokens"], **kwargs}

        response = requests.post(self.base_url, headers=self.headers, json=json_payload)  # noqa: S113 - There is a backoff decorator which handles timeout
        response.raise_for_status()

        """
        Schema of the response:
        {
          "result":
          {
            "response": string,
            "success":boolean,
            "errors":[],
            "messages":[]
          }
        }
        """
        try:
            res = response.json()
            cf = CloudflareAIResponse.model_validate(res)
            result = cf.result

            history_entry = {"prompt": prompt, "response": result.response, "kwargs": kwargs}
            self.history.append(history_entry)

            return result.response
        except ValidationError as e:
            logging.error(f"Error validating response: {e}")
            raise e

    def request(self, prompt: str, **kwargs):  # noqa: ANN201- Other LM implementations don't have a return type
        """Makes an API request to Cloudflare Workers AI with error handling."""
        return self.basic_request(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs):
        """Retrieve the AI completion from Cloudflare Workers AI API."""
        response = self.request(prompt, **kwargs)

        return [response]
