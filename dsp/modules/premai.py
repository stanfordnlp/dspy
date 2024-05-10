import logging
from typing import Optional

import backoff
import premai.errors

from dsp.modules.lm import LM

try:
    import premai
except ImportError as err:
    raise ImportError(
        "Not loading Mistral AI because it is not installed. Install it with `pip install premai`.",
    ) from err


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatPremAPIError(Exception):
    """Error with the `PremAI` API."""


ERROR = ChatPremAPIError


def backoff_hdlr(details) -> None:
    """Handler for the backoff package.

    See more at: https://pypi.org/project/backoff/
    """
    logger.info(
        "Backing off {wait:0.1f} seconds after {tries} tries calling function {target} with kwargs {kwargs}".format(
            **details,
        ),
    )


def giveup_hdlr(details) -> bool:
    """Wrapper function that decides when to give up on retry."""
    if "rate limits" in details.message:
        return False
    return True


class PremAI(LM):
    """Wrapper around Prem AI's API."""

    def __init__(
        self,
        model: str,
        project_id: int,
        api_key: str,
        base_url: Optional[str] = None,
        session_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Parameters

        model: str
            The name of model name
        project_id: int
            "The project ID in which the experiments or deployments are carried out. can find all your projects here: https://app.premai.io/projects/"
        api_key: str
            Prem AI API key, to connect with the API.
        session_id: int
            The ID of the session to use. It helps to track the chat history.
        **kwargs: dict
            Additional arguments to pass to the API provider
        """
        super().__init__(model)
        self.kwargs = kwargs if kwargs == {} else self.kwargs

        self.project_id = project_id
        self.session_id = session_id

        if base_url is not None:
            self.client = premai.Prem(api_key=api_key, base_url=base_url)
        else:
            self.client = premai.Prem(api_key=api_key)
        self.provider = "premai"

        self.kwargs = {
            "model": model,
            "temperature": 0.17,
            "max_tokens": 150,
            **kwargs,
        }
        if session_id is not None:
            kwargs["sesion_id"] = session_id

    def _get_all_kwargs(self, **kwargs) -> dict:
        other_kwargs = {
            "seed": None,
            "logit_bias": None,
            "tools": None,
            "system_prompt": None,
        }
        all_kwargs = {
            **self.kwargs,
            **other_kwargs,
            **kwargs,
        }

        _keys_that_cannot_be_none = [
            "system_prompt",
            "frequency_penalty",
            "presence_penalty",
            "tools",
            "model",
        ]

        for key in _keys_that_cannot_be_none:
            if all_kwargs.get(key) is None:
                all_kwargs.pop(key, None)
        return all_kwargs

    def basic_request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        all_kwargs = self._get_all_kwargs(**kwargs)
        message = []

        if "system_prompt" in all_kwargs:
            message.append({"role": "system", "content": all_kwargs["system_prompt"]})
        message.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            project_id=self.project_id,
            messages=message,
            **all_kwargs,
        )
        if not response.choices:
            raise ChatPremAPIError("ChatResponse must have at least one candidate")

        return response.choices[0].message.content or ""

    @backoff.on_exception(
        backoff.expo,
        (ERROR),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        return self.basic_request(prompt=prompt, **kwargs)

    def __call__(self, prompt, **kwargs):
        return self.request(prompt, **kwargs)
