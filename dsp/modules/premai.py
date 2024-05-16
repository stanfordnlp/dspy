import os
from typing import Any, Optional

import backoff

from dsp.modules.lm import LM

try:
    import premai

    premai_api_error = premai.errors.UnexpectedStatus
except ImportError:
    premai_api_error = Exception
except AttributeError:
    premai_api_error = Exception


def backoff_hdlr(details) -> None:
    """Handler for the backoff package.

    See more at: https://pypi.org/project/backoff/
    """
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries calling function {target} with kwargs {kwargs}".format(
            **details,
        ),
    )


def giveup_hdlr(details) -> bool:
    """Wrapper function that decides when to give up on retry."""
    if "rate limits" in details.message:
        return False
    return True


def get_premai_api_key(api_key: Optional[str] = None) -> str:
    """Retrieve the PreMAI API key from a passed argument or environment variable."""
    api_key = api_key or os.environ.get("PREMAI_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "No API key found. See the quick start guide at https://docs.premai.io/introduction to get your API key.",
        )
    return api_key


class PremAI(LM):
    """Wrapper around Prem AI's API."""

    def __init__(
        self,
        project_id: int,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Parameters

        project_id: int
            "The project ID in which the experiments or deployments are carried out. can find all your projects here: https://app.premai.io/projects/"
        model: Optional[str]
            The name of model deployed on launchpad. When None, it will show 'default'
        api_key: Optional[str]
            Prem AI API key, to connect with the API. If not provided then it will check from env var by the name
                PREMAI_API_KEY
        session_id: Optional[int]
            The ID of the session to use. It helps to track the chat history.
        **kwargs: dict
            Additional arguments to pass to the API provider
        """
        model = "default" if model is None else model
        super().__init__(model)
        if premai_api_error == Exception:
            raise ImportError(
                "Not loading Prem AI because it is not installed. Install it with `pip install premai`.",
            )
        self.kwargs = kwargs if kwargs == {} else self.kwargs

        self.project_id = project_id
        self.session_id = session_id

        api_key = get_premai_api_key(api_key=api_key)
        self.client = premai.Prem(api_key=api_key)
        self.provider = "premai"
        self.history: list[dict[str, Any]] = []

        self.kwargs = {
            "temperature": 0.17,
            "max_tokens": 150,
            **kwargs,
        }
        if session_id is not None:
            self.kwargs["session_id"] = session_id

        # However this is not recommended to change the model once
        # deployed from launchpad

        if model != "default":
            self.kwargs["model"] = model

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
        ]

        for key in _keys_that_cannot_be_none:
            if all_kwargs.get(key) is None:
                all_kwargs.pop(key, None)
        return all_kwargs

    def basic_request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = []

        if "system_prompt" in all_kwargs:
            messages.append({"role": "system", "content": all_kwargs["system_prompt"]})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            project_id=self.project_id,
            messages=messages,
            **all_kwargs,
        )
        if not response.choices:
            raise premai_api_error("ChatResponse must have at least one candidate")

        content = response.choices[0].message.content
        if not content:
            raise premai_api_error("ChatResponse is none")

        output_text = content or ""

        self.history.append(
            {
                "prompt": prompt,
                "response": content,
                "kwargs": all_kwargs,
                "raw_kwargs": kwargs,
            },
        )

        return output_text

    @backoff.on_exception(
        backoff.expo,
        (premai_api_error),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        return self.basic_request(prompt=prompt, **kwargs)

    def __call__(self, prompt, **kwargs):
        return self.request(prompt, **kwargs)
