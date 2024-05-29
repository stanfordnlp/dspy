import os
from typing import Any, Dict, Optional

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
        **kwargs: dict
            Additional arguments to pass to the API provider
        """
        self.model = "default" if model is None else model
        super().__init__(self.model)
        if premai_api_error == Exception:
            raise ImportError(
                "Not loading Prem AI because it is not installed. Install it with `pip install premai`.",
            )

        self.project_id = project_id

        api_key = get_premai_api_key(api_key=api_key)
        self.client = premai.Prem(api_key=api_key)
        self.provider = "premai"
        self.history: list[dict[str, Any]] = []

    @property
    def _default_params(self) -> Dict[str, Any]:
        default_kwargs = {
            "temperature": None, 
            "max_tokens": None, 
            "system_prompt": None, 
            "repositories": None, 
        }

        if self.model != "default":
            default_kwargs["model_name"] = self.model 

        return default_kwargs

    def _get_all_kwargs(self, **kwargs) -> Dict[str, Any]:
        kwargs_to_ignore = [
            "top_p",
            "tools",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "stop",
            "seed",
        ]
        keys_to_remove = []
        for key in kwargs:
            if key in kwargs_to_ignore:
                print(f"WARNING: Parameter {key} is not supported in kwargs.")
                keys_to_remove.append(key)

        for key in keys_to_remove:
            kwargs.pop(key)

        all_kwargs = {**self._default_params, **kwargs}
        for key in list(self._default_params.keys()):
            if all_kwargs.get(key) is None or all_kwargs.get(key) == "":
                all_kwargs.pop(key, None)
        return all_kwargs

    def basic_request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        all_kwargs = self._get_all_kwargs(**kwargs)
        messages = []        
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
        document_chunks = None if response.document_chunks is None else [
            chunk.to_dict() for chunk in response.document_chunks
        ]

        self.history.append(
            {
                "prompt": prompt,
                "response": content,
                "document_chunks": document_chunks, 
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
