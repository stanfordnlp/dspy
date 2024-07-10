import os
import warnings
from typing import Any, Optional

import backoff

from dsp.modules.lm import LM
from dsp.utils.settings import settings

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
    print(  # noqa: T201
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
    def __init__(
        self,
        project_id: int,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: dict,
    ) -> None:
        """Parameters

        project_id: int
            "The project ID in which the experiments or deployments are carried out. can find all your projects here: https://app.premai.io/projects/"
        model: Optional[str]
            The name of model deployed on launchpad. When None, it will show 'default'
        api_key: Optional[str]
            Prem AI API key, to connect with the API. If not provided then it will check from env var by the name
                PREMAI_API_KEY
        kwargs: Optional[dict] For any additional paramters
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
        self.kwargs = kwargs if kwargs else {}

    @property
    def _default_params(self) -> dict[str, Any]:
        default_kwargs = {
            "temperature": None,
            "max_tokens": None,
            "system_prompt": None,
            "repositories": None,
        }

        if self.model != "default":
            default_kwargs["model_name"] = self.model

        return default_kwargs

    def _get_all_kwargs(self, **kwargs) -> dict[str, Any]:
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
        kwargs = {**kwargs, **self.kwargs}

        for key in kwargs:
            if key in kwargs_to_ignore:
                warnings.warn(f"WARNING: Parameter {key} is not supported in kwargs.", stacklevel=2)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            kwargs.pop(key)

        all_kwargs = {**self._default_params, **kwargs}
        for key in list(self._default_params.keys()):
            if all_kwargs.get(key) is None or all_kwargs.get(key) == "":
                all_kwargs.pop(key, None)
        return all_kwargs

    def basic_request(self, prompt, **kwargs) -> list[str]:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "template_id" not in all_kwargs:
            messages = [{"role": "user", "content": prompt}]
        else:
            template_id = all_kwargs["template_id"]
            if (template_id is None) or (template_id == ""):
                raise ValueError("Templates can not be None or ''")
            if "params" not in all_kwargs:
                raise KeyError(
                    "Keyword argument: params must be present if template_id is present",
                )

            params = all_kwargs["params"]
            if not isinstance(params, dict):
                raise TypeError("params must be a dictionary")

            messages = [
                {
                    "role": "user",
                    "template_id": template_id,
                    "params": params,
                },
            ]
            kwargs["template_id"] = all_kwargs.get("template_id", None)
            kwargs["params"] = all_kwargs.get("params", None)

            all_kwargs.pop("template_id")
            all_kwargs.pop("params")

        kwargs = {**kwargs, **all_kwargs}
        response = self.client.chat.completions.create(
            project_id=self.project_id,
            messages=messages,
            **all_kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise premai_api_error("ChatResponse is none")

        output_text = content or ""
        document_chunks = (
            None if response.document_chunks is None else [chunk.to_dict() for chunk in response.document_chunks]
        )

        self.history.append(
            {
                "prompt": prompt,
                "response": content,
                "document_chunks": document_chunks,
                "kwargs": kwargs,
            },
        )
        return [output_text]

    @backoff.on_exception(
        backoff.expo,
        (premai_api_error),
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt, **kwargs) -> str:
        """Handles retrieval of completions from Prem AI whilst handling API errors."""
        return self.basic_request(prompt=prompt, **kwargs)

    def __call__(self, prompt, **kwargs):
        return self.request(prompt, **kwargs)
