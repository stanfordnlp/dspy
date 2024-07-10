"""Module for interacting with Snowflake Cortex."""
import json
from typing import Any

import backoff
from pydantic_core import PydanticCustomError

from dsp.modules.lm import LM
from dsp.utils.settings import settings

try:
    from snowflake.snowpark import Session
    from snowflake.snowpark import functions as snow_func

except ImportError:
    pass


def backoff_hdlr(details) -> None:
    """Handler from https://pypi.org/project/backoff ."""
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries ",
        f"calling function {details['target']} with kwargs",
        f"{details['kwargs']}",
    )


def giveup_hdlr(details) -> bool:
    """Wrapper function that decides when to give up on retry."""
    if "rate limits" in str(details):
        return False
    return True


class Snowflake(LM):
    """Wrapper around Snowflake's CortexAPI.

    Currently supported models include 'snowflake-arctic','mistral-large','reka-flash','mixtral-8x7b',
    'llama2-70b-chat','mistral-7b','gemma-7b','llama3-8b','llama3-70b','reka-core'.
    """

    def __init__(self, model: str = "mixtral-8x7b", credentials=None, **kwargs):
        """Parameters

        ----------
        model : str
            Which pre-trained model from Snowflake to use?
            Choices are 'snowflake-arctic','mistral-large','reka-flash','mixtral-8x7b','llama2-70b-chat','mistral-7b','gemma-7b'
            Full list of supported models is available here: https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions#complete
        credentials: dict
            Snowflake credentials required to initialize the session. 
            Full list of requirements can be found here: https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/latest/api/snowflake.snowpark.Session
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)

        self.model = model
        cortex_models = [
            "llama3-8b",
            "llama3-70b",
            "reka-core",
            "snowflake-arctic",
            "mistral-large",
            "reka-flash",
            "mixtral-8x7b",
            "llama2-70b-chat",
            "mistral-7b",
            "gemma-7b",
        ]

        if model in cortex_models:
            self.available_args = {
                "max_tokens",
                "temperature",
                "top_p",
            }
        else:
            raise PydanticCustomError(
                "model",
                'model name is not valid, got "{model_name}"',
            )

        self.client = self._init_cortex(credentials=credentials)
        self.provider = "Snowflake"
        self.history: list[dict[str, Any]] = []
        self.kwargs = {
            **self.kwargs,
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "top_p": 1.0,
            "top_k": 1,
            **kwargs,
        }

    @classmethod
    def _init_cortex(cls, credentials: dict) -> None:
        session = Session.builder.configs(credentials).create()
        session.query_tag = {"origin": "sf_sit", "name": "dspy", "version": {"major": 1, "minor": 0}}

        return session

    def _prepare_params(
        self,
        parameters: Any,
    ) -> dict:
        params_mapping = {"n": "candidate_count", "max_tokens": "max_output_tokens"}
        params = {params_mapping.get(k, k): v for k, v in parameters.items()}
        params = {**self.kwargs, **params}
        return {k: params[k] for k in set(params.keys()) & self.available_args}

    def _cortex_complete_request(self, prompt: str, **kwargs) -> dict:
        complete = snow_func.builtin("snowflake.cortex.complete")
        cortex_complete_args = complete(
            snow_func.lit(self.model),
            snow_func.lit([{"role": "user", "content": prompt}]),
            snow_func.lit(kwargs),
        )
        raw_response = self.client.range(1).withColumn("complete_cal", cortex_complete_args).collect()

        if len(raw_response) > 0:
            return json.loads(raw_response[0].COMPLETE_CAL)

        else:
            return json.loads('{"choices": [{"messages": "None"}]}')

    def basic_request(self, prompt: str, **kwargs) -> list:
        raw_kwargs = kwargs
        kwargs = self._prepare_params(raw_kwargs)

        response = self._cortex_complete_request(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": {
                "prompt": prompt,
                "choices": [{"text": c} for c in response["choices"]],
            },
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }

        self.history.append(history)

        return [i["text"]["messages"] for i in history["response"]["choices"]]

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def _request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Snowflake Cortex whilst handling API errors."""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        return self._request(prompt, **kwargs)
