import datetime
import hashlib
from typing import Any, Literal, Optional

import backoff

from dsp.modules.lm import LM

try:
    import cohere

    cohere_api_error = cohere.errors.UnauthorizedError
except ImportError:
    cohere_api_error = Exception
    # print("Not loading Cohere because it is not installed.")
except AttributeError:
    cohere_api_error = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


def post_request_metadata(model_name, prompt):
    """Creates a serialized request object for the Ollama API."""
    timestamp = datetime.datetime.now().timestamp()
    id_string = str(timestamp) + model_name + prompt
    hashlib.sha1().update(id_string.encode("utf-8"))
    id_hash = hashlib.sha1().hexdigest()
    return {"id": f"chatcmpl-{id_hash}", "object": "chat.completion", "created": int(timestamp), "model": model_name}


def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


class Cohere(LM):
    """Wrapper around Cohere's API.

    Currently supported models include `command-r-plus`, `command-r`, `command`, `command-nightly`, `command-light`, `command-light-nightly`.
    """

    def __init__(
        self,
        model: str = "command-r",
        api_key: Optional[str] = None,
        stop_sequences: list[str] = [],
        provider: Literal["cohere", "bedrock", "sagemaker"] = "cohere",
        region_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Cohere to use?
            Choices are [`command-r-plus`, `command-r`, `command`, `command-nightly`, `command-light`, `command-light-nightly`]
        api_key : str
            The API key for Cohere.
            It can be obtained from https://dashboard.cohere.ai/register.
        stop_sequences : list of str
            Additional stop tokens to end generation.
        provider : str
            Which Cohere provider to use?
            Choices are ["cohere", "bedrock", "sagemaker"]
            If "cohere", the API key is required.
            Other, the region name is required. Also, the AWS credentials must be set in the environment.
        region_name : str
            The region name for the AWS provider.
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)
        self.provider = provider
        if provider == "cohere":
            self.co = cohere.Client(api_key)
        elif provider == "bedrock":
            self.co = cohere.BedrockClient(aws_region=region_name)
        else:
            self.co = cohere.SagemakerClient(aws_region=region_name)
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 2000,
            "p": 0.99,
            "num_generations": 1,
            **kwargs,
        }
        self.stop_sequences = stop_sequences
        self.max_num_generations = 5

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            "stop_sequences": self.stop_sequences,
            "chat_history": [],
            "message": prompt,
            **kwargs,
        }
        kwargs.pop("num_generations")
        if "n" in kwargs.keys():
            kwargs.pop("n")
        response = self.co.chat(**kwargs)
        request_info = post_request_metadata(kwargs.get("model"), prompt)
        request_info["choices"] = [
            {
                "index": 0,
                "text": response.text,
            }
        ]
        self.history.append(
            {
                "prompt": prompt,
                "response": request_info,
                "kwargs": kwargs,
                "raw_kwargs": raw_kwargs,
            },
        )

        return response

    @backoff.on_exception(
        backoff.expo,
        (cohere_api_error),
        max_time=1000,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Cohere whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        **kwargs,
    ):
        response = self.request(prompt, **kwargs)
        return [response.text]
