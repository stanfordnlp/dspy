"""Module for interacting with Google Vertex AI."""

from typing import Any, Dict

import backoff
from pydantic_core import PydanticCustomError

from dsp.modules.lm import LM
from dsp.utils.settings import settings

try:
    import vertexai  # type: ignore[import-untyped]
    from vertexai.language_models import CodeGenerationModel, TextGenerationModel
    from vertexai.preview.generative_models import GenerativeModel
except ImportError:
    pass


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target']} with kwargs "
        f"{details['kwargs']}",
    )


def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


class GoogleVertexAI(LM):
    """Wrapper around GoogleVertexAI's API.

    Currently supported models include `gemini-pro-1.0`.
    """

    def __init__(
        self,
        model: str = "text-bison@002",
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Google to use?
            Choices are ['gemini-1.0-pro-001', 'gemini-1.0-pro',
            'claude-3-sonnet@20240229', 'claude-3-sonnet@20240229', 'claude-3-haiku@20240307',
            'text-bison@002', 'text-bison-32k@002', 'text-bison',]
            full list at https://console.cloud.google.com/vertex-ai/model-garden
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)
        self._is_gemini = "gemini" in model
        self._init_vertexai(kwargs)
        if "code" in model:
            model_cls = CodeGenerationModel
            self.available_args = {
                "suffix",
                "max_output_tokens",
                "temperature",
                "stop_sequences",
                "candidate_count",
            }
        elif "gemini" in model:
            model_cls = GenerativeModel
            self.available_args = {
                "max_output_tokens",
                "temperature",
                "top_k",
                "top_p",
                "stop_sequences",
                "candidate_count",
            }
        elif "text" in model:
            model_cls = TextGenerationModel
            self.available_args = {
                "max_output_tokens",
                "temperature",
                "top_k",
                "top_p",
                "stop_sequences",
                "candidate_count",
            }
        else:
            raise PydanticCustomError(
                "model",
                'model name is not valid, got "{model_name}"',
                dict(wrong_value=model),
            )
        if self._is_gemini:
            self.client = model_cls(
                model_name=model, safety_settings=kwargs.get("safety_settings"),
            )  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        else:
            self.client = model_cls.from_pretrained(model)
        self.provider = "googlevertexai"
        self.kwargs = {
            **self.kwargs,
            "temperature": 0.7,
            "max_output_tokens": 1024,
            "top_p": 1.0,
            "top_k": 1,
            **kwargs,
        }

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        vertexai.init(
            project=values.get("project"),
            location=values.get("location"),
            credentials=values.get("credentials"),
        )
        return

    def _prepare_params(
        self,
        parameters: Any,
    ) -> dict:
        stop_sequences = parameters.get("stop")
        params_mapping = {"n": "candidate_count", "max_tokens": "max_output_tokens"}
        params = {params_mapping.get(k, k): v for k, v in parameters.items()}
        params = {**self.kwargs, "stop_sequences": stop_sequences, **params}

        if self._is_gemini:
            if "candidate_count" in params and params["candidate_count"] != 1:
                print(
                    f"As of now, Gemini only supports `candidate_count == 1` (see also https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini#parameters). The current value for candidate_count of {params['candidate_count']} will be overridden.",
                )
                params["candidate_count"] = 1

        return {k: params[k] for k in set(params.keys()) & self.available_args}

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = self._prepare_params(raw_kwargs)
        if self._is_gemini:
            response = self.client.generate_content(
                [prompt],
                generation_config=kwargs,
            )
            history = {
                "prompt": prompt,
                "response": {
                    "prompt": prompt,
                    "choices": [
                        {
                            "text": "\n".join(v.text for v in c.content.parts),
                            "safetyAttributes": {
                                v.category: v.probability for v in c.safety_ratings
                            },
                        }
                        for c in response.candidates
                    ],
                },
                "kwargs": kwargs,
                "raw_kwargs": raw_kwargs,
            }
        else:
            response = self.client.predict(prompt, **kwargs).raw_prediction_response
            history = {
                "prompt": prompt,
                "response": {
                    "prompt": prompt,
                    "choices": [
                        {
                            "text": c["content"],
                            "safetyAttributes": c["safetyAttributes"],
                        }
                        for c in response.predictions
                    ],
                },
                "kwargs": kwargs,
                "raw_kwargs": raw_kwargs,
            }
        self.history.append(history)

        return [i["text"] for i in history["response"]["choices"]]

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
        giveup=giveup_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        """Handles retrieval of completions from Google whilst handling API errors"""
        return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        return self.request(prompt, **kwargs)
