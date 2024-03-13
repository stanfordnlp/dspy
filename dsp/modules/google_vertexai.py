from vertexai.generative_models import GenerativeModel, GenerationConfig
from dsp.modules.lm import LM
import os
from collections.abc import Iterable
from typing import Any, Optional
import backoff
import vertexai


class VertexAI(LM):
    """Wrapper around Google's VertexAI API.

    Currently supported models include `gemini-pro-1.0`.
    """

    def __init__(
        self,
        model: str = "gemini-1.0-pro",
        project: str = None,
        location: Optional[str] = "us-central1",
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            Which pre-trained model from Google to use?
            Choices are [`gemini-pro-1.0`]
        api_key : str
            The API key for Google.
            It can be obtained from https://cloud.google.com/generative-ai-studio
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)

        # if no project is provided, raise an error
        if project is None:
            raise ValueError("A project must be provided")

        vertexai.init(project=project, location=location)

        # Google API uses "candidate_count" instead of "n" or "num_generations"
        # For now, google API only supports 1 generation at a time. Raises an error if candidate_count > 1
        num_generations = kwargs.pop("n", kwargs.pop("num_generations", 1))

        self.provider = "vertexai"
        kwargs = {
            "candidate_count": 1,
            "temperature": (
                0.0 if "temperature" not in kwargs else kwargs["temperature"]
            ),
            "max_output_tokens": 8192,
            "top_p": 1,
            "top_k": 20,
            **kwargs,
        }

        self.config = GenerationConfig(**kwargs)
        self.llm = GenerativeModel(
            model_name=model,
            generation_config=self.config,
        )

        # "max_tokens" is expected by DSPy code
        self.kwargs = {
            "n": num_generations,
            "max_tokens": 8192,
            **kwargs,
        }

        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {
            **self.kwargs,
            **kwargs,
        }

        # Google disallows "n" arguments
        n = kwargs.pop("n", None)
        # Google disallows "max_tokens" arguments
        n = kwargs.pop("max_tokens", None)
        if n is not None and n > 1 and kwargs["temperature"] == 0.0:
            kwargs["temperature"] = 0.7

        response = self.llm.generate_content(prompt, generation_config=kwargs)

        history = {
            "prompt": prompt,
            "response": [response],
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    # @backoff.on_exception(
    #     backoff.expo,
    #     (google_api_error),
    #     max_time=1000,
    #     max_tries=8,
    #     on_backoff=backoff_hdlr,
    #     giveup=giveup_hdlr,
    # )
    # def request(self, prompt: str, **kwargs):
    #     """Handles retrieval of completions from Google whilst handling API errors"""
    #     return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.candidates[0].text)

        return completions
