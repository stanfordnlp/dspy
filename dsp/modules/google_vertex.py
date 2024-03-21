import os
from collections.abc import Iterable
from typing import Any, Optional

import backoff

from dsp.modules import LM

try:
    from vertexai.generative_models import GenerativeModel
    from vertexai.generative_models import GenerationConfig, HarmCategory, HarmBlockThreshold
    import vertexai
    from google.api_core.exceptions import GoogleAPICallError
    gcp_api_error = GoogleAPICallError
except ImportError:
    gcp_api_error = Exception


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )

def giveup_hdlr(details):
    """wrapper function that decides when to give up on retry"""
    if "rate limits" in details.message:
        return False
    return True


BLOCK_ONLY_HIGH = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

class GoogleVertex(LM):
    """Wrapper around Google Cloud's Vertex API.

    Supports all text+ models accessible through
        https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    """

    def __init__(
            self,
            project_id: str,
            region: str = "us-central1",
            model: str = "gemini-1.0-pro",
            credentials = None,
            safety_settings: Optional[Iterable] = BLOCK_ONLY_HIGH,
            **kwargs,
    ):
        """
        Parameters
        ----------
        project_id : str
            Which GCP project to use?
        model : str
            Which pre-trained model from Google to use?
            Choices can be found at https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
        **kwargs: dict
            Additional arguments to pass to the API provider.
        """
        super().__init__(model)

        vertexai.init(project=project_id, location=region, credentials=credentials)

        # Google API uses "candidate_count" instead of "n" or "num_generations"
        # Gemini only supports 1 generation at a time. Raises an error if candidate_count > 1.
        # Bison supports up to 4.
        num_generations = kwargs.pop("n", kwargs.pop("num_generations", 1))

        max_output_tokens = kwargs.pop("max_output_tokens", 2048)

        if model.startswith("gemini"):
            num_generations = 1
        else:
            num_generations = 4 # Bison supports upto 4 candidates.


        self.provider = "google_vertex"

        kwargs = {
            "candidate_count": num_generations,
            "temperature": 0.0 if "temperature" not in kwargs else kwargs["temperature"],
            "max_output_tokens": 2048,
            "top_p": 1,
            "top_k": 1,
            **kwargs,
        }

        self.config = GenerationConfig(**kwargs)

        self.llm = GenerativeModel(
            model_name=model,
            generation_config = self.config,
            safety_settings = safety_settings
            )
        
        self.kwargs = {
            "n": num_generations,
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
        if n is not None and n > 1 and kwargs['temperature'] == 0.0:
            kwargs['temperature'] = 0.7

        response = self.llm.generate_content(prompt, generation_config=kwargs)

        history = {
            "prompt": prompt,
            "response": [response],
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        (gcp_api_error),
        max_time=1000,
        max_tries=8,
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
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)

        completions = []
        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response.candidates[0].content.parts[0].text)
        return completions
