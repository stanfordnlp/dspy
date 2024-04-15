"""Clarifai LM integration"""
from typing import Any, Optional

from dsp.modules.lm import LM


class ClarifaiLLM(LM):
    """Integration to call models hosted in clarifai platform.

    Args:
        model (str, optional): Clarifai URL of the model. Defaults to "Mistral-7B-Instruct".
        api_key (Optional[str], optional): CLARIFAI_PAT token. Defaults to None.
        **kwargs: Additional arguments to pass to the API provider.
    Example:
        import dspy
        dspy.configure(lm=dspy.Clarifai(model=MODEL_URL,
                                        api_key=CLARIFAI_PAT,
                                        inference_params={"max_tokens":100,'temperature':0.6}))
    """

    def __init__(
        self,
        model: str = "https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model)

        try:
            from clarifai.client.model import Model
        except ImportError as err:
            raise ImportError("ClarifaiLLM requires `pip install clarifai`.") from err


        self.provider = "clarifai"
        self.pat = api_key
        self._model = Model(url=model, pat=api_key)
        self.kwargs = {"n": 1, **kwargs}
        self.history: list[dict[str, Any]] = []
        self.kwargs["temperature"] = (
            self.kwargs["inference_params"]["temperature"]
            if "inference_params" in self.kwargs
            and "temperature" in self.kwargs["inference_params"]
            else 0.0
        )
        self.kwargs["max_tokens"] = (
            self.kwargs["inference_params"]["max_tokens"]
            if "inference_params" in self.kwargs
            and "max_tokens" in self.kwargs["inference_params"]
            else 150
        )

    def basic_request(self, prompt, **kwargs):
        params = (
            self.kwargs["inference_params"] if "inference_params" in self.kwargs else {}
        )
        response = (
            self._model.predict_by_bytes(
                input_bytes=prompt.encode(encoding="utf-8"),
                input_type="text",
                inference_params=params,
            )
            .outputs[0]
            .data.text.raw
        )
        kwargs = {**self.kwargs, **kwargs}
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        self.history.append(history)
        return response

    def request(self, prompt: str, **kwargs):
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
            completions.append(response)

        return completions
