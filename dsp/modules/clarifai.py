import math
from typing import Any, Optional
import backoff

from dsp.modules.lm import LM

try:
    from clarifai.client.model import Model
except ImportError:
    raise ImportError("ClarifaiLLM requires `pip install clarifai`.")

class ClarifaiLLM(LM):
    """Integration to call models hosted in clarifai platform."""

    def __init__(
        self,
        model: str = "https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct", #defaults to mistral-7B-Instruct
        api_key: Optional[str] = None,
        stop_sequences: list[str] = [],
        **kwargs,
    ):
        super().__init__(model)

        self.provider = "clarifai"
        self.pat=api_key
        self._model= Model(url=model, pat=api_key)
        self.kwargs = {
            "n": 1,
            **kwargs
        }
        self.history :list[dict[str, Any]] = []
        self.kwargs['temperature'] = (
            self.kwargs['inference_params']['temperature'] if
            'inference_params' in self.kwargs and
             'temperature' in self.kwargs['inference_params'] else 0.0
        ) 
        self.kwargs['max_tokens'] = (
            self.kwargs['inference_params']['max_tokens'] if
            'inference_params' in self.kwargs and
             'max_tokens' in self.kwargs['inference_params'] else 150
        )
    
    def basic_request(self, prompt, **kwargs):
        
        params = (
            self.kwargs['inference_params'] if 'inference_params' in self.kwargs
            else {}
        )
        response = (
                self._model.predict_by_bytes(
                    input_bytes= prompt.encode(encoding="utf-8"),
                    input_type= "text",
                    inference_params= params
                ).outputs[0].data.text.raw
                
            )

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        self.history.append(history)
        return response
    
    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        return choice
    
    def request(self, prompt: str, **kwargs):
        return self.basic_request(prompt, **kwargs)
    
    def __call__(self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs
    ):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        n = kwargs.pop("n", 1)
        completions=[]

        for i in range(n):
            response = self.request(prompt, **kwargs)
            completions.append(response)

        return completions
