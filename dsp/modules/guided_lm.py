from dsp.modules.lm import LM
from typing import Any, Literal
from guidance.models import Model
from guidance import instruction, user, system, assistant

class GuidedLM(LM):
    lm: Model
    model_type: Literal["chat", "text", "instruct"]

    def _init_(self, model: str):
        super()._init_(model)

    def basic_request(self, prompt: str, **kwargs):
        kwargs = {**self.kwargs, **kwargs}

        request_info = {}        
        choices = []
        for i in range(kwargs["n"]):
            lmresult = self.lm + prompt    
            choices.append(lmresult._variables)
        
        request_info["choices"] = choices

        '''
        history = {
            "prompt": prompt,
            "response": request_info,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }

        self.history.append(history)
        '''
        return request_info

    def request(self, prompt: str, **kwargs):
         return self.basic_request(prompt, **kwargs)

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from Ollama.

        Args:
            prompt (str): prompt to send to Ollama
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """
#        print(f"Prompt: {prompt}")
        assert only_completed, "for now"
        assert return_sorted is False, "for now"
        response = self.request(prompt, **kwargs)
        completions = response["choices"]
        return completions

    

