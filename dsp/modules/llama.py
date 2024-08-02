from typing import Any, Literal
from dsp.modules.lm import LM

class LlamaCpp(LM):
    def __init__(
            self,
            model: str,  # "llama" or the actual model name
            llama_model: Any = None,
            model_type: Literal["chat", "text"] = None,
            **kwargs,
    ):
        super().__init__(model)

        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModuleNotFoundError(
                """You need to install the llama_cpp library to use gguf models.
                CPU - pip install llama-cpp-python
                CUDA - pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version> e.g. cu121 for CUDA 12.1
                METAL(Mac) - pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
                others: https://pypi.org/project/llama-cpp-python/""",
            ) from exc

        default_model_type = "text"
        self.model_type = model_type if model_type else default_model_type
        self.provider = "llama"

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.loaded_model = llama_model
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if "n" in kwargs:
            del kwargs["n"]

        if self.model_type == "chat":
            kwargs["messages"] = [{"role": "user", "content": prompt}]
            response = self.loaded_model.create_chat_completion(**kwargs)

        else:
            kwargs["prompt"] = prompt
            response = self.loaded_model.create_completion(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        return response

    def request(self, prompt: str, **kwargs):
        if "model_type" in kwargs:
            del kwargs["model_type"]

        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        response = self.request(prompt, **kwargs)
        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]

        return completions
