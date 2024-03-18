from typing import Any, Literal

from dsp.modules.guided_lm import GuidedLM

from guidance import models


class GuidedLlama(GuidedLM):
    """Wrapper around a local llama model with guidance.

    Args:
        model (str, optional): Name of Ollama model. Defaults to "llama2".
        model_type (Literal["chat", "text"], optional): The type of model that was specified. Mainly to decide the optimal prompting strategy. Defaults to "text".
        model_path (str):  Path to the llama model file. Defaults to "llama2.gguf".
        **kwargs: Additional arguments to pass to the API.
    """

    def __init__(
        self,
        model: str = "llama2",
        model_path: str = "llama2.gguf",
        model_type: Literal["chat", "text", "instruct"] = "text",
        temperature: float = 0.0,
        max_tokens: int = 150,
        top_p: int = 1,
        top_k: int = 20,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        n: int = 1,
        num_ctx: int = 1024,
        **kwargs,
    ):
        all_kwargs = {
            "model_type": model_type,
            "model_path": model_path,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "n": n,
            "num_ctx": num_ctx,
            **kwargs,
        }
        super().__init__(model)

        if model_type == "chat":
            self.lm = models.LlamaCppChat(model_path, echo=False, temperature=temperature, n_ctx=num_ctx, **kwargs)
        else:
            self.lm = models.LlamaCpp(model_path, echo=False, temperature=temperature, n_ctx=num_ctx, **kwargs)

        self.kwargs = all_kwargs
        self.provider = f"{self.lm.__class__.__name__}"
        self.model_type = model_type
        self.mode_path = model_path
        self.model_name = model


