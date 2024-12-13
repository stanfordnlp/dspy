import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import litellm

from dspy.adapters.base import Adapter
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import DataFormat, infer_data_format, validate_data_format
from dspy.utils.callback import BaseCallback, with_callbacks

from .base_lm import BaseLM

from dspy.utils.cache import cache_decorator

logger = logging.getLogger(__name__)


class LM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        callbacks: Optional[List[BaseCallback]] = None,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a new language model instance for use with DSPy modules and programs.

        Args:
            model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
                   supported by LiteLLM. For example, ``"openai/gpt-4o"``.
            model_type: The type of the model, either ``"chat"`` or ``"text"``.
            temperature: The sampling temperature to use when generating responses.
            max_tokens: The maximum number of tokens to generate per response.
            cache: Whether to cache the model responses for reuse to improve performance
                   and reduce costs.
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                         network error, rate limiting, etc. Requests are retried with exponential
                         backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
        """
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self.callbacks = callbacks or []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs

        # TODO(bug): Arbitrary model strings could include the substring "o1-".
        # We should find a more robust way to check for the "o1-" family models.
        if "o1-" in model:
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        # Make the request and handle LRU & disk caching.
        completion = litellm_completion if self.model_type == "chat" else litellm_text_completion
        wrapped_completion = completion

        if cache:
            @cache_decorator(keep=litellm.Cache()._get_relevant_args_to_use_for_cache_key())
            def cached_completion(**kwargs):
                return completion(**kwargs, cache={"no-cache": False, "no-store": False})
            
            wrapped_completion = cached_completion

        response = wrapped_completion(model=self.model, messages=messages, num_retries=self.num_retries, **kwargs)
        
        if kwargs.get("logprobs"):
            outputs = [
                {
                    "text": c.message.content if hasattr(c, "message") else c["text"],
                    "logprobs": c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
                }
                for c in response["choices"]
            ]
        else:
            outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in response["choices"]]


        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = dict(prompt=prompt, messages=messages, kwargs=kwargs, response=response)
        entry = dict(**entry, outputs=outputs, usage=dict(response["usage"]))
        entry = dict(**entry, cost=response.get("_hidden_params", {}).get("response_cost"))
        entry = dict(
            **entry,
            timestamp=datetime.now().isoformat(),
            uuid=str(uuid.uuid4()),
            model=self.model,
            model_type=self.model_type,
        )
        self.history.append(entry)
        self.update_global_history(entry)

        return outputs

    def launch(self, launch_kwargs: Optional[Dict[str, Any]] = None):
        launch_kwargs = launch_kwargs or self.launch_kwargs
        self.provider.launch(self.model, launch_kwargs)

    def kill(self, launch_kwargs: Optional[Dict[str, Any]] = None):
        launch_kwargs = launch_kwargs or self.launch_kwargs
        self.provider.kill(self.model, launch_kwargs)

    def finetune(
        self,
        train_data: List[Dict[str, Any]],
        train_kwargs: Optional[Dict[str, Any]] = None,
        data_format: Optional[DataFormat] = None,
    ) -> TrainingJob:
        from dspy import settings as settings

        err = "Fine-tuning is an experimental feature."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        assert settings.experimental, err

        err = f"Provider {self.provider} does not support fine-tuning."
        assert self.provider.finetunable, err

        # Perform data validation before starting the thread to fail early
        train_kwargs = train_kwargs or {}
        if not data_format:
            adapter = self.infer_adapter()
            data_format = infer_data_format(adapter)
        validate_data_format(data=train_data, data_format=data_format)

        # TODO(PR): We can quickly add caching, but doing so requires
        # adding functions that just call other functions as we had in the last
        # iteration, unless people have other ideas.
        def thread_function_wrapper():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=thread_function_wrapper)
        model_to_finetune = self.finetuning_model or self.model
        job = self.provider.TrainingJob(
            thread=thread,
            model=model_to_finetune,
            train_data=train_data,
            train_kwargs=train_kwargs,
            data_format=data_format,
        )
        thread.start()

        return job

    def _run_finetune_job(self, job: TrainingJob):
        # TODO(enhance): We should listen for keyboard interrupts somewhere.
        # Requires TrainingJob.cancel() to be implemented for each provider.
        try:
            model = self.provider.finetune(
                job=job,
                model=job.model,
                train_data=job.train_data,
                train_kwargs=job.train_kwargs,
                data_format=job.data_format,
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.error(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        # TODO(PR): Keeping this function here will require us to import all
        # providers in this file. Is this okay?
        return Provider()

    def infer_adapter(self) -> Adapter:
        import dspy

        if dspy.settings.adapter:
            return dspy.settings.adapter

        model_type_to_adapter = {
            "chat": dspy.ChatAdapter(),
        }
        model_type = self.model_type
        return model_type_to_adapter[model_type]

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters."""

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                new_instance.kwargs[key] = value

        return new_instance


def litellm_completion(cache={"no-cache": True, "no-store": True}, **kwargs):
    return litellm.completion(cache=cache, **kwargs)


def litellm_text_completion(cache={"no-cache": True, "no-store": True}, **kwargs):
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = kwargs.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = kwargs.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = kwargs.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in kwargs.pop("messages")] + ["BEGIN RESPONSE:"])

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        **kwargs,
    )
