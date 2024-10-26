from datetime import datetime
import functools
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional
import ujson
import uuid

from dspy.adapters.base import Adapter
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.openai import OpenAIProvider
from dspy.clients.utils_finetune import (
  DataFormat,
  validate_data_format,
  infer_data_format
)
from dspy.utils.logging import logger

import litellm
from litellm.caching import Cache


DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")
litellm.telemetry = False

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class LM:
    def __init__(
            self, 
            model,
            model_type='chat', 
            temperature=0.0,
            max_tokens=1000,
            cache=True,
            launch_kwargs=None,
            provider=None,
            **kwargs
        ):
        if launch_kwargs is not None or provider is not None:
            import dspy
            assert dspy.settings.experimental, "The `launch_kwargs` and `provider` arguments are experimental. Set `dspy.settings.experimental` to `True` to use them."

        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.launch_kwargs = launch_kwargs or {}
        self.provider = provider or self.infer_provider()
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

        # TODO(bug): Arbitrary model strings could include the substring "o1-".
        # We should find a more robust way to check for the "o1-" family models.
        if "o1-" in model:
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        # Make the request and handle LRU & disk caching.
        if self.model_type == "chat":
            completion = cached_litellm_completion if cache else litellm_completion
        else:
            completion = cached_litellm_text_completion if cache else litellm_text_completion

        response = completion(ujson.dumps(dict(model=self.model, messages=messages, **kwargs)))
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
        
        return outputs
    
    def inspect_history(self, n: int = 1):
        _inspect_history(self, n)

    def launch(self):
        self.provider.launch(self.model, self.launch_kwargs)

    def kill(self):
        self.provider.kill(self.model, self.launch_kwargs)

    def finetune(
            self,
            train_data: List[Dict[str, Any]],
            train_kwargs: Optional[Dict[str, Any]]=None,
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
        job = self.provider.TrainingJob(
            thread=thread,
            model=self.model,
            train_data=train_data,
            train_kwargs=train_kwargs,
            data_format=data_format
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
                data_format=job.data_format
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.error(err)
            job.set_result(err)
    
    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        # TODO(PR): Should we handle AnyScale models here
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



@functools.lru_cache(maxsize=None)
def cached_litellm_completion(request):
    return litellm_completion(request, cache={"no-cache": False, "no-store": False})


def litellm_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)
    return litellm.completion(cache=cache, **kwargs)


@functools.lru_cache(maxsize=None)
def cached_litellm_text_completion(request):
    return litellm_text_completion(request, cache={"no-cache": False, "no-store": False})


def litellm_text_completion(request, cache={"no-cache": True, "no-store": True}):
    kwargs = ujson.loads(request)

    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = kwargs.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the kwargs, or from the environment.
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


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _inspect_history(lm, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in lm.history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")
