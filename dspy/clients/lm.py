import functools
from .base_lm import BaseLM
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm
import ujson
from litellm.caching import Cache

from dspy.clients.finetune import FinetuneJob, TrainingMethod
from dspy.clients.lm_finetune_utils import (
    execute_finetune_job,
    get_provider_finetune_job_class,
)
from dspy.utils.callback import with_callbacks
from dspy.utils.logging import logger

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")
litellm.telemetry = False

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

GLOBAL_HISTORY = []

class LM(BaseLM):
    def __init__(
            self, 
            model,
            model_type='chat', 
            temperature=0.0,
            max_tokens=1000,
            cache=True,
            launch_kwargs=None,
            callbacks=None,
            **kwargs
        ):
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.launch_kwargs = launch_kwargs or {}
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self.callbacks = callbacks or []

        # TODO: Arbitrary model strings could include the substring "o1-". We
        # should find a more robust way to check for the "o1-" family models.
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
        GLOBAL_HISTORY.append(entry)
        
        return outputs

    def launch(self):
        """Send a request to the provider to launch the model, if needed."""
        msg = f"`launch()` is called for the auto-launched model {self.model}"
        msg += " -- no action is taken!"
        logger.info(msg)

    def kill(self):
        """Send a request to the provider to kill the model, if needed."""
        msg = f"`kill()` is called for the auto-launched model {self.model}"
        msg += " -- no action is taken!"
        logger.info(msg)

    def finetune(
            self,
            train_data: List[Dict[str, Any]],
            train_kwargs: Optional[Dict[str, Any]]=None,
            train_method: TrainingMethod = TrainingMethod.SFT,
            provider: str = "openai",
            cache_finetune: bool = True,
        ) -> FinetuneJob:
        """Start model fine-tuning, if supported."""
        from dspy import settings as settings
        err = "Fine-tuning is an experimental feature."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        assert settings.experimental, err

        FinetuneJobClass = get_provider_finetune_job_class(provider=provider)
        finetune_job = FinetuneJobClass(
            model=self.model,
            train_data=train_data,
            train_kwargs=train_kwargs,
            train_method=train_method,
            provider=provider
        )

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(
            execute_finetune_job,
            finetune_job,
            lm=self,
            cache_finetune=cache_finetune
        )
        executor.shutdown(wait=False)

        return finetune_job
    
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
