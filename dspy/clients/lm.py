from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

from litellm.caching import Cache
import litellm
import ujson
import uuid

import dspy
from dspy.utils.logging import logger
from dspy.clients.finetune import FinetuneJob, TrainingMethod
from dspy.clients.lm_finetune_utils import (
    get_provider_finetune_job_class,
    execute_finetune_job,
)
from dspy.primitives.program import handle_async


DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(
    Path.home(), ".dspy_cache"
)
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")
litellm.telemetry = False

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class LM:
    def __init__(
        self,
        model,
        model_type="chat",
        temperature=0.0,
        max_tokens=1000,
        cache=True,
        launch_kwargs=None,
        **kwargs,
    ):
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.launch_kwargs = launch_kwargs or {}
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

        # TODO: Arbitrary model strings could include the substring "o1-". We
        # should find a more robust way to check for the "o1-" family models.
        if "o1-" in model:
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's o1-* models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"

    async def acall(self, prompt=None, messages=None, **kwargs):
        """Async completion method"""
        cache: bool = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = async_litellm_completion
        else:
            completion = async_litellm_text_completion

        response = await completion(
            dict(model=self.model, messages=messages, **kwargs),
            cache=cache,
        )
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        self.history.append(
            {
                "prompt": prompt,
                "messages": messages,
                "kwargs": kwargs,
                "response": response,
                "outputs": outputs,
                "usage": dict(response["usage"]),
                "cost": response.get("_hidden_params", {}).get("response_cost"),
                "timestamp": datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": self.model,
                "model_type": self.model_type,
            }
        )

        return outputs

    @handle_async
    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Main entry point that handles both sync and async calls.
        Uses handle_async decorator to automatically switch between modes.
        """
        import dspy

        if dspy.settings.async_mode:
            return self.acall(prompt, messages, **kwargs)

        # Build the request.
        cache: bool = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "chat":
            completion = litellm_completion
        else:
            completion = litellm_text_completion

        response = completion(
            dict(model=self.model, messages=messages, **kwargs),
            cache=cache,
        )
        outputs = [
            c.message.content if hasattr(c, "message") else c["text"]
            for c in response["choices"]
        ]

        # Logging, with removed api key & where `cost` is None on cache hit.
        self.history.append(
            {
                "prompt": prompt,
                "messages": messages,
                "kwargs": kwargs,
                "response": response,
                "outputs": outputs,
                "usage": dict(response["usage"]),
                "cost": response.get("_hidden_params", {}).get("response_cost"),
                "timestamp": datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": self.model,
                "model_type": self.model_type,
            }
        )

        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self, n)

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
        train_kwargs: Optional[Dict[str, Any]] = None,
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
            provider=provider,
        )

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(
            execute_finetune_job, finetune_job, lm=self, cache_finetune=cache_finetune
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


class LMRequestLRUCache(OrderedDict):
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, request: dict, value):
        key = self.cache_key(request)

        if key in self:
            self.move_to_end(key)
            return

        if len(self) == self.maxsize:
            self.popitem(last=False)

        super().__setitem__(key, value)

    def __getitem__(self, request: dict):
        key = self.cache_key(request)
        return super().__getitem__(key)

    def __contains__(self, request: dict):
        key = self.cache_key(request)
        return super().__contains__(key)

    def get(self, request: dict, default=None):
        key = self.cache_key(request)
        return super().get(key, default)

    def __delitem__(self, request: dict):
        key = self.cache_key(request)
        super().__delitem__(key)

    def pop(self, request: dict, default=None):
        key = self.cache_key(request)
        return super().pop(key, default)

    @staticmethod
    def cache_key(request: dict) -> str:
        return hashlib.sha256(ujson.dumps(request, sort_keys=True).encode()).hexdigest()


def request_cache(
    default_cache=LMRequestLRUCache(maxsize=10_000_000),
) -> LMRequestLRUCache:
    return dspy.settings.request_cache or default_cache


def litellm_completion(request: dict, cache=False):
    if not cache:
        return litellm.completion(**request, cache={"no-cache": True, "no-store": True})

    if response := request_cache().get(request, None):
        return response

    response = litellm.completion(
        **request,
        cache={"no-cache": False, "no-store": False},
    )
    request_cache()[request] = response

    return response


async def async_litellm_completion(request: dict, cache=False):
    if not cache:
        return await litellm.acompletion(
            **request,
            cache={"no-cache": True, "no-store": True},
        )

    if response := request_cache().get(request, None):
        return response

    response = await litellm.acompletion(
        **request,
        cache={"no-cache": False, "no-store": False},
    )

    request_cache()[request] = response

    return response


def _prepare_litellm_text_completion_params(request: dict):
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the kwargs, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join(
        [x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"]
    )

    return {
        "model": f"text-completion-openai/{model}",
        "api_key": api_key,
        "api_base": api_base,
        "prompt": prompt,
        **request,
    }


def litellm_text_completion(request: dict, cache=False):
    if not cache:
        return litellm.text_completion(
            **request,
            cache={"no-cache": True, "no-store": True},
        )

    params = _prepare_litellm_text_completion_params(request.copy())
    response = litellm.text_completion(**params, cache=cache)
    request_cache()[request] = response

    return response


async def async_litellm_text_completion(request: dict, cache=False):
    if not cache:
        return await litellm.atext_completion(
            **request,
            cache={"no-cache": True, "no-store": True},
        )

    params = _prepare_litellm_text_completion_params(request.copy())
    response = await litellm.atext_completion(**params, cache=cache)
    request_cache()[request] = response

    return response


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
