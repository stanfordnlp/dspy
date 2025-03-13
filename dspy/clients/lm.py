import functools
import logging
import os
import re
import threading
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, cast

import litellm
import pydantic
import ujson
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify
from cachetools import LRUCache, cached
from litellm import RetryPolicy

import dspy
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.utils.callback import BaseCallback, with_callbacks

from .base_lm import BaseLM

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
        cache_in_memory: bool = True,
        callbacks: Optional[List[BaseCallback]] = None,
        num_retries: int = 8,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        train_kwargs: Optional[dict[str, Any]] = None,
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
            cache_in_memory: To enable additional caching with LRU in memory.
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
        self.cache_in_memory = cache_in_memory
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.callbacks = callbacks or []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}

        # Handle model-specific configuration for different model families
        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        # Match pattern: o[1,3] at the start, optionally followed by -mini and anything else
        model_pattern = re.match(r"^o([13])(?:-mini)?", model_family)

        if model_pattern:
            # Handle OpenAI reasoning models (o1, o3)
            assert (
                max_tokens >= 5000 and temperature == 1.0
            ), "OpenAI's reasoning models require passing temperature=1.0 and max_tokens >= 5000 to `dspy.LM(...)`"
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    @with_callbacks
    def forward(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        # disable cache will also disable in memory cache
        cache_in_memory = cache and kwargs.pop("cache_in_memory", self.cache_in_memory)
        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        # Make the request and handle LRU & disk caching.
        if cache_in_memory:
            completion = cached_litellm_completion if self.model_type == "chat" else cached_litellm_text_completion

            return completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
            )
        else:
            completion = litellm_completion if self.model_type == "chat" else litellm_text_completion

            return completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
                # only leverage LiteLLM cache in this case
                cache={"no-cache": not cache, "no-store": not cache},
            )

    def launch(self, launch_kwargs: Optional[Dict[str, Any]] = None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: Optional[Dict[str, Any]] = None):
        self.provider.kill(self, launch_kwargs)

    def finetune(
        self,
        train_data: List[Dict[str, Any]],
        train_data_format: Optional[TrainDataFormat],
        train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TrainingJob:
        from dspy import settings as settings

        err = "Fine-tuning is an experimental feature."
        err += " Set `dspy.settings.experimental` to `True` to use it."
        assert settings.experimental, err

        err = f"Provider {self.provider} does not support fine-tuning."
        assert self.provider.finetunable, err

        def thread_function_wrapper():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=thread_function_wrapper)
        train_kwargs = train_kwargs or self.train_kwargs
        model_to_finetune = self.finetuning_model or self.model
        job = self.provider.TrainingJob(
            thread=thread,
            model=model_to_finetune,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
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
                train_data_format=job.train_data_format,
                train_kwargs=job.train_kwargs,
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.error(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    def dump_state(self):
        state_keys = [
            "model",
            "model_type",
            "cache",
            "cache_in_memory",
            "num_retries",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
        ]
        return {key: getattr(self, key) for key in state_keys} | self.kwargs


def request_cache(maxsize: Optional[int] = None):
    """
    A threadsafe decorator to create an in-memory LRU cache for LM inference functions that accept
    a dictionary-like LM request. An in-memory cache for LM calls is critical for ensuring
    good performance when optimizing and evaluating DSPy LMs (disk caching alone is too slow).

    Args:
        maxsize: The maximum size of the cache. If unspecified, no max size is enforced (cache is unbounded).

    Returns:
        A decorator that wraps the target function with caching.
    """

    def cache_key(request: Dict[str, Any]) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.

        Note: Values that cannot be converted to JSON should *not* be ignored / discarded, since
        that would potentially lead to cache collisions. For example, consider request A
        containing only JSON-convertible values and request B containing the same JSON-convertible
        values in addition to one unconvertible value. Discarding the unconvertible value would
        lead to a cache collision between requests A and B, even though they are semantically
        different.
        """

        def transform_value(value):
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                return value.model_json_schema()
            elif isinstance(value, pydantic.BaseModel):
                return value.model_dump()
            elif callable(value) and hasattr(value, "__code__") and hasattr(value.__code__, "co_code"):
                return value.__code__.co_code.decode("utf-8")
            else:
                # Note: We don't attempt to compute a hash of the value, since the default
                # implementation of hash() is id(), which may collide if the same memory address
                # is reused for different objects at different times
                return value

        params = {k: transform_value(v) for k, v in request.items()}
        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()

    def decorator(func):
        @cached(
            # NB: cachetools doesn't support maxsize=None; it recommends using float("inf") instead
            cache=LRUCache(maxsize=maxsize or float("inf")),
            key=lambda key, request, *args, **kwargs: key,
            # Use a lock to ensure thread safety for the cache when DSPy LMs are queried
            # concurrently, e.g. during optimization and evaluation
            lock=threading.RLock(),
        )
        def func_cached(key: str, request: Dict[str, Any], *args, **kwargs):
            return func(request, *args, **kwargs)

        @functools.wraps(func)
        def wrapper(request: dict, *args, **kwargs):
            try:
                key = cache_key(request)
            except Exception:
                # If the cache key cannot be computed (e.g. because it contains a value that cannot
                # be converted to JSON), bypass the cache and call the target function directly
                return func(request, *args, **kwargs)
            return func_cached(key, request, *args, **kwargs)

        return wrapper

    return decorator


@request_cache(maxsize=None)
def cached_litellm_completion(request: Dict[str, Any], num_retries: int):
    return litellm_completion(
        request,
        cache={"no-cache": False, "no-store": False},
        num_retries=num_retries,
    )


def litellm_completion(request: Dict[str, Any], num_retries: int, cache={"no-cache": True, "no-store": True}):
    retry_kwargs = dict(
        retry_policy=_get_litellm_retry_policy(num_retries),
        # In LiteLLM version 1.55.3 (the first version that supports retry_policy as an argument
        # to completion()), the default value of max_retries is non-zero for certain providers, and
        # max_retries is stacked on top of the retry_policy. To avoid this, we set max_retries=0
        max_retries=0,
    )

    stream = dspy.settings.send_stream
    if stream is None:
        return litellm.completion(
            cache=cache,
            **retry_kwargs,
            **request,
        )

    # The stream is already opened, and will be closed by the caller.
    stream = cast(MemoryObjectSendStream, stream)

    @syncify
    async def stream_completion():
        response = await litellm.acompletion(
            cache=cache,
            stream=True,
            **retry_kwargs,
            **request,
        )
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            await stream.send(chunk)
        return litellm.stream_chunk_builder(chunks)

    return stream_completion()


@request_cache(maxsize=None)
def cached_litellm_text_completion(request: Dict[str, Any], num_retries: int):
    return litellm_text_completion(
        request,
        num_retries=num_retries,
        cache={"no-cache": False, "no-store": False},
    )


def litellm_text_completion(request: Dict[str, Any], num_retries: int, cache={"no-cache": True, "no-store": True}):
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        retry_policy=_get_litellm_retry_policy(num_retries),
        # In LiteLLM version 1.55.3 (the first version that supports retry_policy as an argument
        # to completion()), the default value of max_retries is non-zero for certain providers, and
        # max_retries is stacked on top of the retry_policy. To avoid this, we set max_retries=0
        max_retries=0,
        **request,
    )


def _get_litellm_retry_policy(num_retries: int) -> RetryPolicy:
    """
    Get a LiteLLM retry policy for retrying requests when transient API errors occur.
    Args:
        num_retries: The number of times to retry a request if it fails transiently due to
                     network error, rate limiting, etc. Requests are retried with exponential
                     backoff.
    Returns:
        A LiteLLM RetryPolicy instance.
    """
    return RetryPolicy(
        TimeoutErrorRetries=num_retries,
        RateLimitErrorRetries=num_retries,
        InternalServerErrorRetries=num_retries,
        ContentPolicyViolationErrorRetries=num_retries,
        # We don't retry on errors that are unlikely to be transient
        # (e.g. bad request, invalid auth credentials)
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )
