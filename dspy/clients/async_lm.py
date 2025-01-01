import os
from typing import Any, Awaitable, Dict, cast

import litellm
from anyio.streams.memory import MemoryObjectSendStream
from litellm.types.router import RetryPolicy

import dspy
from dspy.clients.lm import LM, request_cache
from dspy.utils import with_callbacks


class AsyncLM(LM):
    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs) -> Awaitable:
        async def _async_call(prompt, messages, **kwargs):
            # Build the request.
            cache = kwargs.pop("cache", self.cache)
            messages = messages or [{"role": "user", "content": prompt}]
            kwargs = {**self.kwargs, **kwargs}

            # Make the request and handle LRU & disk caching.
            if self.model_type == "chat":
                completion = cached_litellm_completion if cache else litellm_acompletion
            else:
                completion = cached_litellm_text_completion if cache else litellm_text_acompletion

            response = await completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
            )
            outputs = [c.message.content if hasattr(c, "message") else c["text"] for c in response["choices"]]
            self._log_entry(prompt, messages, kwargs, response, outputs)
            return outputs

        return _async_call(prompt, messages, **kwargs)


@request_cache(maxsize=None)
async def cached_litellm_completion(request: Dict[str, Any], num_retries: int):
    return await litellm_acompletion(
        request,
        cache={"no-cache": False, "no-store": False},
        num_retries=num_retries,
    )


async def litellm_acompletion(request: Dict[str, Any], num_retries: int, cache={"no-cache": True, "no-store": True}):
    retry_kwargs = dict(
        retry_policy=_get_litellm_retry_policy(num_retries),
        # In LiteLLM version 1.55.3 (the first version that supports retry_policy as an argument
        # to completion()), the default value of max_retries is non-zero for certain providers, and
        # max_retries is stacked on top of the retry_policy. To avoid this, we set max_retries=0
        max_retries=0,
    )

    stream = dspy.settings.send_stream
    if stream is None:
        return await litellm.acompletion(
            cache=cache,
            **retry_kwargs,
            **request,
        )

    # The stream is already opened, and will be closed by the caller.
    stream = cast(MemoryObjectSendStream, stream)

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

    return await stream_completion()


@request_cache(maxsize=None)
async def cached_litellm_text_completion(request: Dict[str, Any], num_retries: int):
    return await litellm_text_acompletion(
        request,
        num_retries=num_retries,
        cache={"no-cache": False, "no-store": False},
    )


async def litellm_text_acompletion(
    request: Dict[str, Any], num_retries: int, cache={"no-cache": True, "no-store": True}
):
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return await litellm.atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
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
