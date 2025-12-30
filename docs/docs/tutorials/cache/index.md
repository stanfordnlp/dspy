# Use and Customize DSPy Cache

In this tutorial, we will explore the design of DSPy's caching mechanism and demonstrate how to effectively use and customize it.

## DSPy Cache Structure

DSPy's caching system is architected in three distinct layers:

1.  **In-memory cache**: Implemented using `cachetools.LRUCache`, this layer provides fast access to frequently used data.
2.  **On-disk cache**: Leveraging `diskcache.FanoutCache`, this layer offers persistent storage for cached items.
3.  **Prompt cache (Server-side cache)**: This layer is managed by the LLM service provider (e.g., OpenAI, Anthropic).

While DSPy does not directly control the server-side prompt cache, it offers users the flexibility to enable, disable, and customize the in-memory and on-disk caches to suit their specific requirements.

## Using DSPy Cache

By default, both in-memory and on-disk caching are automatically enabled in DSPy. No specific action is required to start using the cache. When a cache hit occurs, you will observe a significant reduction in the module call's execution time. Furthermore, if usage tracking is enabled, the usage metrics for a cached call will be `None`.

Consider the following example:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")
```

A sample output looks like:

```
Time elapse:  4.384113
Total usage: {'openai/gpt-4o-mini': {'completion_tokens': 97, 'prompt_tokens': 144, 'total_tokens': 241, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0, 'text_tokens': None, 'image_tokens': None}}}

Time elapse:  0.000529
Total usage: {}
```

## Using Provider-Side Prompt Caching

In addition to DSPy's built-in caching mechanism, you can leverage provider-side prompt caching offered by LLM providers like Anthropic and OpenAI. This feature is particularly useful when working with modules like `dspy.ReAct()` that send similar prompts repeatedly, as it reduces both latency and costs by caching prompt prefixes on the provider's servers.

You can enable prompt caching by passing the `cache_control_injection_points` parameter to `dspy.LM()`. This works with supported providers like Anthropic and OpenAI. For more details on this feature, see the [LiteLLM prompt caching documentation](https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration).

```python
import dspy
import os

os.environ["ANTHROPIC_API_KEY"] = "{your_anthropic_key}"
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[
        {
            "location": "message",
            "role": "system",
        }
    ],
)
dspy.configure(lm=lm)

# Use with any DSPy module
predict = dspy.Predict("question->answer")
result = predict(question="What is the capital of France?")
```

This is especially beneficial when:

- Using `dspy.ReAct()` with the same instructions
- Working with long system prompts that remain constant
- Making multiple requests with similar context

## Disabling/Enabling DSPy Cache

There are scenarios where you might need to disable caching, either entirely or selectively for in-memory or on-disk caches. For instance:

- You require different responses for identical LM requests.
- You lack disk write permissions and need to disable the on-disk cache.
- You have limited memory resources and wish to disable the in-memory cache.

DSPy provides the `dspy.configure_cache()` utility function for this purpose. You can use the corresponding flags to control the enabled/disabled state of each cache type:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

In additions, you can manage the capacity of the in-memory and on-disk caches:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

Please note that `disk_size_limit_bytes` defines the maximum size in bytes for the on-disk cache, while `memory_max_entries` specifies the maximum number of entries for the in-memory cache.

## Understanding and Customizing the Cache

In specific situations, you might want to implement a custom cache, for example, to gain finer control over how cache keys are generated. By default, the cache key is derived from a hash of all request arguments sent to `litellm`, excluding credentials like `api_key`.

To create a custom cache, you need to subclass `dspy.clients.Cache` and override the relevant methods:

```python
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {write your own constructor}

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {write your logic of computing cache key}

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {write your cache read logic}

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {write your cache write logic}
```

To ensure seamless integration with the rest of DSPy, it is recommended to implement your custom cache using the same method signatures as the base class, or at a minimum, include `**kwargs` in your method definitions to prevent runtime errors during cache read/write operations.

Once your custom cache class is defined, you can instruct DSPy to use it:

```python
dspy.cache = CustomCache()
```

Let's illustrate this with a practical example. Suppose we want the cache key computation to depend solely on the request message content, ignoring other parameters like the specific LM being called. We can create a custom cache as follows:

```python
class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)
```

For comparison, consider executing the code below without the custom cache:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")
```

The time elapsed will indicate that the cache is not hit on the second call. However, when using the custom cache:

```python
import dspy
import os
import time
from typing import Dict, Any, Optional
import ujson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")
```

You will observe that the cache is hit on the second call, demonstrating the effect of the custom cache key logic.