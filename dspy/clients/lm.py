import functools
import os
import uuid
from datetime import datetime
from pathlib import Path

import litellm
import ujson
from litellm.caching import Cache

disk_cache_dir = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=disk_cache_dir, type="disk")
litellm.telemetry = False

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class LM:
    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

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
        completion = self._get_completion_func(cache=cache)

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


    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.__dict__, **kwargs}
        return self.__class__(**kwargs)

    def _get_completion_func(self, cache: bool = False):
        if self.model_type == "chat":
            return cached_litellm_completion if cache else litellm_completion
        else:
            return cached_litellm_text_completion if cache else litellm_text_completion


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


class RoutedLM(LM):
    """LM which uses LiteLLM Router to perform completion requests"""

    def __init__(self, model, router, **kwargs):
        # Type checking that router must be a litellm.Router instance, with model in router.model_names
        if not isinstance(router, litellm.router.Router):
            raise TypeError(
                f"The 'router' argument must be an instance of {litellm.router.Router.__name__}, but received a type '{type(router).__name__}' instead."
            )
        # Check if model is supported by the router
        available_models = router.get_model_names()
        if model not in available_models:
            raise ValueError(
                f"The model '{model}' must be one of the router's model_names. Available models on router: {available_models}"
            )

        super().__init__(model, **kwargs)
        self.router = router

    def _get_completion_func(self, cache):
        if self.model_type == "chat":
            return self._cached_router_completion if cache else self._router_completion
        else:
            return (
                self._cached_router_text_completion
                if cache
                else self._router_text_completion
            )

    @functools.lru_cache(maxsize=None)
    def _cached_router_completion(self, request):
        """Cache-enabled completion method that uses the router."""
        return self._router_completion(
            request, cache={"no-cache": False, "no-store": False}
        )

    def _router_completion(self, request, cache={"no-cache": True, "no-store": True}):
        """Actual completion logic using the router."""
        kwargs = ujson.loads(request)
        return self.router.completion(cache=cache, **kwargs)

    @functools.lru_cache(maxsize=None)
    def _cached_router_text_completion(self, request):
        return self._router_text_completion(
            request, cache={"no-cache": False, "no-store": False}
        )

    def _router_text_completion(
        self, request, cache={"no-cache": True, "no-store": True}
    ):
        kwargs = ujson.loads(request)

        # The model alias for litellm.Router assigned by user, not the official model name
        model_name = kwargs.pop("model")
        prompt = "\n\n".join(
            [x["content"] for x in kwargs.pop("messages")] + ["BEGIN RESPONSE:"]
        )

        return self.router.text_completion(
            cache=cache, model=model_name, prompt=prompt, **kwargs
        )
