"""Route `dspy.LM` to concrete normalized language model backends."""

from __future__ import annotations

import inspect
import os
import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

from dspy.clients.base_lm import BaseLM

LMBackendFactory = Callable[..., BaseLM | None]

_BACKEND_FACTORIES: list[LMBackendFactory] = []


@overload
def register_lm_backend(factory: LMBackendFactory, /) -> LMBackendFactory: ...


@overload
def register_lm_backend(factory: type[BaseLM], /, *, prefix: str) -> type[BaseLM]: ...


def register_lm_backend(factory: LMBackendFactory | type[BaseLM], /, *, prefix: str | None = None):
    """Register a backend used by `dspy.LM`.

    Register either a routing factory or a `BaseLM` subclass with a
    model-prefix. A factory receives the same constructor arguments as `LM` and
    returns a `BaseLM` when it owns the model, or `None` to let the next
    factory try.

    Examples:
        ```python
        @dspy.register_lm_backend
        def route_acme(model: str, **kwargs):
            if model.startswith("acme/"):
                return AcmeLM(model, **kwargs)
            return None

        dspy.register_lm_backend(AcmeLM, prefix="acme")
        ```
    """
    if prefix is None:
        _BACKEND_FACTORIES.append(factory)  # type: ignore[arg-type]
        return factory

    if not inspect.isclass(factory) or not issubclass(factory, BaseLM):
        raise TypeError("`prefix=` registration requires a BaseLM subclass.")
    normalized_prefix = prefix.rstrip("/") + "/"

    def route_prefixed_backend(model: str, *args: Any, **kwargs: Any) -> BaseLM | None:
        if model.startswith(normalized_prefix):
            return factory(model, *args, **kwargs)
        return None

    _BACKEND_FACTORIES.append(route_prefixed_backend)
    return factory


def LMRouter(  # noqa: N802
    model: str | None = None,
    *args: Any,
    api_key: str | None = None,
    api_base: str | None = None,
    base_url: str | None = None,
    endpoint_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    callbacks: list[Any] | None = None,
    num_retries: int | None = None,
    backend: BaseLM | None = None,
    **kwargs: Any
) -> BaseLM:
    """Create a concrete normalized language model backend.

    Args:
        model: Model name or deployment identifier. Required unless `backend`
            is supplied.
        *args: Positional arguments forwarded to registered backend constructors.
        api_key: Optional provider API key. If omitted, concrete backends use
            their provider-specific environment variables when available.
        api_base: Optional provider or OpenAI-compatible API base URL.
        base_url: Deprecated alias for `api_base`.
        endpoint_url: Optional complete OpenAI-compatible endpoint URL. When
            supplied, DSPy infers chat completions vs responses from the suffix.
        temperature: Default sampling temperature for this LM.
        max_tokens: Default output-token budget for this LM.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks attached to this LM instance.
        num_retries: Number of retry attempts for retryable provider errors. If omitted, the selected backend uses its own default.
        backend: Optional prebuilt backend. When supplied, returned as-is.
        **kwargs: Additional constructor arguments for the selected backend.

    Returns:
        A concrete `BaseLM` instance.
    """
    if backend is not None:
        return backend
    if model is None:
        raise TypeError("LM requires `model` unless `backend` is provided.")
    api_base = _resolve_api_base_alias(api_base=api_base, base_url=base_url, kwargs=kwargs)
    _reject_router_only_kwargs(kwargs)
    _reject_legacy_lm_only_kwargs(kwargs)
    explicit_kwargs = _explicit_router_kwargs(
        api_key=api_key,
        api_base=api_base,
        endpoint_url=endpoint_url,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        callbacks=callbacks,
        num_retries=num_retries,
    )
    return _route_lm_backend(model, *args, **{**kwargs, **explicit_kwargs})


LM = LMRouter


def _explicit_router_kwargs(
    *,
    api_key: str | None,
    api_base: str | None,
    endpoint_url: str | None,
    temperature: float | None,
    max_tokens: int | None,
    cache: bool,
    callbacks: list[Any] | None,
    num_retries: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"cache": cache}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if api_base is not None:
        kwargs["api_base"] = api_base
    if endpoint_url is not None:
        kwargs["endpoint_url"] = endpoint_url
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    if num_retries is not None:
        kwargs["num_retries"] = num_retries
    return kwargs


def _resolve_api_base_alias(
    *,
    api_base: str | None,
    base_url: str | None,
    kwargs: dict[str, Any],
) -> str | None:
    if "base_url" in kwargs:
        if base_url is not None and kwargs["base_url"] != base_url:
            raise ValueError("Pass `base_url` only once.")
        base_url = kwargs.pop("base_url")
    if base_url is None:
        return api_base
    if api_base is not None and api_base != base_url:
        raise ValueError("Pass only one of `api_base` or deprecated `base_url`, not both.")
    warnings.warn(
        "`base_url` is deprecated for normalized `dspy.LM`; use `api_base` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return base_url


def _reject_router_only_kwargs(kwargs: dict[str, Any]) -> None:
    if "model_type" not in kwargs:
        return
    raise TypeError(
        "`model_type` is not supported by the normalized `dspy.LM` router. "
        "Instantiate `dspy.LiteLLMLM(..., model_type=...)`, `dspy.OpenAIChatLM`, "
        "`dspy.OpenAITextLM`, or `dspy.OpenAIResponsesLM` directly."
    )


def _reject_legacy_lm_only_kwargs(kwargs: dict[str, Any]) -> None:
    legacy_only = {
        "provider",
        "finetuning_model",
        "launch_kwargs",
        "train_kwargs",
        "use_developer_role",
    }
    present = sorted(legacy_only & set(kwargs))
    if not present:
        return
    names = ", ".join(f"`{name}`" for name in present)
    raise TypeError(
        f"{names} are legacy `dspy.LM` constructor arguments and are not supported by the normalized LM router. "
        "Instantiate the legacy `dspy.clients.lm.LM` directly, or use a concrete `BaseLM` backend."
    )


def _route_lm_backend(
    model: str,
    *args: Any,
    **kwargs: Any
) -> BaseLM:
    for factory in reversed(_BACKEND_FACTORIES):
        backend = factory(model, *args, **kwargs)
        if backend is not None:
            return backend
    return _default_builtin_backend(model, *args, **kwargs)


def _default_builtin_backend(
    model: str,
    **kwargs: Any
) -> BaseLM:
    from dspy.clients.language_models.anthropic import AnthropicLM
    from dspy.clients.language_models.gemini import GenAILM
    from dspy.clients.language_models.litellm import LiteLLMLM
    from dspy.clients.language_models.openai import OpenAIChatLM, OpenAIResponsesLM, OpenAITextLM

    endpoint_model_type = _model_type_from_endpoint_url(kwargs.get("endpoint_url"))

    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if provider == "anthropic":
        return AnthropicLM(model=model, **kwargs)
    if provider in {"gemini", "google", "genai"}:
        return GenAILM(model=model, **kwargs)

    routed = _known_openai_compatible_backend(
        model,
        provider=provider,
        endpoint_model_type=endpoint_model_type,
        **kwargs,
    )
    if routed is not None:
        return routed

    if endpoint_model_type == "responses":
        return OpenAIResponsesLM(model=model, **kwargs)
    if endpoint_model_type == "chat":
        return OpenAIChatLM(model=model, **kwargs)
    if endpoint_model_type == "text":
        return OpenAITextLM(model=model, **kwargs)

    if provider == "openai":
        if endpoint_model_type == "chat":
            return OpenAIChatLM(model=model, **kwargs)
        if endpoint_model_type == "text":
            return OpenAITextLM(model=model, **kwargs)
        return OpenAIResponsesLM(model=model, **kwargs)

    return LiteLLMLM(model=model, model_type="chat", **kwargs)


def _known_openai_compatible_backend(
    model: str,
    *,
    provider: str,
    endpoint_model_type: Literal["chat", "text", "responses"] | None,
    **kwargs: Any,
) -> BaseLM | None:
    from dspy.clients.language_models.openai import OpenAIChatLM, OpenAIResponsesLM, OpenAITextLM

    route = _KNOWN_OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if route is None:
        return None

    provider_model = model.split("/", 1)[1] if "/" in model else model
    kwargs.setdefault("api_base", route["api_base"])
    if kwargs.get("api_key") is None:
        key = os.environ.get(route["api_key_env"]) if route.get("api_key_env") else None
        kwargs["api_key"] = key or route.get("default_api_key")

    selected = endpoint_model_type or _default_model_type_for_provider_model(provider, provider_model)
    if selected == "responses":
        return OpenAIResponsesLM(model=provider_model, **kwargs)
    if selected == "chat":
        return OpenAIChatLM(model=provider_model, **kwargs)
    if selected == "text":
        return OpenAITextLM(model=provider_model, **kwargs)
    return None


def _default_model_type_for_provider_model(provider: str, provider_model: str) -> Literal["chat", "responses"]:
    if provider == "groq" and provider_model.startswith("openai/gpt-oss"):
        return "responses"
    return "chat"


def _model_type_from_endpoint_url(endpoint_url: Any) -> Literal["chat", "text", "responses"] | None:
    if not isinstance(endpoint_url, str):
        return None
    path = endpoint_url.rstrip("/")
    if path.endswith("/responses"):
        return "responses"
    if path.endswith("/chat/completions"):
        return "chat"
    if path.endswith("/completions"):
        return "text"
    return None


_KNOWN_OPENAI_COMPATIBLE_PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {"api_base": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "ollama": {"api_base": "http://localhost:11434/v1", "api_key_env": "", "default_api_key": "ollama"},
    "openrouter": {"api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "fireworks": {"api_base": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "fireworks_ai": {"api_base": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "together": {"api_base": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "together_ai": {"api_base": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "deepinfra": {"api_base": "https://api.deepinfra.com/v1/openai", "api_key_env": "DEEPINFRA_API_KEY"},
}
