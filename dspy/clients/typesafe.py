"""System One client for the non-generative TypeSafe API."""

import copy
import os

from dspy.dsp.utils.settings import settings


class TypeSafe:
    """Call TypeSafe's System One API with DSPy's shared request cache.

    Install ``dspy[typesafe]`` to use this client. Model, API key, and endpoint
    default to TYPESAFE_DEFAULT_MODEL, TYPESAFE_API_KEY, and TYPESAFE_BASE_URL.
    Configure with ``dspy.configure(system_one=dspy.TypeSafe(...))`` or pass the
    client to ``Decide``. This is not a text-generating ``dspy.LM``.

    Args:
        model: Model name (defaults to ``jev-latest``).
        api_key: Credential; omitted from saved state and history.
        base_url: Endpoint (defaults to ``https://api.typesafe.ai``).
        cache: Whether to use DSPy's cache.
        timeout: Per-operation timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: bool = True,
        timeout: float = 10.0,
    ):
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        self.base_url = (base_url or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.history = []

    def dump_state(self):
        """Return reconstruction settings, without credentials or runtime history."""
        return {"model": self.model, "base_url": self.base_url, "cache": self.cache, "timeout": self.timeout}

    def _request(self, state, questions):
        return {
            "provider": "typesafe",
            "model": self.model,
            "base_url": self.base_url,
            "state": state,
            "questions": questions,
        }

    def _cached(self, request):
        import dspy

        # The endpoint is part of the identity: different deployments must not share answers.
        return dspy.cache.get(request) if self.cache else None

    def _finish(self, request, response, cache_hit):
        import dspy

        if self.cache and not cache_hit:
            dspy.cache.put(request, response)
        usage = {} if cache_hit else response["usage"]
        if settings.usage_tracker and usage:
            settings.usage_tracker.add_usage(response["model"], usage)
        if not settings.disable_history and settings.max_history_size > 0:
            self.history.append(
                {
                    "request": copy.deepcopy(request),
                    "response": copy.deepcopy(response),
                    "usage": usage,
                    "cache_hit": cache_hit,
                }
            )
            del self.history[: -settings.max_history_size]
        return copy.deepcopy(response["answers"])

    def _sdk_kwargs(self):
        return {"model": self.model, "api_key": self.api_key, "base_url": self.base_url, "timeout": self.timeout}

    @staticmethod
    def _response(response):
        # Both SDK 0.6's dataclasses and 0.7's Pydantic models expose these attributes.
        return {
            "model": response.model,
            "usage": {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens},
            "answers": {
                name: {
                    key: getattr(answer, key)
                    for key in ("noul", "score", "choice", "confidence", "probabilities")
                    if hasattr(answer, key)
                }
                for name, answer in response.answers.items()
            },
        }

    def __call__(self, state, questions):
        request = self._request(state, questions)
        response = self._cached(request)
        cache_hit = response is not None
        if not cache_hit:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError:
                raise ImportError('Install TypeSafe support with `pip install "dspy[typesafe]"`.') from None
            with TypeSafeClient(**self._sdk_kwargs()) as client:
                response = self._response(client.system_one(state=state, questions=questions))
        return self._finish(request, response, cache_hit)

    async def acall(self, state, questions):
        """Async equivalent using the SDK's native asynchronous client."""
        request = self._request(state, questions)
        response = self._cached(request)
        cache_hit = response is not None
        if not cache_hit:
            try:
                from typesafe_sdk import AsyncTypeSafeClient
            except ImportError:
                raise ImportError('Install TypeSafe support with `pip install "dspy[typesafe]"`.') from None
            async with AsyncTypeSafeClient(**self._sdk_kwargs()) as client:
                response = self._response(await client.system_one(state=state, questions=questions))
        return self._finish(request, response, cache_hit)
