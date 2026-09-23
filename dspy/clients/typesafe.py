"""System One client for the non-generative TypeSafe API."""

import copy
import json
import os

import dspy
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, record_history
from dspy.dsp.utils.settings import settings
from dspy.utils.annotation import experimental
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history


@experimental
class TypeSafe:
    """Call TypeSafe's System One API with DSPy's shared request cache.

    Install ``dspy[typesafe]`` to use this client. Model, API key, and endpoint
    default to TYPESAFE_DEFAULT_MODEL, TYPESAFE_API_KEY, and TYPESAFE_BASE_URL.
    Configure with ``dspy.configure(lm=dspy.experimental.TypeSafe(...))`` or pass
    ``lm=`` to Predict. Predict selects System One translation automatically.

    Args:
        model: Model name (defaults to ``jev-latest``).
        api_key: Credential; omitted from saved state and history.
        base_url: Endpoint (defaults to ``https://api.typesafe.ai``).
        cache: Whether to use DSPy's cache.
        timeout: Per-operation timeout in seconds.
    """

    supports_decision_requests = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: bool = True,
        timeout: float = 10.0,
        callbacks=None,
    ):
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        self.callbacks = list(callbacks or [])
        self.base_url = (base_url or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        self.api_key = api_key
        self.cache = cache
        self.timeout = timeout
        self.history = []

    def copy(self, **kwargs):
        """Copy connection settings and callbacks, starting with empty history."""
        state = self.dump_state()
        state.pop(LM_CLASS_STATE_KEY)
        return type(self)(**{**state, "api_key": self.api_key, "callbacks": self.callbacks, **kwargs})

    def inspect_history(self, n=1, file=None):
        pretty_print_history(self.history, n, file=file)

    @classmethod
    def load_state(cls, state):
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        return cls(**state)

    def dump_state(self):
        """Return reconstruction settings, without credentials or runtime history."""
        return {
            LM_CLASS_STATE_KEY: "dspy.clients.typesafe.TypeSafe",
            "model": self.model,
            "base_url": self.base_url,
            "cache": self.cache,
            "timeout": self.timeout,
        }

    def _request(self, state, questions):
        return {
            "provider": "typesafe",
            "model": self.model,
            "base_url": self.base_url,
            "state": state,
            "questions": questions,
        }

    def _finish(self, request, response, cache_hit):
        if self.cache and not cache_hit:
            dspy.cache.put(request, response)
        usage = {} if cache_hit else response["usage"]
        if settings.usage_tracker and usage:
            settings.usage_tracker.add_usage(response["model"], usage)
        if not settings.disable_history:
            record_history(
                self,
                {
                    "request": copy.deepcopy(request),
                    "response": copy.deepcopy(response),
                    "usage": usage,
                    "cache_hit": cache_hit,
                    "messages": None,
                    "prompt": json.dumps({"state": request["state"], "questions": request["questions"]}),
                    "outputs": [json.dumps(response["answers"])],
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

    @with_callbacks
    def __call__(self, state, questions):
        request = self._request(state, questions)
        response = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if not cache_hit:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError:
                raise ImportError('Install TypeSafe support with `pip install "dspy[typesafe]"`.') from None
            with TypeSafeClient(**self._sdk_kwargs()) as client:
                response = self._response(client.system_one(state=state, questions=questions))
        return self._finish(request, response, cache_hit)

    @with_callbacks
    async def acall(self, state, questions):
        """Async equivalent using the SDK's native asynchronous client."""
        request = self._request(state, questions)
        response = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if not cache_hit:
            try:
                from typesafe_sdk import AsyncTypeSafeClient
            except ImportError:
                raise ImportError('Install TypeSafe support with `pip install "dspy[typesafe]"`.') from None
            async with AsyncTypeSafeClient(**self._sdk_kwargs()) as client:
                response = self._response(await client.system_one(state=state, questions=questions))
        return self._finish(request, response, cache_hit)
