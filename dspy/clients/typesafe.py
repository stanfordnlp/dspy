"""System One client for the non-generative TypeSafe API."""

import copy
import json
import os

import dspy
from dspy import lm15
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, record_history
from dspy.dsp.utils.settings import settings
from dspy.utils.annotation import experimental
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history


@experimental
class TypeSafe:
    """Call TypeSafe's System One API with DSPy's shared request cache.

    Uses bundled lm15. Model, API key, and endpoint
    default to TYPESAFE_DEFAULT_MODEL, TYPESAFE_API_KEY, and TYPESAFE_BASE_URL.
    Configure with ``dspy.configure(lm=dspy.experimental.TypeSafe(...))`` or pass
    ``lm=`` when calling a predictor. Predict selects System One translation automatically.

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

    def _config(self):
        return lm15.RouterConfig(
            api_keys={"typesafe": self.api_key} if self.api_key else None,
            base_urls={"typesafe": self.base_url},
            timeouts=lm15.Timeouts(connect=self.timeout, read=self.timeout, write=self.timeout, pool=self.timeout),
        )

    def _lm15_request(self, state, questions):
        properties = {}
        for name, question in questions.items():
            kind = question["type"]
            if kind == "noul":
                properties[name] = lm15.yes_no(name)
            elif kind == "score":
                properties[name] = lm15.score(name, [str(i) for i in range(len(question["criteria"]))])
            else:
                properties[name] = lm15.choice(name, list(question["criteria"]))
        # The schema declares answer spaces; native questions retain structured instructions and criteria.
        response_format = {"type": "json_schema", "schema": {"type": "object", "properties": properties}}
        return lm15.Request(
            model=f"typesafe:{self.model}", messages=(lm15.Message.user(lm15.DataPart(value=state)),),
            config=lm15.Config(response_format=response_format, extensions={"questions": questions}),
        )

    @staticmethod
    def _response(response):
        return {
            "model": response.model,
            "usage": {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens},
            "answers": {
                name: {
                    key: answer[key]
                    for key in ("noul", "score", "choice", "confidence", "probabilities")
                    if key in answer
                }
                for name, answer in response.provider_data["typesafe"]["answers"].items()
            },
        }

    @with_callbacks
    def __call__(self, state, questions):
        request = self._request(state, questions)
        response = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if not cache_hit:
            with lm15.LMRouter(self._config()) as client:
                response = self._response(client.complete(self._lm15_request(state, questions)))
        return self._finish(request, response, cache_hit)

    @with_callbacks
    async def acall(self, state, questions):
        """Async equivalent using lm15's native asynchronous client."""
        request = self._request(state, questions)
        response = dspy.cache.get(request) if self.cache else None
        cache_hit = response is not None
        if not cache_hit:
            async with lm15.AsyncLMRouter(self._config()) as client:
                response = self._response(await client.complete(self._lm15_request(state, questions)))
        return self._finish(request, response, cache_hit)
