import logging
import re
import threading
import warnings
from typing import Any, Literal

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import ContextWindowExceededError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _backend_capability(name):
    """Property that delegates a capability query to the backend."""
    return property(lambda self: getattr(self._get_backend(), name)(self.model))


def _resolve_backend(model: str, model_type: str):
    """Resolve a backend module from a model string.

    Routes openai/* and azure/* chat models to the native OpenAI SDK backend.
    Everything else falls back to litellm.
    """
    prefix = model.split("/", 1)[0] if "/" in model else ""

    if prefix in ("openai", "azure") or model.startswith("ft:"):
        from dspy.clients import _openai
        return _openai

    if prefix == "anthropic":
        from dspy.clients import _anthropic
        return _anthropic

    if prefix in ("google", "gemini"):
        from dspy.clients import _google
        return _google

    from dspy.clients import _litellm
    return _litellm


class LM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
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
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                         network error, rate limiting, etc. Requests are retried with exponential
                         backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
            rollout_id: Optional integer used to differentiate cache entries for otherwise
                identical requests. Different values bypass DSPy's caches while still caching
                future calls with the same inputs and rollout ID. Note that `rollout_id`
                only affects generation when `temperature` is non-zero. This argument is
                stripped before sending requests to the provider.
        """
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role
        self._warned_zero_temp_rollout = False

        # Handle model-specific configuration for different model families
        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        # Recognize OpenAI reasoning models (o1, o3, o4, gpt-5 family)
        # Exclude non-reasoning variants like gpt-5-chat this is in azure ai foundry
        # Allow date suffixes like -2023-01-01 after model name or mini/nano/pro
        # For gpt-5, use negative lookahead to exclude -chat and allow other suffixes
        model_pattern = re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
            model_family,
        )

        if model_pattern:
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise ValueError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                    "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)"
                )
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)

        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    # ------------------------------------------------------------------
    # Backend wiring
    # ------------------------------------------------------------------

    def _get_backend(self):
        """Lazily resolve and cache the backend module."""
        if not hasattr(self, "_backend"):
            self._backend = _resolve_backend(self.model, self.model_type)
        return self._backend

    supports_function_calling = _backend_capability("supports_function_calling")
    supports_reasoning = _backend_capability("supports_reasoning")
    supports_response_schema = _backend_capability("supports_response_schema")
    supported_params = _backend_capability("supported_params")

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _build_request(self, prompt, messages, kwargs):
        """Shared request-building logic for forward/aforward."""
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)
        request = dict(model=self.model, messages=messages, **kwargs)
        return request, cache

    def _apply_cache(self, fn, cache):
        """Wrap *fn* with DSPy's request cache when *cache* is enabled."""
        if cache:
            fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=["api_key", "api_base", "base_url"],
            )(fn)
        return fn

    def _post_process(self, results):
        """Shared post-processing for forward/aforward."""
        self._check_truncation(results)
        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    def forward(self, prompt=None, messages=None, **kwargs):
        request, cache = self._build_request(prompt, messages, kwargs)
        backend = self._get_backend()
        completion = self._apply_cache(backend.complete_request, cache)
        try:
            results = completion(request=request, model_type=self.model_type, num_retries=self.num_retries)
        except backend.ContextWindowError as e:
            raise ContextWindowExceededError(model=self.model) from e
        return self._post_process(results)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        request, cache = self._build_request(prompt, messages, kwargs)
        backend = self._get_backend()
        completion = self._apply_cache(backend.acomplete_request, cache)
        try:
            results = await completion(request=request, model_type=self.model_type, num_retries=self.num_retries)
        except backend.ContextWindowError as e:
            raise ContextWindowExceededError(model=self.model) from e
        return self._post_process(results)

    def launch(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.kill(self, launch_kwargs)

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> TrainingJob:
        from dspy import settings as settings

        if not self.provider.finetunable:
            raise ValueError(
                f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
                "setting `provider` when creating the `dspy.LM` instance. For example, "
                "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`."
            )

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

    def reinforce(self, train_kwargs) -> ReinforceJob:
        # TODO(GRPO Team): Should we return an initialized job here?
        from dspy import settings as settings

        err = f"Provider {self.provider} does not implement the reinforcement learning interface."
        assert self.provider.reinforceable, err

        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
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
            "num_retries",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
        ]
        # Exclude api_key from kwargs to prevent API keys from being saved in plain text
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != "api_key"}
        return {key: getattr(self, key) for key in state_keys} | filtered_kwargs

    def _check_truncation(self, results):
        choices = results.choices if hasattr(results, "choices") else results.get("choices", [])
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in choices):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )



# ---------------------------------------------------------------------------
# Backward-compatible re-exports for tests that import old names from here.
# ---------------------------------------------------------------------------

from dspy.clients._litellm import (  # noqa: F401, E402
    _convert_chat_request_to_responses_request,
    _convert_content_item_to_responses_format,
    _get_stream_completion_fn,
    acomplete as alitellm_completion,
    atext_complete as alitellm_text_completion,
    aresponses_complete as alitellm_responses_completion,
    complete as litellm_completion,
    text_complete as litellm_text_completion,
    responses_complete as litellm_responses_completion,
)
