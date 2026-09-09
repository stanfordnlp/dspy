import functools
import logging
import os
import re
import threading
import warnings
from typing import Any, Literal, cast

import anyio.from_thread
from anyio.streams.memory import MemoryObjectSendStream

import dspy
from dspy.clients._litellm import get_litellm
from dspy.clients.cache import request_cache
from dspy.clients.legacy_requests import chat_to_responses
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import LMConfigurationError, LMError, LMUnsupportedFeatureError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


def _get_litellm():
    return get_litellm(feature="dspy.LM")


def _is_openai_reasoning_model(model: str) -> bool:
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return re.match(
        r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
        model_family,
    ) is not None


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
        """Create a new language model instance for use with DSPy modules and programs.

        Args:
            model: The model to use. This should be a string of the form
                `"llm_provider/llm_name"` supported by LiteLLM. For example,
                `"openai/gpt-4o"`.
            model_type: The type of the model, such as `"chat"`, `"text"`, or
                `"responses"`.
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
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            callbacks=callbacks,
            **kwargs,
        )

        self.provider = provider or self.infer_provider()
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role

        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        # Override BaseLM's default kwargs shape for LiteLLM/model-family-specific token parameters.
        if _is_openai_reasoning_model(self.model):
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise LMConfigurationError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                    "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)",
                    model=self.model,
                    provider=self._provider_name,
                )
            initial_kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            initial_kwargs = super()._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)

        if initial_kwargs.get("rollout_id") is None:
            initial_kwargs.pop("rollout_id", None)
        return initial_kwargs

    @property
    def _provider_name(self) -> str:
        """Extract the provider name from the model string (e.g., 'openai' from 'openai/gpt-4o')."""
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"

    @property
    def supports_function_calling(self) -> bool:
        return _get_litellm().supports_function_calling(model=self.model)

    @property
    def supports_reasoning(self) -> bool:
        return _get_litellm().supports_reasoning(self.model)

    @property
    def supports_response_schema(self) -> bool:
        return _get_litellm().supports_response_schema(model=self.model, custom_llm_provider=self._provider_name)

    @property
    def supported_params(self) -> set[str]:
        params = _get_litellm().get_supported_openai_params(model=self.model, custom_llm_provider=self._provider_name)
        return set(params) if params else set()

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _get_cached_completion_fn(self, completion_fn, cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)

        litellm_cache_args = {"no-cache": True, "no-store": True}

        return completion_fn, litellm_cache_args

    def _wrap_litellm_exception(self, exc: Exception) -> LMError:
        from dspy.clients.engines.errors import wrap_error

        return wrap_error(exc, model=self.model, provider=self._provider_name)

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Call the configured LM synchronously.

        LiteLLM/provider exceptions are wrapped in DSPy's structured LM error
        hierarchy before they are re-raised.

        Args:
            prompt: Optional prompt text. Ignored when `messages` is provided.
            messages: Optional chat messages to send to the LM.
            **kwargs: Per-call LM parameters that override defaults from `LM(...)`.

        Raises:
            dspy.LMError: Base class for wrapped LM configuration, transport,
                provider, and unsupported-feature failures. Notable subclasses
                include `dspy.ContextWindowExceededError` for context-window
                failures, which adapters use to avoid inappropriate fallback
                retries when the prompt is too long.
        """
        # Build the request.
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        if self.model_type == "chat":
            completion = litellm_completion
        elif self.model_type == "text":
            completion = litellm_text_completion
        elif self.model_type == "responses":
            completion = litellm_responses_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

        try:
            results = completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except Exception as e:
            if isinstance(e, LMError):
                raise
            raise self._wrap_litellm_exception(e) from e

        self._check_truncation(results)

        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Call the configured LM asynchronously.

        LiteLLM/provider exceptions are wrapped in DSPy's structured LM error
        hierarchy before they are re-raised.

        Args:
            prompt: Optional prompt text. Ignored when `messages` is provided.
            messages: Optional chat messages to send to the LM.
            **kwargs: Per-call LM parameters that override defaults from `LM(...)`.

        Raises:
            dspy.LMError: Base class for wrapped LM configuration, transport,
                provider, and unsupported-feature failures. Notable subclasses
                include `dspy.ContextWindowExceededError` for context-window
                failures, which adapters use to avoid inappropriate fallback
                retries when the prompt is too long.
        """
        # Build the request.
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        if self.model_type == "chat":
            completion = alitellm_completion
        elif self.model_type == "text":
            completion = alitellm_text_completion
        elif self.model_type == "responses":
            completion = alitellm_responses_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

        try:
            results = await completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except Exception as e:
            if isinstance(e, LMError):
                raise
            raise self._wrap_litellm_exception(e) from e

        self._check_truncation(results)

        return results

    def _typed_completion(self, request, controls, *, asynchronous):
        from dspy.clients.lm15_boundary import request_kwargs

        if self.model_type == "text":
            raise LMUnsupportedFeatureError(
                "Explicit lm15.Request calls require model_type='chat' or 'responses'. "
                "Text-completion models remain available through ordinary prompt/messages calls.",
                model=self.model,
            )
        data = request_kwargs(request, self.model_type)
        # Only client configuration is inherited. Generation parameters are
        # entirely owned by the explicit Request, including absent values.
        client_keys = {
            "api_key", "api_base", "base_url", "headers", "extra_headers", "timeout",
            "azure_ad_token_provider", "api_version", "organization", "project",
        }
        data = {**{key: val for key, val in self.kwargs.items() if key in client_keys}, **data}
        data["model"] = self.model
        if controls.get("rollout_id") is not None:
            data["rollout_id"] = controls["rollout_id"]
        self._warn_zero_temp_rollout(request.config.temperature, controls.get("rollout_id"))
        if self.model_type == "chat":
            data["n"] = 1
            fn = alitellm_completion if asynchronous else litellm_completion
        else:
            fn = alitellm_responses_completion if asynchronous else litellm_responses_completion
        fn, cache_args = self._get_cached_completion_fn(fn, controls.get("cache", self.cache))
        return fn, {"request": data, "num_retries": self.num_retries, "cache": cache_args}

    def _forward_request(self, request, **controls):
        try:
            fn, kwargs = self._typed_completion(request, controls, asynchronous=False)
            response = fn(**kwargs)
            self._check_typed_truncation(response, request)
            return response
        except LMError:
            raise
        except Exception as exc:
            raise self._wrap_litellm_exception(exc) from exc

    async def _aforward_request(self, request, **controls):
        try:
            fn, kwargs = self._typed_completion(request, controls, asynchronous=True)
            response = await fn(**kwargs)
            self._check_typed_truncation(response, request)
            return response
        except LMError:
            raise
        except Exception as exc:
            raise self._wrap_litellm_exception(exc) from exc

    def _check_typed_truncation(self, response, request):
        from dspy.clients.legacy_outputs import value

        truncated = any(value(choice, "finish_reason") == "length" for choice in value(response, "choices", []) or [])
        if self.model_type == "responses":
            truncated = value(value(response, "incomplete_details", {}) or {}, "reason") == "max_output_tokens"
        if truncated:
            logger.warning(
                "LM response was truncated at the request's token limit (%s). "
                "Inspect history or increase Request.config.max_tokens.", request.config.max_tokens,
            )

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
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
                "setting `provider` when creating the `dspy.LM` instance. For example, "
                "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`.",
                model=self.model,
                provider=self._provider_name,
                features=["finetuning"],
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

        if not self.provider.reinforceable:
            raise LMUnsupportedFeatureError(
                f"Provider {self.provider} does not implement the reinforcement learning interface.",
                model=self.model,
                provider=self._provider_name,
                features=["reinforce"],
            )

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
        """Return a sanitized reconstruction state for this LM.

        Returns:
            A dictionary that can be passed to `BaseLM.load_state` to
            reconstruct this `LM`. The state excludes API keys.
        """
        state = super().dump_state()
        state.update(
            {
                "finetuning_model": self.finetuning_model,
                "launch_kwargs": self.launch_kwargs,
                "train_kwargs": self.train_kwargs,
            }
        )
        if self.use_developer_role:
            state["use_developer_role"] = self.use_developer_role
        if _is_openai_reasoning_model(self.model) and "max_completion_tokens" in state:
            state["max_tokens"] = state.pop("max_completion_tokens")
        return state

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False):
        state = dict(state)

        model = state.get("model")
        if isinstance(model, str) and _is_openai_reasoning_model(model) and "max_completion_tokens" in state:
            if "max_tokens" not in state:
                state["max_tokens"] = state["max_completion_tokens"]
            state.pop("max_completion_tokens")

        return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)

    def _check_truncation(self, results):
        if self.model_type != "responses" and any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )


def _get_stream_completion_fn(
    request: dict[str, Any],
    cache_kwargs: dict[str, Any],
    sync=True,
    headers: dict[str, Any] | None = None,
):
    stream = dspy.settings.send_stream
    caller_predict = dspy.settings.caller_predict

    if stream is None:
        return None

    # The stream is already opened, and will be closed by the caller.
    stream = cast(MemoryObjectSendStream, stream)
    caller_predict_id = id(caller_predict) if caller_predict else None

    if dspy.settings.track_usage:
        request["stream_options"] = {"include_usage": True}

    async def stream_completion(request: dict[str, Any], cache_kwargs: dict[str, Any]):
        response = await _get_litellm().acompletion(
            cache=cache_kwargs,
            stream=True,
            headers=headers,
            **request,
        )
        chunks = []
        async for chunk in response:
            if caller_predict_id:
                # Add the predict id to the chunk so that the stream listener can identify which predict produces it.
                chunk.predict_id = caller_predict_id
            chunks.append(chunk)
            await stream.send(chunk)
        return _get_litellm().stream_chunk_builder(chunks)

    def sync_stream_completion():
        return anyio.from_thread.run(functools.partial(stream_completion, request, cache_kwargs))

    async def async_stream_completion():
        return await stream_completion(request, cache_kwargs)

    if sync:
        return sync_stream_completion
    else:
        return async_stream_completion


def litellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    stream_completion = _get_stream_completion_fn(request, cache, sync=True, headers=headers)
    if stream_completion is None:
        return _get_litellm().completion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=headers,
            **request,
        )

    return stream_completion()


def litellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return _get_litellm().text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    stream_completion = _get_stream_completion_fn(request, cache, sync=False, headers=headers)
    if stream_completion is None:
        return await _get_litellm().acompletion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=headers,
            **request,
        )

    return await stream_completion()


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    model = request.pop("model").split("/", 1)
    headers = request.pop("headers", None)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return await _get_litellm().atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def litellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return _get_litellm().responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return await _get_litellm().aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def _convert_chat_request_to_responses_request(request: dict[str, Any]):
    """Translate ordinary calls without the retired experimental types."""
    if "input" in request:
        data = dict(request)
        data.pop("messages", None)
        return data
    return chat_to_responses(request)


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }

