import logging
import os
import re
import threading
import warnings
from typing import Any, Literal, cast

import anyio.from_thread
import litellm
import pydantic
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify

import dspy
from dspy.clients._litellm import normalize_litellm_error
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.openai_format import to_openai_responses_request
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.core.types import LMRequest
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


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

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and (temperature is None or temperature == 0):
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

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
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
        except Exception as err:
            raise normalize_litellm_error(err, model=self.model) from err

        self._check_truncation(results)

        if not getattr(results, "cache_hit", False):
            usage = getattr(results, "usage", None)
            if dspy.settings.usage_tracker and usage is not None:
                settings.usage_tracker.add_usage(self.model, dict(usage))
            if dspy.settings.cost_tracker:
                cost = getattr(results, "_hidden_params", {}).get("response_cost")
                usage_dict = dict(usage) if usage is not None else None
                settings.cost_tracker.add_cost(self.model, cost, usage_dict)
        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
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
        except Exception as err:
            raise normalize_litellm_error(err, model=self.model) from err

        self._check_truncation(results)

        if not getattr(results, "cache_hit", False):
            usage = getattr(results, "usage", None)
            if dspy.settings.usage_tracker and usage is not None:
                settings.usage_tracker.add_usage(self.model, dict(usage))
            if dspy.settings.cost_tracker:
                cost = getattr(results, "_hidden_params", {}).get("response_cost")
                usage_dict = dict(usage) if usage is not None else None
                settings.cost_tracker.add_cost(self.model, cost, usage_dict)
        return results

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
        """Return a sanitized reconstruction state for this LM.

        Returns:
            A dictionary that can be passed to `BaseLM.load_state` to
            reconstruct this `LM`. The state excludes API keys and callbacks
            (callbacks are runtime objects and not JSON-serializable).
        """
        state = super().dump_state()
        state.update(
            {
                "finetuning_model": self.finetuning_model,
                "launch_kwargs": self.launch_kwargs,
                "train_kwargs": self.train_kwargs,
            }
        )
        # Persist developer-role flag; omit callbacks (not serializable).
        if self.use_developer_role:
            state["use_developer_role"] = self.use_developer_role
        return state

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
        response = await litellm.acompletion(
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
        return litellm.stream_chunk_builder(chunks)

    def sync_stream_completion():
        syncified_stream_completion = syncify(stream_completion)
        return syncified_stream_completion(request, cache_kwargs)

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
        return litellm.completion(
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

    return litellm.text_completion(
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
        return await litellm.acompletion(
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

    return await litellm.atext_completion(
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

    return litellm.responses(
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

    return await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def _convert_chat_request_to_responses_request(request: dict[str, Any]):
    """Convert legacy chat-shaped LM kwargs into Responses API kwargs.

    This is the legacy door into the normalized Responses mapper. Inputs the
    old pass-through converter forwarded verbatim — Responses-native shapes and
    provider-SDK dumps — are tolerated here and only here; the typed path stays
    strict.
    """
    request = dict(request)
    model = request.pop("model", None)
    raw_messages = request.pop("messages", None)
    tools = list(request.pop("tools", None) or [])

    if raw_messages:
        messages = [_sanitize_legacy_message(m) if isinstance(m, dict) else m for m in raw_messages]
        content_blocks = []
        for msg in messages:
            c = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                for item in c:
                    content_blocks.append(_convert_content_item_to_responses_format(item))
        request["input"] = [{"role": msg.get("role", "user") if isinstance(msg, dict) else "user", "content": content_blocks}]

    if "max_completion_tokens" in request and "max_tokens" not in request:
        request["max_tokens"] = request.pop("max_completion_tokens")
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        if request.get("reasoning") is None:
            request["reasoning"] = {"effort": effort, "summary": "auto"}

    # Hosted Responses tools (web_search, file_search, code_interpreter, mcp, ...)
    # have no normalized LMToolSpec representation; send them through unchanged.
    function_tools = [tool for tool in tools if not _is_hosted_responses_tool(tool)]

    # tool_choice dicts that aren't the chat-nested function form are already
    # Responses-native (flat function, hosted, allowed_tools, ...); send them
    # through unchanged.
    responses_native_tool_choice = None
    if isinstance(request.get("tool_choice"), dict) and "function" not in request["tool_choice"]:
        responses_native_tool_choice = request.pop("tool_choice")

    lm_request = LMRequest.from_call(model=model, messages=messages, tools=function_tools or None, **request)
    # The old converter never validated reasoning/temperature combinations
    # client-side; keep the provider as the authority on this door.
    data = to_openai_responses_request(lm_request, enforce_reasoning_temperature=False)
    if tools:
        normalized_function_tools = iter(data.pop("tools", []))
        data["tools"] = [
            tool if _is_hosted_responses_tool(tool) else next(normalized_function_tools) for tool in tools
        ]
    if responses_native_tool_choice is not None:
        data["tool_choice"] = responses_native_tool_choice
    return data


def _is_hosted_responses_tool(tool: Any) -> bool:
    return isinstance(tool, dict) and "function" not in tool and tool.get("type") not in (None, "function")


def _sanitize_legacy_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one legacy message the old converter would have passed through.

    Strips provider-SDK output fields that are not message inputs and rewrites
    Responses-native content blocks into their chat forms so the mapper can
    re-emit them with role-correct direction.
    """
    if not isinstance(message, dict):
        return message
    message_keys = {"role", "content", "name", "metadata", "tool_calls", "tool_call_id"}
    cleaned = {key: value for key, value in message.items() if key in message_keys}
    content = cleaned.get("content")
    if isinstance(content, list):
        cleaned["content"] = [_normalize_legacy_content_block(block) for block in content]
    return cleaned


def _normalize_legacy_content_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    block_type = block.get("type")
    if block_type in ("input_text", "output_text"):
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "input_image" and isinstance(block.get("image_url"), str):
        return {"type": "image_url", "image_url": {"url": block["image_url"]}}
    if block_type == "input_file":
        file = {key: block[key] for key in ("file_data", "file_id", "filename") if block.get(key) is not None}
        return {"type": "file", "file": file}
    return block


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }
