import logging
import re
import threading
import warnings
from typing import Any, Literal

from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.lm15 import CacheConfig
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import LMConfigurationError, LMUnsupportedFeatureError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)

ENGINE_SELECTIONS = ("auto", "lm15", "litellm")


def _is_openai_reasoning_model(model: str) -> bool:
    model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return re.match(
        r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
        model_family,
    ) is not None


class LM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.

    `lm(request)` runs one `dspy.lm15.Request` and returns a `dspy.lm15.Response`;
    `lm.generate(request, n=...)` returns several. `lm("hello")` is a convenience
    that renders one user message with this LM's generation defaults and returns
    a list of outputs. Custom backends are engines passed with `engine=`; see
    https://dspy.ai/community/normalized-lm-api-migration/.
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
        engine: Any = "auto",
        async_engine: Any = None,
        prompt_cache: CacheConfig | None = None,
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
            engine: 'auto' prefers lm15 for representable requests; 'litellm' preserves the compatibility backend;
                'lm15' refuses unsupported mappings rather than selecting LiteLLM. A custom engine implements
                complete(Request) -> Response and optionally stream(Request). Engines are borrowed.
            async_engine: Async counterpart when supplying a custom engine object.
            prompt_cache: Optional lm15 CacheConfig for provider-side prompt caching on ordinary calls.
                Separate from DSPy's response cache. Requires native lm15 or a canonical custom engine;
                may incur cache-write/storage charges. A call-time value overrides this default, and None
                removes it. Explicit Request calls use only Request.config.cache. No cache resource is created.
            provider: The training/launch provider. This does not select the inference engine.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
            rollout_id: Optional integer used to differentiate cache entries for otherwise
                identical requests. Different values bypass DSPy's caches while still caching
                future calls with the same inputs and rollout ID. Note that `rollout_id`
                only affects generation when `temperature` is non-zero. This argument is
                stripped before sending requests to the provider.
        """
        if isinstance(engine, str):
            if engine not in ENGINE_SELECTIONS:
                raise ValueError("engine must be 'auto', 'lm15', 'litellm', or an engine object")
            if async_engine is not None:
                raise ValueError("async_engine is only used with a custom engine object")
        elif not callable(getattr(engine, "complete", None)):
            raise TypeError("A custom engine must implement complete(Request) -> Response")
        if isinstance(num_retries, bool) or not isinstance(num_retries, int) or num_retries < 0:
            raise ValueError("num_retries must be a nonnegative integer")
        if prompt_cache is not None:
            if not isinstance(prompt_cache, CacheConfig):
                raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
            kwargs["prompt_cache"] = prompt_cache
        self._engine_store = {}
        self._engine_lock = threading.RLock()
        self._warned_zero_temp_rollout = False
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            callbacks=callbacks,
            engine=None if isinstance(engine, str) else engine,
            async_engine=async_engine,
            **kwargs,
        )
        if isinstance(engine, str):
            self._engine_spec = engine

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
        from dspy.clients.capabilities import capabilities

        return capabilities(self).function_calling

    @property
    def supports_reasoning(self) -> bool:
        from dspy.clients.capabilities import capabilities

        return capabilities(self).reasoning

    @property
    def supports_response_schema(self) -> bool:
        from dspy.clients.capabilities import capabilities

        return capabilities(self).response_schema

    @property
    def supported_params(self) -> set[str]:
        from dspy.clients.capabilities import capabilities

        return set(capabilities(self).params)

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def close(self):
        """Close owned synchronous engine pools after active calls have finished.

        Custom engines are borrowed. Async pools must be closed with aclose().
        Copies share owned pools; closing one releases those shared resources.
        """
        with self._engine_lock:
            keys = [key for key in self._engine_store if key[0] is None]
            engines = [self._engine_store.pop(key) for key in keys]
        from contextlib import ExitStack

        with ExitStack() as cleanup:
            for engine in engines:
                cleanup.callback(engine.close)

    async def aclose(self):
        """Close this event loop's owned pools and the synchronous pools."""
        import asyncio
        from contextlib import AsyncExitStack

        loop = asyncio.get_running_loop()
        with self._engine_lock:
            keys = [key for key in self._engine_store if key[0] is loop]
            engines = [self._engine_store.pop(key) for key in keys]
        try:
            async with AsyncExitStack() as cleanup:
                for engine in engines:
                    cleanup.push_async_callback(engine.aclose)
        finally:
            self.close()

    def __copy__(self):
        import types

        copied = object.__new__(type(self))
        copied.__dict__.update(self.__dict__)
        for cls in type(self).__mro__:
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, types.MemberDescriptorType) and hasattr(self, name):
                    setattr(copied, name, getattr(self, name))
        return copied

    def copy(self, **kwargs):
        if kwargs.get("prompt_cache") is not None and not isinstance(kwargs["prompt_cache"], CacheConfig):
            raise TypeError("prompt_cache must be a dspy.lm15.CacheConfig or None")
        spec = kwargs.pop("engine", self._engine_spec)
        async_spec = kwargs.pop("async_engine", self._async_engine_spec)
        if isinstance(spec, str) and spec not in ENGINE_SELECTIONS:
            raise ValueError("Unknown engine selection")
        if not isinstance(spec, str) and not callable(getattr(spec, "complete", None)):
            raise TypeError("Custom engines must implement complete(Request)")
        copied = super().copy(**kwargs)
        copied._engine_spec = spec
        copied._async_engine_spec = async_spec
        return copied

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_engine_lock", None)
        state.pop("_engine_store", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._engine_spec = getattr(self, "_engine_spec", "auto")
        self._async_engine_spec = getattr(self, "_async_engine_spec", None)
        self._engine_lock = threading.RLock()
        self._engine_store = {}

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
        if not isinstance(self._engine_spec, str):
            raise TypeError("Custom engine objects require custom dump_state/load_state methods; they cannot be stored in JSON LM state.")
        state = super().dump_state()
        if state.get("prompt_cache") is not None:
            from dspy._vendor.lm15.serde import cache_config_to_dict

            state["prompt_cache"] = cache_config_to_dict(state["prompt_cache"])
        if self._engine_spec != "auto":
            state["engine"] = self._engine_spec
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
        if isinstance(state.get("prompt_cache"), dict):
            from dspy._vendor.lm15.serde import cache_config_from_dict

            state["prompt_cache"] = cache_config_from_dict(state["prompt_cache"])

        model = state.get("model")
        if isinstance(model, str) and _is_openai_reasoning_model(model) and "max_completion_tokens" in state:
            if "max_tokens" not in state:
                state["max_tokens"] = state["max_completion_tokens"]
            state.pop("max_completion_tokens")

        return super().load_state(state, allow_custom_lm_class=allow_custom_lm_class)
