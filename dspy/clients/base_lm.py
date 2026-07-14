import copy as copy_module
import datetime
import importlib
import inspect
import uuid
import warnings
from typing import Any, Literal, TextIO

from dspy.clients.openai_format import (
    completion_to_lm_response,
    cost_from_response,
    lm_response_from_legacy_outputs,
    responses_to_lm_response,
    to_openai_chat_request,
    usage_from_response,
)
from dspy.core.types import LMHistoryEntry, LMRequest, LMResponse
from dspy.dsp.utils import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []
LM_CLASS_STATE_KEY = "_dspy_lm_class"
_BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"
ForwardContract = Literal["legacy", "typed_lm"]


def _import_lm_class(class_path: str) -> type:
    parts = class_path.split(".")
    last_error = None

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise

        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, type):
            raise TypeError(f"Serialized LM class `{class_path}` did not resolve to a class.")
        return obj

    raise ImportError(f"Could not import serialized LM class `{class_path}`.") from last_error


class BaseLM:
    """Base class for DSPy language models.

    Most users should use `dspy.LM`, which is a `BaseLM` subclass.

    For advanced use cases, such as custom language model backends, users can
    subclass `BaseLM` and implement `forward()`.

    DSPy is migrating `forward()` from the legacy OpenAI/LiteLLM-shaped
    contract to a typed DSPy contract. During this migration, subclasses should
    declare which contract they implement with `forward_contract`:

    - `forward_contract = "typed_lm"`: implement
      `forward(request: dspy.LMRequest) -> dspy.LMResponse`. This is the
      preferred contract for new custom LMs.
    - `forward_contract = "legacy"`: implement
      `forward(prompt=None, messages=None, **kwargs)` and return an OpenAI-like
      provider response. This remains the default during the migration.

    `BaseLM.__call__()` is the compatibility boundary. In DSPy 3.3 and 3.4,
    ordinary calls preserve the legacy public return value, `list[str | dict]`:

    ```python
    outputs = lm("What is DSPy?")
    outputs = lm(messages=[{"role": "user", "content": "What is DSPy?"}])
    ```

    Calls can flow internally through the typed `LMRequest` / `LMResponse` path
    without changing the public return shape. The typed path is used when the
    caller passes an explicit `dspy.LMRequest`, when
    `dspy.context(experimental=True)` is active, or when the subclass declares
    `forward_contract = "typed_lm"`. It accepts richer direct-call inputs,
    including `dspy.System`, `dspy.User`, `dspy.Assistant`, `dspy.ToolResult`,
    content parts, and prior `dspy.LMResponse` objects. The public return value
    remains legacy outputs unless the caller explicitly opts into typed output
    with an `LMRequest` or `experimental=True`.

    Example typed direct call:

    ```python
    with dspy.context(experimental=True):
        response = lm(
            dspy.System("You are concise."),
            dspy.User("What is DSPy?"),
        )
        print(response.text)
    ```

    `LMResponse` is designed to feel familiar to users of the legacy output
    list while carrying substantially more structure, including typed outputs,
    usage, cache status, provider metadata, tool calls, reasoning, citations,
    and multimodal content.

    LMs must be serializable as part of saved DSPy programs. The default
    `dump_state()` and `load_state()` implementations support subclasses whose
    persistent state is fully captured by `BaseLM.__init__()` arguments. If a
    subclass stores additional persistent state, override both methods.

    Examples:
        Preferred typed custom LM:

        ```python
        import dspy


        class EchoLM(dspy.BaseLM):
            forward_contract = "typed_lm"

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                return dspy.LMResponse.from_text("hello", model=request.model)


        lm = EchoLM(model="test/echo")

        with dspy.context(experimental=True):
            response = lm(dspy.User("Say hello."))
            print(response.text)
        ```

        Legacy custom LM for an OpenAI-like provider:

        ```python
        from openai import OpenAI

        import dspy


        class MyLegacyLM(dspy.BaseLM):
            forward_contract = "legacy"

            def forward(self, prompt=None, messages=None, **kwargs):
                client = OpenAI()
                return client.chat.completions.create(
                    model=self.model,
                    messages=messages or [{"role": "user", "content": prompt}],
                    **self.kwargs,
                    **kwargs,
                )


        lm = MyLegacyLM(model="gpt-4o-mini")
        dspy.configure(lm=lm)
        print(dspy.Predict("q -> a")(q="Why did the chicken cross the kitchen?"))
        ```
    """

    forward_contract: ForwardContract = "legacy"
    """The `forward()` implementation contract used by this LM.

    `"legacy"` means `forward(prompt=None, messages=None, **kwargs)` returns an
    OpenAI-like provider response. `"typed_lm"` means
    `forward(request: dspy.LMRequest) -> dspy.LMResponse`.
    """

    def __init__(
        self,
        model,
        model_type="chat",
        temperature=None,
        max_tokens=None,
        cache=True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        **kwargs,
    ):
        """Initialize a base language model.

        Args:
            model: The model identifier.
            model_type: The LM API type, such as `"chat"`, `"text"`, or
                `"responses"`.
            temperature: The default sampling temperature.
            max_tokens: The default maximum number of output tokens.
            cache: Whether requests should use DSPy's cache by default.
            num_retries: The default number of provider request retries.
            callbacks: Optional instance-level callback handlers.
            **kwargs: Additional default request parameters stored in
                `self.kwargs`.
        """
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = list(callbacks or [])
        self.num_retries = num_retries
        self.kwargs = self._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self._warned_zero_temp_rollout = False

    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        return dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _declares_forward_contract(self) -> bool:
        """Return whether this concrete LM class declares `forward_contract`."""
        return "forward_contract" in type(self).__dict__

    def _get_forward_contract(self) -> ForwardContract:
        """Return the declared `forward()` contract for this LM.

        DSPy deliberately does not inspect `forward()` signatures during the
        migration to typed LM calls. Subclasses opt into the typed contract by
        setting `forward_contract = "typed_lm"`; all existing custom LMs remain
        on the legacy contract unless they say otherwise.
        """
        contract = getattr(type(self), "forward_contract", "legacy")
        if contract not in {"legacy", "typed_lm"}:
            raise ValueError(
                f"{type(self).__name__}.forward_contract must be 'legacy' or 'typed_lm', "
                f"but got {contract!r}."
            )
        return contract

    def _validate_typed_lm_response(self, response: Any) -> LMResponse:
        """Validate the result of a typed `forward(request)` implementation."""
        if isinstance(response, LMResponse):
            return response
        raise TypeError(
            f"{type(self).__name__}.forward_contract='typed_lm' requires forward(request) "
            f"to return dspy.LMResponse, but got {type(response).__name__}."
        )

    def _validate_legacy_lm_response(
        self,
        response: Any,
        *,
        stacklevel: int = 2,
    ) -> LMResponse | None:
        """Validate a legacy `forward()` result that already looks typed.

        During the 3.3 migration, custom LMs that inherit the default legacy
        contract may accidentally return `LMResponse`; warn so authors can add
        `forward_contract = "typed_lm"`. If a class explicitly declares
        `forward_contract = "legacy"`, treat `LMResponse` as a contract
        mismatch and fail fast.
        """
        if not isinstance(response, LMResponse):
            return None
        if self._declares_forward_contract():
            raise TypeError(
                f"{type(self).__name__}.forward_contract='legacy' requires forward() to return an "
                "OpenAI-like provider response, but got dspy.LMResponse. Set forward_contract='typed_lm'."
            )
        warnings.warn(
            f"{type(self).__name__}.forward() returned dspy.LMResponse while using the default legacy "
            "forward_contract. Set forward_contract='typed_lm' before the typed LM API becomes the default.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        return response

    @property
    def supports_function_calling(self) -> bool:
        """Whether the model supports function calling (tool use)."""
        return False

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports native reasoning (extended thinking)."""
        return False

    @property
    def supports_response_schema(self) -> bool:
        """Whether the model supports structured output via response schema."""
        return False

    @property
    def supported_params(self) -> set[str]:
        """Set of supported OpenAI-style parameter names for the model."""
        return set()

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        if not getattr(response, "cache_hit", False) and settings.usage_tracker:
            settings.usage_tracker.add_usage(self.model, dict(getattr(response, "usage", {}) or {}))

        if settings.disable_history:
            return outputs

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(getattr(response, "usage", {}) or {}),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }

        self.update_history(entry)

        return outputs

    @with_callbacks
    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ) -> LMResponse | list[dict[str, Any] | str]:
        """Call the language model synchronously.

        The default call path preserves DSPy's legacy behavior and returns `list[str | dict]`. Calls flow internally
        through `LMRequest` / `LMResponse` when either:

        - `request=` is provided or the first positional argument is an `LMRequest`, or
        - `dspy.context(experimental=True)` is active, or
        - the subclass declares `forward_contract = "typed_lm"`.

        In the typed request path, positional `items` are normalized with `LMRequest.from_call()`. This supports
        a single prompt string, direct message objects such as `dspy.System(...)`, `dspy.User(...)`, and
        `dspy.Assistant(...)`, `dspy.ToolResult(...)` tool messages, and prior `LMResponse` values as assistant turns.

        Args:
            *items: Optional direct-call inputs. In the legacy path this may contain at most one prompt string. In the
                typed path it may contain normalized messages, message sequences, prior `LMResponse` values, or content
                parts accepted by `LMRequest.from_call()`.
            prompt: Optional prompt string. Do not combine with positional prompt input.
            messages: Optional OpenAI-chat-shaped messages. Do not combine with `items` or `prompt` in the typed path.
            request: Optional explicit normalized request. Call kwargs override request config when provided.
            **kwargs: Per-call generation parameters.

        Returns:
            `LMResponse` for explicit `LMRequest` calls or `experimental=True`; otherwise DSPy's legacy list of output
            strings or dictionaries, even when a typed LM subclass uses the typed path internally.
        """
        return_typed_response, forward_contract, normalized_request = self._prepare_lm_call(
            items=items,
            prompt=prompt,
            messages=messages,
            request=request,
            kwargs=kwargs,
        )
        if normalized_request is None:
            return self._legacy_call_direct(*items, prompt=prompt, messages=messages, **kwargs)

        if forward_contract == "typed_lm":
            response = self.forward(normalized_request)
            response = self._finalize_lm_response(normalized_request, self._validate_typed_lm_response(response))
        else:
            response = self._legacy_forward_as_lm_response(normalized_request)
        if return_typed_response:
            return response
        return response.to_legacy_outputs()

    @with_callbacks
    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ) -> LMResponse | list[dict[str, Any] | str]:
        """Asynchronously call the language model.

        This is the async equivalent of `__call__()`. It preserves legacy outputs by default and returns
        `dspy.LMResponse` for explicit `LMRequest` calls or experimental direct calls.
        """
        return_typed_response, forward_contract, normalized_request = self._prepare_lm_call(
            items=items,
            prompt=prompt,
            messages=messages,
            request=request,
            kwargs=kwargs,
        )
        if normalized_request is None:
            return await self._legacy_acall_direct(*items, prompt=prompt, messages=messages, **kwargs)

        if forward_contract == "typed_lm":
            response = await self.aforward(normalized_request)
            response = self._finalize_lm_response(normalized_request, self._validate_typed_lm_response(response))
        else:
            response = await self._legacy_aforward_as_lm_response(normalized_request)
        if return_typed_response:
            return response
        return response.to_legacy_outputs()

    def _prepare_lm_call(
        self,
        *,
        items: tuple[Any, ...],
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        request: LMRequest | None,
        kwargs: dict[str, Any],
    ) -> tuple[bool, ForwardContract, LMRequest | None]:
        explicit_request = request is not None or bool(items and isinstance(items[0], LMRequest))
        return_typed_response = explicit_request or bool(settings.get("experimental", False))
        forward_contract = self._get_forward_contract()
        if not return_typed_response and forward_contract != "typed_lm":
            return return_typed_response, forward_contract, None

        normalized_request = self._normalize_lm_call(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        return return_typed_response, forward_contract, normalized_request

    def _legacy_call_direct(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any] | str]:
        """Execute the pre-typed synchronous call path and return legacy outputs."""
        prompt = self._legacy_prompt_from_items(items, prompt=prompt)
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        if isinstance(response, LMResponse):
            raise TypeError(
                f"{type(self).__name__}.forward() returned dspy.LMResponse on the legacy direct path. "
                "Set forward_contract='typed_lm' or pass an LMRequest/use dspy.context(experimental=True)."
            )
        return self._process_lm_response(response, prompt, messages, **kwargs)

    async def _legacy_acall_direct(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any] | str]:
        """Execute the pre-typed asynchronous call path and return legacy outputs."""
        prompt = self._legacy_prompt_from_items(items, prompt=prompt)
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        if isinstance(response, LMResponse):
            raise TypeError(
                f"{type(self).__name__}.aforward() returned dspy.LMResponse on the legacy direct path. "
                "Set forward_contract='typed_lm' or pass an LMRequest/use dspy.context(experimental=True)."
            )
        return self._process_lm_response(response, prompt, messages, **kwargs)

    def _legacy_prompt_from_items(self, items: tuple[Any, ...], *, prompt: str | None) -> str | None:
        """Validate and extract the one positional prompt accepted by legacy calls."""
        if len(items) > 1:
            raise TypeError(
                "Legacy BaseLM calls accept at most one positional prompt. "
                "Use dspy.context(experimental=True) or pass an LMRequest for typed multi-item LM calls."
            )
        if items and prompt is not None:
            raise TypeError("Pass a prompt either positionally or by keyword, not both.")
        if items and isinstance(items[0], LMRequest):
            raise TypeError("LMRequest calls require the typed LM path; this should be unreachable.")
        return items[0] if items else prompt

    def _normalize_lm_call(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMRequest:
        """Normalize direct call inputs or an explicit request into `LMRequest`."""
        if request is None and items and isinstance(items[0], LMRequest):
            request = items[0]
            items = items[1:]

        if request is not None:
            if items or prompt is not None or messages is not None:
                raise ValueError(
                    "Pass either an LMRequest or direct-call inputs, not both. "
                    "Use call kwargs to override request config."
                )
            return request.with_config_overrides(**kwargs) if kwargs else request

        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs.setdefault("cache", self.cache)
        return LMRequest.from_call(
            model=self.model,
            items=items,
            prompt=prompt,
            messages=messages,
            **merged_kwargs,
        )

    def _legacy_forward_as_lm_response(self, request: LMRequest) -> LMResponse:
        """Call a legacy `forward()` implementation and normalize its provider response."""
        data = self._legacy_forward_kwargs(request)
        messages = data.pop("messages", None)
        prompt = self._prompt_from_lm_request(request)
        if prompt is not None:
            messages = None
        response = self.forward(prompt=prompt, messages=messages, **data)
        typed_response = self._validate_legacy_lm_response(response, stacklevel=4)
        if typed_response is not None:
            return self._finalize_lm_response(request, typed_response)
        with settings.context(disable_history=True, usage_tracker=None):
            outputs = self._process_lm_response(response, prompt, messages, **data)
        lm_response = self._legacy_outputs_to_lm_response(outputs, request=request, provider_response=response)
        return self._finalize_lm_response(request, lm_response)

    async def _legacy_aforward_as_lm_response(self, request: LMRequest) -> LMResponse:
        """Async variant of `_legacy_forward_as_lm_response()`."""
        data = self._legacy_forward_kwargs(request)
        messages = data.pop("messages", None)
        prompt = self._prompt_from_lm_request(request)
        if prompt is not None:
            messages = None
        response = await self.aforward(prompt=prompt, messages=messages, **data)
        typed_response = self._validate_legacy_lm_response(response, stacklevel=4)
        if typed_response is not None:
            return self._finalize_lm_response(request, typed_response)
        with settings.context(disable_history=True, usage_tracker=None):
            outputs = self._process_lm_response(response, prompt, messages, **data)
        lm_response = self._legacy_outputs_to_lm_response(outputs, request=request, provider_response=response)
        return self._finalize_lm_response(request, lm_response)

    def _legacy_outputs_to_lm_response(
        self,
        outputs: list[dict[str, Any] | str | None],
        *,
        request: LMRequest,
        provider_response: Any,
    ) -> LMResponse:
        """Convert legacy post-processed outputs and provider metadata into `LMResponse`."""
        if self.model_type == "responses":
            response = responses_to_lm_response(provider_response, request)
        elif self.model_type in {"chat", "text"}:
            response = completion_to_lm_response(provider_response, request)
        else:
            response = lm_response_from_legacy_outputs(outputs, request)

        return response.model_copy(
            update={
                "model": getattr(provider_response, "model", None) or response.model,
                "usage": usage_from_response(provider_response),
                "cost": cost_from_response(provider_response),
                "cache_hit": bool(getattr(provider_response, "cache_hit", False)),
                "provider_response": provider_response,
            }
        )

    def _legacy_forward_kwargs(self, request: LMRequest) -> dict[str, Any]:
        """Convert a normalized request into kwargs for legacy `forward()` implementations."""
        data = to_openai_chat_request(request)
        data.pop("model", None)
        if request.config.cache is not None:
            if request.config.cache.enabled is not None:
                data["cache"] = request.config.cache.enabled
            if request.config.cache.rollout_id is not None:
                data["rollout_id"] = request.config.cache.rollout_id
        return data

    def _prompt_from_lm_request(self, request: LMRequest) -> str | None:
        """Return the legacy prompt when a normalized request is exactly one text user message."""
        if len(request.messages) != 1:
            return None
        message = request.messages[0]
        if message.role != "user" or len(message.parts) != 1:
            return None
        part = message.parts[0]
        return part.text if getattr(part, "type", None) == "text" else None

    def _finalize_lm_response(self, request: LMRequest, response: LMResponse) -> LMResponse:
        """Record usage and typed history for a normalized LM response."""
        if not getattr(response, "cache_hit", False) and settings.usage_tracker:
            usage = response.usage_as_dict()
            if usage:
                settings.usage_tracker.add_usage(self.model, usage)

        if not settings.disable_history:
            entry = LMHistoryEntry(
                request=request,
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)
        return response

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Forward pass for the language model.

        Subclasses must implement this method according to `forward_contract`.

        For `forward_contract = "legacy"`, implement
        `forward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

        - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
        - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
        - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

        For `forward_contract = "typed_lm"`, implement `forward(request: dspy.LMRequest) -> dspy.LMResponse`.

        Raises:
            dspy.LMError: Base class for LM configuration, transport, provider,
                and unsupported-feature failures. Notable subclasses include
                `dspy.ContextWindowExceededError` for context-window failures,
                which adapters use to avoid inappropriate fallback retries when
                the prompt is too long. Each subclass should catch its
                provider's native context-window error and re-raise it as
                `dspy.ContextWindowExceededError`.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Async forward pass for the language model.

        Subclasses that support async calls must implement this method according to `forward_contract`.

        For `forward_contract = "legacy"`, implement
        `aforward(prompt=None, messages=None, **kwargs)` and return one of these OpenAI-like provider responses:

        - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
        - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
        - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

        For `forward_contract = "typed_lm"`, implement `aforward(request: dspy.LMRequest) -> dspy.LMResponse`.

        Raises:
            dspy.LMError: Base class for LM configuration, transport, provider,
                and unsupported-feature failures. Notable subclasses include
                `dspy.ContextWindowExceededError` for context-window failures,
                which adapters use to avoid inappropriate fallback retries when
                the prompt is too long. Each subclass should catch its
                provider's native context-window error and re-raise it as
                `dspy.ContextWindowExceededError`.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def dump_state(self) -> dict[str, Any]:
        """Return a sanitized reconstruction state for this LM.

        Subclasses whose state is captured by `BaseLM.__init__` can use this
        default. Subclasses with extra persistent state should override both
        `dump_state` and `load_state`.

        Returns:
            A dictionary that can be passed to `BaseLM.load_state`. The state
            excludes API keys.
        """
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)}
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": getattr(self, "num_retries", 3),
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> "BaseLM":
        """Reconstruct an LM from `dump_state` output.

        Legacy states without a class marker load as `dspy.LM`. Custom LM
        classes must be importable by their module-qualified class path and are
        only loaded when `allow_custom_lm_class=True`.

        Args:
            state: Serialized LM state produced by `dump_state`.
            allow_custom_lm_class: If True, allow importing and loading custom
                `BaseLM` subclasses recorded in `state`. Enable only for trusted
                state.

        Returns:
            The reconstructed LM instance.

        Raises:
            ValueError: If `state` references a custom LM class and
                `allow_custom_lm_class` is False.
            ImportError: If the serialized LM class cannot be imported.
            TypeError: If the serialized class is not a `BaseLM` subclass.
        """
        state = dict(state)
        class_path = state.pop(LM_CLASS_STATE_KEY, None)

        if cls is BaseLM:
            if class_path is None:
                # Legacy saved programs did not record the concrete LM class.
                from dspy.clients.lm import LM

                return LM(**state)

            if class_path != _BUILTIN_LM_CLASS_PATH and not allow_custom_lm_class:
                raise ValueError(
                    f"Refusing to import custom serialized LM class `{class_path}`. "
                    "Pass allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
                )

            lm_cls = _import_lm_class(class_path)
            if not issubclass(lm_cls, BaseLM):
                raise TypeError(f"Serialized LM class `{class_path}` must be a subclass of dspy.BaseLM.")
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)

        return cls(**state)

    def copy(self, **kwargs):
        """Return a copy of the language model with updated parameters.

        The default implementation makes a shallow runtime copy. Provider
        clients, sessions, and local model handles are preserved by reference.
        DSPy-owned mutable state is isolated for `history`, the `callbacks`
        list, and the `kwargs` dict. Other attributes are shared by reference.
        Subclasses with additional mutable DSPy-owned state should override this
        method.

        Args:
            **kwargs: Attribute or request-parameter updates to apply to the
                copy. For example, `lm.copy(rollout_id=1, temperature=1.0)`
                returns an LM whose requests use a different rollout ID at
                non-zero temperature to bypass cache collisions.

        Returns:
            A copied LM instance.
        """

        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if (key in new_instance.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False

        return new_instance

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry):
        if settings.disable_history:
            return

        # Global LM history
        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)

        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        # dspy.LM.history
        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)

        self.history.append(entry)

        # Per-module history
        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _process_completion(self, response, merged_kwargs):
        """Process the response of OpenAI chat completion API and extract outputs.

        Args:
            response: The OpenAI chat completion response
                https://platform.openai.com/docs/api-reference/chat/object
            merged_kwargs: Merged kwargs from self.kwargs and method kwargs

        Returns:
            List of processed outputs
        """
        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]

            if hasattr(c, "message") and hasattr(c.message, "reasoning_content") and c.message.reasoning_content:
                output["reasoning_content"] = c.message.reasoning_content

            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls

            # Extract citations from LiteLLM response if available
            citations = self._extract_citations_from_response(c)
            if citations:
                output["citations"] = citations

            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            # Return a list if every output only has "text" key
            outputs = [output["text"] for output in outputs]
        return outputs

    def _extract_citations_from_response(self, choice):
        """Extract citations from LiteLLM response if available.
        Reference: https://docs.litellm.ai/docs/providers/anthropic#beta-citations-api

        Args:
            choice: The choice object from response.choices

        Returns:
            A list of citation dictionaries or None if no citations found
        """
        try:
            # Check for citations in LiteLLM provider_specific_fields
            citations_data = choice.message.provider_specific_fields.get("citations")
            if isinstance(citations_data, list):
                return [citation for citations in citations_data for citation in citations]
        except Exception:
            return None

    def _process_response(self, response):
        """Process the response of OpenAI Response API and extract outputs.

        Args:
            response: OpenAI Response API response
                https://platform.openai.com/docs/api-reference/responses/object

        Returns:
            List of processed outputs, which is always of size 1 because the Response API only supports one output.
        """
        text_outputs = []
        tool_calls = []
        reasoning_contents = []

        for output_item in response.output:
            output_item_type = output_item.type
            if output_item_type == "message":
                for content_item in output_item.content:
                    text_outputs.append(content_item.text)
            elif output_item_type == "function_call":
                tool_calls.append(output_item.model_dump(exclude_none=True))
            elif output_item_type == "reasoning":
                if getattr(output_item, "content", None) and len(output_item.content) > 0:
                    for content_item in output_item.content:
                        reasoning_contents.append(content_item.text)
                elif getattr(output_item, "summary", None) and len(output_item.summary) > 0:
                    for summary_item in output_item.summary:
                        reasoning_contents.append(summary_item.text)

        result = {}
        if len(text_outputs) > 0:
            result["text"] = "".join(text_outputs)
        if len(tool_calls) > 0:
            result["tool_calls"] = tool_calls
        if len(reasoning_contents) > 0:
            result["reasoning_content"] = "".join(reasoning_contents)
        # All `response.output` items map to one answer, so we return a list of size 1.
        return [result]


def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """The global history shared across all LMs.

    Args:
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to. When
            provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
    """
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
