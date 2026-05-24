import copy as copy_module
import datetime
import importlib
import inspect
import uuid
from typing import Any, TextIO

from dspy.dsp.utils import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []
LM_CLASS_STATE_KEY = "_dspy_lm_class"
_BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"


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
    """Base class for handling LLM calls.

    Most users can directly use the `dspy.LM` class, which is a subclass of `BaseLM`. Users can also implement their
    own subclasses of `BaseLM` to support custom LLM providers and inject custom logic. To do so, simply override the
    `forward` method and make sure the return format is identical to the
    [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).

    Subclasses whose state is captured by `BaseLM.__init__` can use the default `dump_state` and `load_state`
    methods. Subclasses with extra persistent state should override both methods.

    Examples:

    ```python
    from openai import OpenAI

    import dspy


    class MyLM(dspy.BaseLM):
        @property
        def supports_function_calling(self) -> bool:
            return self.model.startswith("openai/gpt-4o")

        @property
        def supports_reasoning(self) -> bool:
            return self.model.startswith("anthropic/claude-3-7")

        @property
        def supports_response_schema(self) -> bool:
            return self.model.startswith("openai/gpt-4o")

        @property
        def supported_params(self) -> set[str]:
            if self.model.startswith("openai/gpt-4o"):
                return {"response_format"}  # accepts response_format=...
            return set()

        def forward(self, prompt, messages=None, **kwargs):
            client = OpenAI()
            return client.chat.completions.create(
                model=self.model,
                messages=messages or [{"role": "user", "content": prompt}],
                **self.kwargs,
            )


    lm = MyLM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    print(dspy.Predict("q->a")(q="Why did the chicken cross the kitchen?"))
    ```
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
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)

        return outputs

    @with_callbacks
    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)
        return outputs

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Forward pass for the language model.

        Subclasses must implement this method, and the response should be identical to either of the following formats:

        - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
        - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
        - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

        Raises:
            dspy.ContextWindowExceededError: When the request fails because the
                input exceeds the model's context window. DSPy adapters and
                modules rely on this error to trigger fallback behavior (e.g.
                truncating the prompt and retrying). Each subclass is
                responsible for catching its provider's native error and
                re-raising it as `dspy.ContextWindowExceededError`.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Async forward pass for the language model.

        Subclasses must implement this method, and the response should be identical to either of the following formats:

        - [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object)
        - [OpenAI chat completion format](https://platform.openai.com/docs/api-reference/chat/object)
        - [OpenAI text completion format](https://platform.openai.com/docs/api-reference/completions/object)

        Raises:
            dspy.ContextWindowExceededError: When the request fails because the
                input exceeds the model's context window. DSPy adapters and
                modules rely on this error to trigger fallback behavior (e.g.
                truncating the prompt and retrying). Each subclass is
                responsible for catching its provider's native error and
                re-raising it as `dspy.ContextWindowExceededError`.
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
                tool_calls.append(output_item.model_dump())
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
