import datetime
import inspect
import uuid
import warnings
from typing import Any, TextIO

from dspy.clients.language_models.base import LanguageModel
from dspy.dsp.utils import settings
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []

# Sentinel class attribute. Set in __init_subclass__ from the subclass's
# `forward` signature. v1: legacy `forward(prompt, messages, **kwargs)` returning
# an OpenAI-shaped object. v2: typed `forward(request: LMRequest) -> LMResponse`.
_LM_MIGRATION_URL = "https://dspy.ai/migration/baselm"


def _detect_contract_version(cls: type) -> int:
    """Return 1 for legacy forward(prompt, messages) and 2 for typed forward(request).

    Heuristic:
      - Subclass defines a `forward` parameter named `prompt` or `messages` → v1.
      - Subclass defines a single non-self positional parameter (optionally
        annotated `LMRequest`) → v2.
      - Subclass does not override `forward` (BaseLM default) → v2.
      - Anything else (e.g. `*args, **kwargs` passthrough) → v1 (legacy is the
        safer default during the deprecation cycle).
    """
    fwd = cls.__dict__.get("forward")
    if fwd is None:
        # Subclass didn't override `forward`; treat as v2 (BaseLM default raises).
        return 2
    try:
        sig = inspect.signature(fwd)
    except (TypeError, ValueError):
        return 1
    params = [p for p in sig.parameters.values() if p.name != "self"]
    names = {p.name for p in params}
    if "prompt" in names or "messages" in names:
        return 1
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1:
        return 2
    return 1


class BaseLM(LanguageModel):
    """Base class for handling LLM calls.

    `BaseLM` is the single normalized base class for DSPy language models. New
    implementations override **typed** `forward(self, request: LMRequest) -> LMResponse`
    (plus optionally `aforward`, `forward_stream`, `aforward_stream`,
    `normalize_error`, `dump_state`, `load_state`).

    Legacy subclasses that override `forward(self, prompt, messages=None, **kwargs)`
    and return an OpenAI-shaped response continue to work during the
    deprecation cycle. The base class detects the legacy signature at
    class-definition time, emits a one-shot `DeprecationWarning` pointing at
    the migration guide, and routes calls through a translation shim.

    The legacy `forward(prompt, messages)` signature is removed in DSPy 4.0.

    Examples:

    Legacy v1 subclass (deprecated, will warn at class definition):

    ```python
    class MyLM(dspy.BaseLM):
        def forward(self, prompt, messages=None, **kwargs):
            ...  # returns OpenAI-shaped response
    ```

    New v2 subclass (recommended):

    ```python
    class MyLM(dspy.BaseLM):
        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            ...  # returns a typed LMResponse

        @property
        def capabilities(self) -> dspy.LMCapabilities:
            return dspy.LMCapabilities(function_calling=True, streaming=True)
    ```
    """

    # The base class itself is v2 (its `forward` is the typed NotImplementedError).
    _lm_contract_version: int = 2

    def __init_subclass__(cls, *, _internal: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._lm_contract_version = _detect_contract_version(cls)
        if _internal or cls.__module__.startswith("dspy."):
            return
        if cls._lm_contract_version == 1:
            warnings.warn(
                "Subclassing dspy.BaseLM with `forward(self, prompt, messages, ...)` is "
                "the legacy LM contract. The legacy signature is deprecated and will be "
                "removed in DSPy 4.0. Override "
                "`forward(self, request: LMRequest) -> LMResponse` instead. "
                f"See {_LM_MIGRATION_URL}.",
                DeprecationWarning,
                stacklevel=2,
            )

    _V1_DEFAULT_TEMPERATURE = 0.0
    _V1_DEFAULT_MAX_TOKENS = 1000
    _UNSET: Any = object()

    def __init__(
        self,
        model,
        model_type="chat",
        temperature: Any = _UNSET,
        max_tokens: Any = _UNSET,
        cache: bool = True,
        **kwargs,
    ):
        """Unified BaseLM constructor.

        For v1 subclasses (legacy `forward(prompt, messages)`), historical
        defaults apply: `temperature=0.0`, `max_tokens=1000`. For v2 subclasses
        (typed `forward(request)`), omitted values stay omitted from
        `self.kwargs` (matching `LanguageModel.__init__` semantics).
        """
        callbacks = kwargs.pop("callbacks", None)
        num_retries = kwargs.pop("num_retries", 0)

        is_v1 = type(self)._lm_contract_version == 1
        if temperature is BaseLM._UNSET:
            temperature = self._V1_DEFAULT_TEMPERATURE if is_v1 else None
        if max_tokens is BaseLM._UNSET:
            max_tokens = self._V1_DEFAULT_MAX_TOKENS if is_v1 else None

        LanguageModel.__init__(
            self,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self.model_type = model_type

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

    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: Any = None,
        **kwargs,
    ):
        """Dispatch on the subclass's `_lm_contract_version`.

        - v2 subclass (typed `forward(request)`), or `request=` passed: route through
          `LanguageModel.__call__` (which fires its own callbacks) and return `LMResponse`.
        - v1 subclass (legacy `forward(prompt, messages)`) called without `request=`:
          preserve the historical `list[str | dict]` return via `_v1_call`.
        """
        if request is not None or self._lm_contract_version == 2:
            return LanguageModel.__call__(
                self,
                *items,
                prompt=prompt,
                messages=messages,
                request=request,
                **kwargs,
            )
        # v1 backward-compat: `lm("text")` historically meant prompt="text".
        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional "
                "content items require a v2 subclass. Use prompt= or messages=."
            )
        return self._v1_call(prompt=prompt, messages=messages, **kwargs)

    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: Any = None,
        **kwargs,
    ):
        if request is not None or self._lm_contract_version == 2:
            return await LanguageModel.acall(
                self,
                *items,
                prompt=prompt,
                messages=messages,
                request=request,
                **kwargs,
            )
        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional "
                "content items require a v2 subclass. Use prompt= or messages=."
            )
        return await self._v1_acall(prompt=prompt, messages=messages, **kwargs)

    @with_callbacks
    def _v1_call(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    @with_callbacks
    async def _v1_acall(self, prompt=None, messages=None, **kwargs):
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    # `forward` and `aforward` are inherited from LanguageModel with the typed
    # signature `forward(self, request: LMRequest) -> LMResponse`. Legacy
    # subclasses that override `forward(self, prompt, messages, **kwargs)` are
    # detected at class-definition time by `__init_subclass__` (see
    # `_lm_contract_version`) and routed through `__call__`'s v1 dispatch arm
    # until DSPy 4.0.

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters.

        v2 subclasses use the typed `LanguageModel.copy` (shallow copy of provider
        resources, isolated DSPy state). v1 subclasses preserve historical
        `copy.deepcopy` behavior.
        """
        if self._lm_contract_version == 2:
            return LanguageModel.copy(self, **kwargs)

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
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
