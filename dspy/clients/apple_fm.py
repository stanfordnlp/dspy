"""DSPy adapter for Apple's on-device Foundation Models.

Requires macOS 26+ with Apple Intelligence enabled and the Apple Foundation Models
SDK (``apple-fm-sdk``), installed via Apple's distribution channel. Refer to
Apple's documentation for installation instructions.
Usage::

    import dspy
    lm = dspy.AppleFoundationLM()
    dspy.configure(lm=lm)
    result = dspy.Predict("question -> answer")(question="What is DSPy?")
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import platform
from typing import Any, Iterator, Literal, get_args, get_origin

from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic OpenAI-compatible response types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FMMessage:
    """A single message in a completion response, mirroring OpenAI's format.

    Attributes:
        content: The text content of the message.
        tool_calls: Optional list of tool-call objects returned by the model.
    """

    content: str
    tool_calls: list[Any] | None = None


@dataclasses.dataclass
class _FMChoice:
    """One completion choice, mirroring OpenAI's ``Choice`` object.

    Attributes:
        message: The assistant message produced for this choice.
    """

    message: _FMMessage


@dataclasses.dataclass
class _FMUsage:
    """Token-usage statistics for a completion, mirroring OpenAI's ``Usage`` object.

    Implements the mapping protocol (``__iter__``) so that ``dict(usage)``
    works as expected by ``BaseLM`` history tracking.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Yield (key, value) pairs so that ``dict(usage)`` works as expected by BaseLM.

        Yields:
            Two-tuples of field name and integer value for each token count.
        """
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


@dataclasses.dataclass
class _FMResponse:
    """A completion response object compatible with ``BaseLM._process_completion``.

    Mirrors the subset of OpenAI's ``ChatCompletion`` that DSPy accesses.

    Attributes:
        choices: List of completion choices (always length 1 for Apple adapters).
        usage: Token-usage breakdown for the request.
        model: Model identifier string, echoed from the request.
        _hidden_params: DSPy internal metadata dict.  ``response_cost`` is set
            to ``0.0`` because on-device inference has no monetary cost.
    """

    choices: list[_FMChoice]
    usage: _FMUsage
    model: str
    # BaseLM reads getattr(response, "_hidden_params", {}).get("response_cost").
    # On-device inference has no monetary cost; declaring the field explicitly
    # satisfies the interface without relying on the getattr fallback.
    _hidden_params: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten a list of role/content message dicts into a single prompt string.

    Apple's ``LanguageModelSession.respond()`` takes a plain string rather than
    a structured message list.  System instructions are included as plain context
    at the top — bracket prefixes such as ``[System]:`` trigger Apple's on-device
    content guardrails and must be avoided.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts following the
            OpenAI chat format.  Multi-modal ``content`` lists are supported;
            only text blocks are extracted.

    Returns:
        A single string with all non-empty message contents joined by ``"\\n\\n"``.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Multi-modal content blocks — extract text parts only.
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        if not content:
            continue
        if role == "system":
            # No bracket prefix — "[System]:" triggers Apple's on-device content
            # guardrails (pattern-matched as a jailbreak attempt).  System content
            # is included as plain context at the top of the prompt, which is
            # semantically correct for a flat string.
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _pydantic_to_generable(model_cls: type, fm: Any) -> type | None:
    """Dynamically build an Apple ``@generable`` dataclass from a Pydantic model.

    Field type mapping:

    * ``Literal[a, b, c]`` → ``fm.guide(name, anyOf=[a, b, c])``
    * ``int`` with ``ge`` + ``le`` metadata → ``fm.guide(name, range=(ge, le))``
    * ``str`` with ``pattern`` metadata → ``fm.guide(name, regex=pattern)``
    * Everything else → plain annotation with no constraint (fallback)

    Args:
        model_cls: A Pydantic ``BaseModel`` subclass whose fields define the
            desired output schema.
        fm: The imported ``apple_fm_sdk`` module, passed explicitly to avoid
            re-importing inside the helper.

    Returns:
        A ``@generable``-decorated dataclass, or ``None`` if the conversion
        fails entirely (caller should fall back to prompt-based JSON schema).
    """
    fields: list[tuple[str, type, Any]] = []

    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)
        args = get_args(annotation)

        guide_kwargs: dict[str, Any] = {}
        raw_annotation = annotation

        if origin is Literal:
            guide_kwargs["anyOf"] = list(args)
            # Use the type of the first literal value as the field's raw annotation.
            raw_annotation = type(args[0]) if args else str
        else:
            ge_val: Any = None
            le_val: Any = None
            for meta in getattr(field_info, "metadata", []):
                if hasattr(meta, "ge") and meta.ge is not None:
                    ge_val = meta.ge
                if hasattr(meta, "le") and meta.le is not None:
                    le_val = meta.le
                pattern = getattr(meta, "pattern", None)
                if pattern:
                    guide_kwargs["regex"] = pattern
            if ge_val is not None and le_val is not None:
                # Both bounds present — map to an integer range constraint.
                guide_kwargs["range"] = (ge_val, le_val)

        if guide_kwargs:
            try:
                default = fm.guide(field_name, **guide_kwargs)
                fields.append((field_name, raw_annotation, default))
                continue
            except Exception:
                logger.warning(
                    "apple_fm: could not create fm.guide for field %r (%s), using unconstrained",
                    field_name,
                    guide_kwargs,
                )

        # Unconstrained field — use a sensible default so make_dataclass is happy.
        default_val: Any = "" if raw_annotation is str else None
        fields.append((field_name, raw_annotation, dataclasses.field(default=default_val)))

    try:
        dyn_cls = dataclasses.make_dataclass(model_cls.__name__, fields)
        return fm.generable(dyn_cls)
    except Exception as exc:
        logger.warning("apple_fm: failed to build @generable class from %r: %s", model_cls, exc)
        return None


def _run_async(coro: Any) -> Any:
    """Execute an async coroutine synchronously, regardless of event-loop state.

    Works in both plain Python scripts (no running event loop) and Jupyter
    notebooks / async frameworks (running event loop).  In the latter case,
    ``nest_asyncio`` must be installed::

        pip install nest_asyncio

    Args:
        coro: An awaitable coroutine to execute.

    Returns:
        The return value of the coroutine.

    Raises:
        RuntimeError: If called from within a running event loop and
            ``nest_asyncio`` is not installed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
        except ImportError:
            raise RuntimeError(
                "AppleFoundationLM.forward() was called from within a running event loop "
                "(e.g. a Jupyter notebook). Install nest_asyncio to enable this:\n"
                "    pip install nest_asyncio"
            )
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


# Cache of dynamically generated fm.Tool subclasses, keyed by (tool_name, id(func)).
# Avoids creating a throwaway class on every call for the same DSPy tool object.
# id(func) is stable for the lifetime of DSPy program objects.
_tool_class_cache: dict[tuple[str, int], type] = {}


def _dspy_tool_to_apple_tool(dspy_tool: Any, fm: Any) -> Any:
    """Wrap a DSPy tool in an Apple ``fm.Tool`` subclass.

    DSPy tools expose a ``.name`` attribute and are callable (or have a
    ``.func`` attribute).  The Apple tool's ``call()`` method delegates
    directly to that callable.

    Generated subclasses are cached by ``(tool_name, id(func))`` so the same
    class object is reused across calls for the same tool — Apple's SDK
    identifies tools by class name, which is preserved across calls.

    Args:
        dspy_tool: A DSPy tool object with a ``.name`` attribute and either
            being callable itself or exposing a ``.func`` attribute.
        fm: The imported ``apple_fm_sdk`` module.

    Returns:
        An instantiated ``fm.Tool`` subclass wired to the DSPy tool's callable.
    """
    tool_name = getattr(dspy_tool, "name", type(dspy_tool).__name__)
    func = dspy_tool if callable(dspy_tool) else getattr(dspy_tool, "func", None)

    cache_key = (tool_name, id(func))
    if cache_key not in _tool_class_cache:

        class _WrappedTool(fm.Tool):
            def call(self, **kwargs: Any) -> Any:
                """Delegate the tool call to the underlying DSPy callable."""
                if func is None:
                    raise NotImplementedError(f"Tool {tool_name!r} has no callable implementation")
                return func(**kwargs)

        _WrappedTool.__name__ = tool_name
        _tool_class_cache[cache_key] = _WrappedTool

    return _tool_class_cache[cache_key]()


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class AppleFoundationLM(BaseLM):
    """DSPy language model adapter for Apple's on-device Foundation Models.

    Wraps ``apple_fm_sdk.SystemLanguageModel`` + ``LanguageModelSession`` in a
    ``dspy.BaseLM`` subclass.  Key features:

    * **Native guided generation**: when ``response_format`` is a Pydantic
      model, the adapter dynamically builds an Apple ``@generable`` dataclass
      and uses the model's native constrained decoding instead of injecting a
      JSON schema into the prompt.
    * **Tool calling**: DSPy tools are converted to ``fm.Tool`` subclasses and
      registered on the session.
    * **Async bridging**: Apple's SDK is async-only; ``forward()`` bridges to
      sync via ``asyncio.run()`` with ``nest_asyncio`` support for notebooks.

    Requirements:
        * macOS 26+ with Apple Intelligence enabled.
        * ``pip install apple-fm-sdk``

    Args:
        model: Identifier string stored in history and cache keys.
            Currently only one on-device model exists, so this is cosmetic.
        temperature: Passed to ``GenerationOptions`` if supported by the SDK.
            ``None`` omits the option entirely (model uses its default).
        max_tokens: Reserved for future SDK support; logged but not yet wired.
        cache: Whether to enable DSPy's request cache.
        **kwargs: Additional keyword arguments forwarded to ``BaseLM.__init__``.

    Raises:
        RuntimeError: If not running on macOS, or if Apple Intelligence is
            unavailable on the current device.
        ImportError: If ``apple-fm-sdk`` is not installed.
    """

    def __init__(
        self,
        model: str = "apple/on-device",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter, validate platform, and connect to the on-device model.

        Args:
            model: Identifier string stored in history and cache keys.
            temperature: Sampling temperature, or ``None`` to use the model default.
            max_tokens: Maximum tokens to generate (reserved; not yet wired to SDK).
            cache: Whether to enable DSPy's request cache.
            **kwargs: Forwarded to ``BaseLM.__init__``.

        Raises:
            RuntimeError: If not running on macOS, or if Apple Intelligence is
                unavailable on the current device.
            ImportError: If ``apple-fm-sdk`` is not installed.
        """
        super().__init__(
            model=model,
            model_type="chat",
            # BaseLM requires concrete floats/ints; use safe defaults when not provided.
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens if max_tokens is not None else 1000,
            cache=cache,
            **kwargs,
        )

        if platform.system() != "Darwin":
            raise RuntimeError(
                "AppleFoundationLM requires macOS 26+ with Apple Intelligence enabled. "
                f"Current platform: {platform.system()!r}"
            )

        try:
            import apple_fm_sdk as fm
        except ImportError as exc:
            raise ImportError(
                "apple-fm-sdk is required to use AppleFoundationLM. Install it with:\n    pip install apple-fm-sdk"
            ) from exc

        self._fm = fm
        self._apple_model = fm.SystemLanguageModel()

        available, reason = self._apple_model.is_available()
        if not available:
            raise RuntimeError(
                f"Apple Foundation Model is not available on this device: {reason}\n"
                "Ensure Apple Intelligence is enabled in System Settings → Apple Intelligence & Siri."
            )

        self._temperature = temperature
        self._max_tokens = max_tokens
        # Apple's on-device model has a fixed context window.  The SDK does not expose this
        # value programmatically; 4096 is the documented limit for the initial release.
        # Update if Apple exposes a query API in a future SDK version.
        self.context_window: int = 4096

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_generation_options(self) -> Any | None:
        """Build a ``GenerationOptions`` object if any sampling params are set.

        Returns:
            A ``fm.GenerationOptions`` instance, or ``None`` if no sampling
            parameters have been configured (letting the SDK use its defaults).
        """
        fm = self._fm
        opts: dict[str, Any] = {}
        if self._temperature is not None:
            opts["temperature"] = self._temperature
        if not opts:
            return None
        try:
            return fm.GenerationOptions(**opts)
        except Exception as exc:
            logger.debug("apple_fm: could not create GenerationOptions(%s): %s", opts, exc)
            return None

    def _build_response(self, text: str) -> _FMResponse:
        """Wrap a raw text string in an OpenAI-compatible ``_FMResponse``.

        Args:
            text: The model's generated text.

        Returns:
            An ``_FMResponse`` with a single choice, zeroed usage counters
            (Apple's SDK does not expose token counts), and ``response_cost=0.0``.
        """
        return _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content=text))],
            usage=_FMUsage(),
            model=self.model,
            _hidden_params={"response_cost": 0.0},
        )

    # ------------------------------------------------------------------
    # BaseLM interface
    # ------------------------------------------------------------------

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Synchronous forward pass — checks DSPy cache, then bridges to ``aforward``.

        Args:
            prompt: Plain-text prompt string.  Mutually exclusive with ``messages``;
                if both are provided, ``messages`` takes precedence.
            messages: List of ``{"role": ..., "content": ...}`` dicts.  If
                ``None``, ``prompt`` is wrapped in a single user message.
            **kwargs: Generation parameters forwarded to ``aforward``.  Known
                DSPy-internal keys (``num_retries``, ``stream``, ``n``) are
                stripped before the cache key is built.

        Returns:
            An ``_FMResponse`` compatible with ``BaseLM._process_completion``.

        Raises:
            NotImplementedError: If ``stream=True`` is passed (not yet supported).
        """
        import dspy

        cache = kwargs.pop("cache", self.cache)

        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported for AppleFoundationLM. Call forward() for a blocking response."
            )

        # Normalise to a messages list so the cache key is consistent regardless
        # of whether the caller used prompt= or messages=.
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        # Build the cache-request dict using only semantically meaningful keys.
        # DSPy-internal / LiteLLM-only kwargs (num_retries, stream, …) are
        # excluded so they don't create spurious cache misses.
        _skip = {"num_retries", "stream", "n"}
        cache_request = {
            "model": self.model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k not in _skip},
        }

        if cache:
            cached = dspy.cache.get(cache_request)
            if cached is not None:
                return cached

        response = _run_async(self.aforward(prompt=None, messages=messages, **kwargs))

        if cache:
            dspy.cache.put(cache_request, response)

        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Async forward pass — the primary implementation for Apple on-device inference.

        Structured output path (``response_format`` is a Pydantic model):
            Builds an Apple ``@generable`` class from the Pydantic schema and
            calls ``session.respond(generating=...)``.  On failure, recreates
            the session and retries without the schema constraint so DSPy's
            prompt-based JSON injection handles it.

        Plain text path:
            Calls ``session.respond(prompt=...)`` and returns the string directly.

        Args:
            prompt: Plain-text prompt string.  Ignored if ``messages`` is provided.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Supports ``response_format`` (Pydantic model), ``tools``
                (list of DSPy tool objects), and standard DSPy-internal keys
                which are stripped before reaching the SDK.

        Returns:
            An ``_FMResponse`` with the model's generated text.
        """
        fm = self._fm

        flat_prompt = _flatten_messages(messages) if messages else (prompt or "")

        # Consume DSPy/LiteLLM-only kwargs so they don't reach the SDK.
        response_format = kwargs.pop("response_format", None)
        raw_tools = kwargs.pop("tools", [])
        kwargs.pop("num_retries", None)
        kwargs.pop("stream", None)
        kwargs.pop("n", None)
        kwargs.pop("cache", None)

        # Anything still in kwargs is unrecognised — warn and clear.
        # Apple's SDK does not accept arbitrary generation parameters (no top_p, stop, etc.).
        if kwargs:
            logger.warning(
                "apple_fm: ignoring unsupported kwargs %s (Apple SDK does not accept arbitrary generation parameters)",
                sorted(kwargs),
            )
            kwargs.clear()

        # Structured output: try native guided generation.
        generable_cls: type | None = None

        if response_format is not None:
            try:
                from pydantic import BaseModel as PydanticBaseModel

                if isinstance(response_format, type) and issubclass(response_format, PydanticBaseModel):
                    generable_cls = _pydantic_to_generable(response_format, fm)
            except ImportError:
                pass

            if generable_cls is None:
                logger.warning(
                    "apple_fm: response_format %r could not be mapped to @generable; "
                    "falling back to prompt-based JSON schema (structured output quality may vary)",
                    response_format,
                )

        # Tool conversion: wrap each DSPy tool in an fm.Tool subclass.
        apple_tools: list[Any] = []
        for tool in raw_tools:
            try:
                apple_tools.append(_dspy_tool_to_apple_tool(tool, fm))
            except Exception as exc:
                logger.warning("apple_fm: skipping tool %r: %s", tool, exc)

        # Build session kwargs, conditionally including tools.
        session_kwargs: dict[str, Any] = {"model": self._apple_model}
        if apple_tools:
            session_kwargs["tools"] = apple_tools
        session = fm.LanguageModelSession(**session_kwargs)

        # Build generation options (may be None if no sampling params are set).
        gen_opts = self._make_generation_options()
        respond_kwargs: dict[str, Any] = {}
        if gen_opts is not None:
            respond_kwargs["options"] = gen_opts

        # Call the model — prefer native constrained decoding when available.
        if generable_cls is not None:
            try:
                result = await session.respond(
                    prompt=flat_prompt,
                    generating=generable_cls,
                    **respond_kwargs,
                )
                # Serialize the @generable dataclass result to JSON so DSPy's
                # output parser receives the same format as the prompt-based path.
                text = json.dumps(dataclasses.asdict(result))
            except Exception as exc:
                logger.warning(
                    "apple_fm: native @generable generation failed (%s); "
                    "recreating session and retrying without schema constraint "
                    "(DSPy will handle JSON schema injection via prompt)",
                    exc,
                )
                # Recreate: a session that raised during respond() may be in an
                # undefined state and must not be reused for the fallback call.
                del session
                session = fm.LanguageModelSession(**session_kwargs)
                text = await session.respond(prompt=flat_prompt, **respond_kwargs)
        else:
            text = await session.respond(prompt=flat_prompt, **respond_kwargs)

        # Prompt ARC to release underlying OS objects before returning.
        del session
        return self._build_response(text)
