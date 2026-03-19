"""DSPy adapter for locally-managed Apple Silicon models.

Supports two inference backends via the ``backend`` parameter:

* **mlx** — runs any ``mlx-lm``-compatible model (HuggingFace repo or local
  directory) using Apple's `MLX <https://github.com/ml-explore/mlx>`_ framework.
  Hundreds of pre-quantized models are available under the ``mlx-community/``
  HuggingFace organisation.

* **coreml** — reserved for compiled ``.mlpackage`` models via ``coremltools``.
  Not yet implemented; contributions welcome.

----

The primary use case is as a **free, private, offline preprocessing layer**
in DSPy pipelines that otherwise use expensive cloud LLMs.  Because DSPy
supports per-module LM overrides, you can route cheap grunt-work stages to
``AppleLocalLM`` and reserve Anthropic/OpenAI for the reasoning steps::

    import dspy

    local_lm  = dspy.AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit")
    cloud_lm  = dspy.LM("anthropic/claude-sonnet-4-6")

    class PreprocessAndReason(dspy.Module):
        def __init__(self):
            # extraction runs on-device at zero cost
            self.extract = dspy.Predict("raw_text -> entities, dates, sentiment_passages",
                                        lm=local_lm)
            # reasoning runs in the cloud
            self.reason  = dspy.Predict("entities, dates, sentiment_passages -> verdict",
                                        lm=cloud_lm)

        def forward(self, raw_text):
            extracted = self.extract(raw_text=raw_text)
            return self.reason(**extracted)

Requirements (MLX backend):
    * macOS 14+ on Apple Silicon (M1 / M2 / M3 / M4)
    * ``pip install mlx-lm``
"""

from __future__ import annotations

import asyncio
import logging
import platform
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from dspy.clients.apple_fm import _FMChoice, _FMMessage, _FMResponse, _FMUsage
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("mlx", "coreml")


@dataclass
class _LocalStreamChunk:
    """A single token chunk emitted during AppleLocalLM streaming.

    Yielded by ``dspy.streamify()`` for each token as the model generates.
    Not a ``ModelResponseStream`` instance, so DSPy's ``StreamListener``
    field-extraction is not available — callers receive raw token text via
    ``chunk.text`` and accumulate it manually.

    Attributes:
        text: The raw token text for this chunk.
        model: Model identifier, echoed from the ``AppleLocalLM`` instance.
        predict_id: ``id()`` of the ``dspy.Predict`` module that initiated the
            call, or ``None`` if called outside a Predict context.
    """

    text: str
    model: str
    predict_id: int | None = None


# ---------------------------------------------------------------------------
# Chat-template helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Format a message list using the tokenizer's built-in chat template.

    Falls back to simple role-prefixed concatenation if the tokenizer does not
    expose ``apply_chat_template`` or if the call raises.

    Args:
        tokenizer: A HuggingFace tokenizer object, or any object that optionally
            exposes an ``apply_chat_template`` method.
        messages: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        A formatted prompt string ready to pass to ``mlx_lm.generate()``.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.debug("apply_chat_template failed (%s); falling back to concat", exc)

    # Fallback: reuse the same simple flattener as AppleFoundationLM.
    from dspy.clients.apple_fm import _flatten_messages  # noqa: PLC0415

    return _flatten_messages(messages)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class AppleLocalLM(BaseLM):
    """DSPy language model adapter for locally-managed Apple Silicon models.

    Wraps ``mlx-lm`` (and in future ``coremltools``) inside a ``dspy.BaseLM``
    subclass so that any MLX-compatible model can participate in a DSPy
    pipeline, including mixed-LM pipelines where cheap on-device inference
    handles preprocessing and cloud LLMs handle reasoning.

    Streaming is supported via ``dspy.streamify()``: tokens are forwarded to
    the anyio ``MemoryObjectSendStream`` as ``_LocalStreamChunk`` objects.

    Args:
        model: HuggingFace repo ID (e.g. ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``)
            or an absolute path to a local MLX model directory.  Also used as the
            DSPy cache key, so two instances pointing to the same model share a cache.
        backend: Inference engine.  ``"mlx"`` (default) uses ``mlx-lm``.
            ``"coreml"`` is reserved and raises ``NotImplementedError`` until
            implemented.
        bits: Informational hint for the expected quantization level (4 or 8).
            Logged at init time and stored on the instance.  Does **not**
            trigger automatic quantization — convert the model first with
            ``mlx_lm.convert()`` if needed.
        temperature: Sampling temperature passed to ``mlx_lm.generate()``.
        max_tokens: Maximum tokens to generate per call.
        cache: Whether to enable DSPy's request cache (keyed on model + prompt).
        max_concurrency: Maximum number of simultaneous ``aforward()`` calls
            when used outside ``streamify()``.  MLX thread-safety on a single
            model instance is not guaranteed above 1.
        **kwargs: Additional keyword arguments forwarded to ``BaseLM.__init__``.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
        NotImplementedError: If ``backend="coreml"`` is requested.
        RuntimeError: If not running on macOS on Apple Silicon.
        ImportError: If ``mlx-lm`` is not installed.

    Example::

        import dspy

        lm = dspy.AppleLocalLM(
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            bits=4,
            temperature=0.0,
            max_tokens=512,
        )
        dspy.configure(lm=lm)
        result = dspy.Predict("question -> answer")(question="What is DSPy?")
    """

    def __init__(
        self,
        model: str,
        backend: str = "mlx",
        bits: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        cache: bool = True,
        max_concurrency: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter, validate platform and backend, then load the model.

        Args:
            model: HuggingFace repo ID or local path to an MLX model directory.
            backend: Inference engine — only ``"mlx"`` is currently supported.
            bits: Informational quantization hint (4 or 8); does not trigger conversion.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate per call.
            cache: Whether to enable DSPy's request cache.
            max_concurrency: Semaphore limit for concurrent ``aforward()`` calls.
            **kwargs: Forwarded to ``BaseLM.__init__``.

        Raises:
            ValueError: If ``backend`` is not in ``_SUPPORTED_BACKENDS``.
            NotImplementedError: If ``backend="coreml"`` is requested.
            RuntimeError: If not running on macOS on Apple Silicon (arm64).
            ImportError: If ``mlx-lm`` is not installed.
        """
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose from: {_SUPPORTED_BACKENDS}"
            )

        if backend == "coreml":
            raise NotImplementedError(
                "CoreML backend is not yet implemented. "
                "Contributions welcome — see dspy/clients/apple_local.py.\n"
                "For now, use backend='mlx' or dspy.AppleFoundationLM() for the "
                "Apple Intelligence system model."
            )

        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        if platform.system() != "Darwin":
            raise RuntimeError(
                "AppleLocalLM requires macOS on Apple Silicon. "
                f"Current platform: {platform.system()!r}"
            )
        if platform.machine() != "arm64":
            raise RuntimeError(
                "AppleLocalLM requires Apple Silicon (arm64). "
                f"Current architecture: {platform.machine()!r}"
            )

        self._backend = backend
        self._bits = bits
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_concurrency = max_concurrency
        if max_concurrency > 1:
            logger.warning(
                "AppleLocalLM: max_concurrency=%d — MLX generate() thread-safety "
                "on a single model instance is not guaranteed. "
                "If you observe crashes or hangs, reduce to max_concurrency=1.",
                max_concurrency,
            )
        # Lazily initialised in aforward() to avoid binding to the wrong event loop.
        self._semaphore: asyncio.Semaphore | None = None

        if bits is not None:
            logger.info(
                "AppleLocalLM: loading %r (expected %d-bit quantization)", model, bits
            )

        self._mlx_model, self._mlx_tokenizer = self._load_mlx(model)
        # HuggingFace tokenizers carry model_max_length in their saved config.
        # Fall back to 4096 for tokenizers that omit it (conservative but safe).
        self.context_window: int = getattr(self._mlx_tokenizer, "model_max_length", 4096)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_mlx(self, model_path: str) -> tuple[Any, Any]:
        """Load an MLX model and tokenizer from a HuggingFace repo or local path.

        Args:
            model_path: HuggingFace repo ID or absolute path to a local model directory.

        Returns:
            A ``(model, tokenizer)`` tuple as returned by ``mlx_lm.load()``.

        Raises:
            ImportError: If ``mlx-lm`` is not installed.
        """
        try:
            import mlx_lm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for AppleLocalLM(backend='mlx'). Install it:\n"
                "    pip install mlx-lm"
            ) from exc

        logger.info("AppleLocalLM: loading model %r via mlx-lm…", model_path)
        model, tokenizer = mlx_lm.load(model_path)
        logger.info("AppleLocalLM: model loaded")
        return model, tokenizer

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Run synchronous (non-streaming) MLX inference.

        Only the semantically meaningful arguments are accepted; DSPy/LiteLLM
        kwargs are stripped before this method is called.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts to format
                via the tokenizer's chat template.
            temperature: Sampling temperature forwarded to ``make_sampler``.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            A ``(generated_text, flat_prompt)`` tuple.  ``flat_prompt`` is
            returned so the caller can compute token counts without applying
            the chat template a second time.
        """
        import mlx_lm  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)

        sampler = make_sampler(temp=float(temperature))
        text = mlx_lm.generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=flat_prompt,
            max_tokens=int(max_tokens),
            sampler=sampler,
            verbose=False,
        )
        return text, flat_prompt

    def _build_response(self, text: str, usage: _FMUsage | None = None) -> _FMResponse:
        """Wrap a raw text string in an OpenAI-compatible ``_FMResponse``.

        Args:
            text: The model's generated text.
            usage: Pre-computed token-usage statistics.  Defaults to zeroed
                counters if not provided.

        Returns:
            An ``_FMResponse`` with a single choice and ``response_cost=0.0``.
        """
        return _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content=text))],
            usage=usage or _FMUsage(),
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
        """Synchronous forward pass — checks DSPy cache, then runs MLX inference.

        When ``dspy.settings.send_stream`` is set (i.e. the call originates from
        ``dspy.streamify()`` via ``asyncify``), uses ``mlx_lm.stream_generate()``
        and forwards each token to the anyio stream via ``anyio.from_thread.run()``.
        Otherwise runs ``mlx_lm.generate()`` in a single blocking call.

        Args:
            prompt: Plain-text prompt string.  Wrapped in a user message if
                ``messages`` is not provided.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Supports ``temperature``, ``max_tokens``, and ``cache``
                overrides.  Unknown kwargs are warned and discarded.

        Returns:
            An ``_FMResponse`` compatible with ``BaseLM._process_completion``.

        Raises:
            NotImplementedError: If ``tools`` or ``stream=True`` is passed.
        """
        import dspy  # noqa: PLC0415  (lazy to avoid circular-import at module load)

        cache = kwargs.pop("cache", self.cache)

        # Normalise to a messages list for consistent cache keying.
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        # Extract generation params; let per-call overrides take priority.
        temperature = float(kwargs.pop("temperature", self._temperature))
        max_tokens = int(kwargs.pop("max_tokens", self._max_tokens))

        # mlx-lm has no native tool API — raise early rather than silently dropping.
        if kwargs.get("tools"):
            raise NotImplementedError(
                "Tool calling is not supported for AppleLocalLM (mlx-lm has no native tool API). "
                "Use AppleFoundationLM for native tool support on macOS 26+."
            )

        if kwargs.get("stream"):
            raise NotImplementedError(
                "AppleLocalLM does not support stream=True in forward(). "
                "Use dspy.streamify() to wrap your module for async streaming."
            )

        # Discard LiteLLM-only / DSPy-internal kwargs that mlx_lm doesn't accept.
        for _k in ("response_format", "tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)

        # Anything still in kwargs is unrecognised — warn and clear so it doesn't
        # silently pollute the cache key without affecting generation.
        if kwargs:
            logger.warning(
                "apple_local: ignoring unsupported kwargs %s "
                "(mlx-lm does not accept arbitrary generation parameters)",
                sorted(kwargs),
            )
            kwargs.clear()

        cache_request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        send_stream = dspy.settings.send_stream

        # Skip cache lookup during streaming — every call must emit live chunks.
        if cache and send_stream is None:
            cached = dspy.cache.get(cache_request)
            if cached is not None:
                return cached

        if send_stream is not None:
            # Streaming path: called from streamify() via asyncify (anyio-managed thread).
            # stream_generate() yields tokens synchronously; each is forwarded to the
            # anyio MemoryObjectSendStream using from_thread.run().
            import mlx_lm  # noqa: PLC0415
            from anyio.from_thread import run as _anyio_run  # noqa: PLC0415
            from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

            caller_predict = dspy.settings.caller_predict
            # id() is stable for the lifetime of the Predict object; used by
            # StreamListener to route chunks to the correct field extractor.
            predict_id = id(caller_predict) if caller_predict else None
            flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
            sampler = make_sampler(temp=float(temperature))

            full_text = ""
            for _response in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=flat_prompt,
                max_tokens=int(max_tokens),
                sampler=sampler,
            ):
                full_text += _response.text
                _chunk = _LocalStreamChunk(
                    text=_response.text, model=self.model, predict_id=predict_id
                )
                # anyio.from_thread.run() schedules the async send on the event loop
                # from within the anyio-managed worker thread started by asyncify.
                _anyio_run(send_stream.send, _chunk)

            text = full_text
        else:
            text, flat_prompt = self._generate(messages, temperature, max_tokens)

        prompt_tokens = len(self._mlx_tokenizer.encode(flat_prompt))
        if prompt_tokens > self.context_window - max_tokens:
            logger.warning(
                "apple_local: prompt (%d tokens) + max_tokens (%d) exceeds context_window (%d); "
                "generation may be truncated",
                prompt_tokens, max_tokens, self.context_window,
            )
        completion_tokens = len(self._mlx_tokenizer.encode(text))
        usage = _FMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        response = self._build_response(text, usage=usage)

        if cache:
            dspy.cache.put(cache_request, response)

        return response

    async def _stream_generate_async(
        self,
        flat_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Bridge ``mlx_lm.stream_generate()`` (sync generator) to an async generator.

        Runs the synchronous MLX generator in a thread-pool executor and forwards
        each token text to the caller via an ``asyncio.Queue``, keeping the event
        loop unblocked between tokens.

        Args:
            flat_prompt: Pre-formatted prompt string from ``_apply_chat_template``.
            temperature: Sampling temperature forwarded to ``make_sampler``.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual token strings as they are produced by the model.
        """
        import mlx_lm  # noqa: PLC0415
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        sampler = make_sampler(temp=float(temperature))

        def _run() -> None:
            """Generate tokens in a thread and enqueue each one for the async consumer."""
            for response in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=flat_prompt,
                max_tokens=int(max_tokens),
                sampler=sampler,
            ):
                # call_soon_threadsafe is required to safely enqueue from a non-async thread.
                loop.call_soon_threadsafe(queue.put_nowait, response.text)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # None sentinel signals completion

        # Run the blocking generator concurrently; await its completion after draining.
        executor_task = asyncio.ensure_future(asyncio.to_thread(_run))
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
        await executor_task

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Async forward pass for direct callers of ``await lm.aforward()``.

        When ``dspy.settings.send_stream`` is set, runs ``mlx_lm.stream_generate()``
        via ``_stream_generate_async`` and sends each ``_LocalStreamChunk`` to the
        stream as tokens arrive, then returns the full ``_FMResponse`` at the end.

        Without streaming, delegates to ``forward()`` via a thread-pool executor so
        MLX inference does not block the event loop.  Concurrent calls are gated by a
        semaphore (default: 1) to prevent out-of-memory errors on Apple Silicon.

        Note:
            The primary streaming path for ``dspy.streamify()`` goes through
            ``forward()`` (via ``asyncify``), not this method.  This method handles
            streaming only for direct ``await lm.aforward()`` callers.

        Args:
            prompt: Plain-text prompt string.  Wrapped in a user message if
                ``messages`` is not provided.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Supports ``temperature``, ``max_tokens``, and ``cache``
                overrides.  Unknown kwargs are warned and discarded.

        Returns:
            An ``_FMResponse`` compatible with ``BaseLM._process_completion``.
        """
        import dspy  # noqa: PLC0415

        send_stream = dspy.settings.send_stream

        if send_stream is None:
            # Non-streaming path: delegate to forward() in a thread-pool executor.
            if self._semaphore is None:
                # Lazy init avoids binding to the wrong event loop at construction time.
                self._semaphore = asyncio.Semaphore(self._max_concurrency)
            async with self._semaphore:
                return await asyncio.to_thread(
                    self.forward, prompt=prompt, messages=messages, **kwargs
                )

        # ------------------------------------------------------------------
        # Streaming path (direct aforward() callers only)
        # ------------------------------------------------------------------
        cache = kwargs.pop("cache", self.cache)
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        temperature = float(kwargs.pop("temperature", self._temperature))
        max_tokens = int(kwargs.pop("max_tokens", self._max_tokens))
        # Strip DSPy-internal keys that mlx-lm doesn't accept.
        for _k in ("response_format", "tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)
        if kwargs:
            logger.warning(
                "apple_local: ignoring unsupported kwargs %s", sorted(kwargs)
            )

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
        caller_predict = dspy.settings.caller_predict
        # id() is stable for the lifetime of the Predict object.
        predict_id = id(caller_predict) if caller_predict else None

        full_text = ""
        async for token_text in self._stream_generate_async(flat_prompt, temperature, max_tokens):
            full_text += token_text
            chunk = _LocalStreamChunk(text=token_text, model=self.model, predict_id=predict_id)
            await send_stream.send(chunk)

        prompt_tokens = len(self._mlx_tokenizer.encode(flat_prompt))
        completion_tokens = len(self._mlx_tokenizer.encode(full_text))
        usage = _FMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        response = self._build_response(full_text, usage=usage)

        if cache:
            cache_request = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            dspy.cache.put(cache_request, response)

        return response
