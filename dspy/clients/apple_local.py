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
import json
import logging
import platform
from typing import Any

from dspy.clients.apple_base import _AppleBaseLM, _FMResponse, _FMUsage
from dspy.clients.apple_local_mlx import (
    _apply_chat_template,
    _LocalStreamChunk,
    _MLXMixin,
    _response_format_to_schema,
)

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("mlx", "coreml")


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class AppleLocalLM(_AppleBaseLM, _MLXMixin):
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
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {_SUPPORTED_BACKENDS}")

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
            raise RuntimeError(f"AppleLocalLM requires macOS on Apple Silicon. Current platform: {platform.system()!r}")
        if platform.machine() != "arm64":
            raise RuntimeError(
                f"AppleLocalLM requires Apple Silicon (arm64). Current architecture: {platform.machine()!r}"
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
        # Cache of compiled outlines logits processors, keyed by JSON-serialised schema.
        # Building an FSM from a schema is expensive; cache per instance so it is paid
        # at most once per schema per loaded model.
        self._schema_processor_cache: dict[str, Any] = {}

        if bits is not None:
            logger.info("AppleLocalLM: loading %r (expected %d-bit quantization)", model, bits)

        self._mlx_model, self._mlx_tokenizer = self._load_mlx(model)
        # HuggingFace tokenizers carry model_max_length in their saved config.
        # Fall back to 4096 for tokenizers that omit it (conservative but safe).
        self.context_window: int = getattr(self._mlx_tokenizer, "model_max_length", 4096)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_usage(self, flat_prompt: str, text: str, max_tokens: int) -> _FMUsage:
        """Compute token-usage statistics and warn if the context window is tight.

        Args:
            flat_prompt: The formatted prompt string that was sent to the model.
            text: The generated text returned by the model.
            max_tokens: The maximum tokens budget for this call (used for the
                context-window headroom check).

        Returns:
            An ``_FMUsage`` instance with prompt, completion, and total token counts.
        """
        prompt_tokens = len(self._mlx_tokenizer.encode(flat_prompt))
        if prompt_tokens > self.context_window - max_tokens:
            logger.warning(
                "apple_local: prompt (%d tokens) + max_tokens (%d) exceeds context_window (%d); "
                "generation may be truncated",
                prompt_tokens,
                max_tokens,
                self.context_window,
            )
        completion_tokens = len(self._mlx_tokenizer.encode(text))
        return _FMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
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
        import dspy

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

        # Intercept response_format before the discard loop so we can build a
        # constrained logits processor from it.  All other LiteLLM/DSPy-internal
        # kwargs that mlx_lm doesn't understand are discarded below.
        response_format = kwargs.pop("response_format", None)
        for _k in ("tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)

        schema = _response_format_to_schema(response_format)
        logits_processors: list[Any] | None = None
        if schema is not None:
            proc = self._build_schema_processor(schema)
            if proc is not None:
                logits_processors = [proc]

        # Anything still in kwargs is unrecognised — warn and clear so it doesn't
        # silently pollute the cache key without affecting generation.
        if kwargs:
            logger.warning(
                "apple_local: ignoring unsupported kwargs %s (mlx-lm does not accept arbitrary generation parameters)",
                sorted(kwargs),
            )
            kwargs.clear()

        cache_request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if schema is not None:
            # Include the schema in the cache key so constrained and unconstrained
            # calls for the same prompt are cached separately.
            cache_request["response_schema"] = json.dumps(schema, sort_keys=True)

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
            import mlx_lm
            from anyio.from_thread import run as _anyio_run
            from mlx_lm.sample_utils import make_sampler

            caller_predict = dspy.settings.caller_predict
            # id() is stable for the lifetime of the Predict object; used by
            # StreamListener to route chunks to the correct field extractor.
            predict_id = id(caller_predict) if caller_predict else None
            flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
            sampler = make_sampler(temp=float(temperature))

            stream_kwargs: dict[str, Any] = {
                "max_tokens": int(max_tokens),
                "sampler": sampler,
            }
            if logits_processors:
                stream_kwargs["logits_processors"] = logits_processors

            _chunks: list[str] = []
            for _response in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=flat_prompt,
                **stream_kwargs,
            ):
                _chunks.append(_response.text)
                _chunk = _LocalStreamChunk(text=_response.text, model=self.model, predict_id=predict_id)
                # anyio.from_thread.run() schedules the async send on the event loop
                # from within the anyio-managed worker thread started by asyncify.
                _anyio_run(send_stream.send, _chunk)

            text = "".join(_chunks)
        else:
            text, flat_prompt = self._generate(messages, temperature, max_tokens, logits_processors)

        usage = self._compute_usage(flat_prompt, text, max_tokens)
        response = self._build_response(text, usage=usage)

        if cache:
            dspy.cache.put(cache_request, response)

        return response

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
        import dspy

        send_stream = dspy.settings.send_stream

        if send_stream is None:
            # Non-streaming path: delegate to forward() in a thread-pool executor.
            if self._semaphore is None:
                # Lazy init avoids binding to the wrong event loop at construction time.
                self._semaphore = asyncio.Semaphore(self._max_concurrency)
            async with self._semaphore:
                return await asyncio.to_thread(self.forward, prompt=prompt, messages=messages, **kwargs)

        # ------------------------------------------------------------------
        # Streaming path (direct aforward() callers only)
        # ------------------------------------------------------------------
        cache = kwargs.pop("cache", self.cache)
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        temperature = float(kwargs.pop("temperature", self._temperature))
        max_tokens = int(kwargs.pop("max_tokens", self._max_tokens))
        # Intercept response_format before discarding to build a constrained
        # logits processor when outlines is available.
        response_format = kwargs.pop("response_format", None)
        for _k in ("tools", "num_retries", "stream", "n"):
            kwargs.pop(_k, None)
        if kwargs:
            logger.warning("apple_local: ignoring unsupported kwargs %s", sorted(kwargs))

        schema = _response_format_to_schema(response_format)
        logits_processors: list[Any] | None = None
        if schema is not None:
            proc = self._build_schema_processor(schema)
            if proc is not None:
                logits_processors = [proc]

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)
        caller_predict = dspy.settings.caller_predict
        # id() is stable for the lifetime of the Predict object.
        predict_id = id(caller_predict) if caller_predict else None

        full_text_parts: list[str] = []
        async for token_text in self._stream_generate_async(
            flat_prompt, temperature, max_tokens, logits_processors
        ):
            full_text_parts.append(token_text)
            chunk = _LocalStreamChunk(text=token_text, model=self.model, predict_id=predict_id)
            await send_stream.send(chunk)

        full_text = "".join(full_text_parts)
        usage = self._compute_usage(flat_prompt, full_text, max_tokens)
        response = self._build_response(full_text, usage=usage)

        if cache:
            cache_request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if schema is not None:
                cache_request["response_schema"] = json.dumps(schema, sort_keys=True)
            dspy.cache.put(cache_request, response)

        return response
