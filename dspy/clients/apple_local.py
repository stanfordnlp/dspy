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
from typing import Any

from dspy.clients.apple_fm import _FMChoice, _FMMessage, _FMResponse, _FMUsage
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

_SUPPORTED_BACKENDS = ("mlx", "coreml")


# ---------------------------------------------------------------------------
# Chat-template helpers
# ---------------------------------------------------------------------------


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    """Format a message list using the tokenizer's built-in chat template.

    Falls back to simple role-prefixed concatenation if the tokenizer does not
    expose ``apply_chat_template`` or if the call raises.
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

    # Fallback: reuse the same simple flattener as AppleFoundationLM
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

    Args:
        model: HuggingFace repo ID (e.g. ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``)
            or an absolute path to a local MLX model directory.  Also used as the
            DSPy cache key, so two instances pointing to the same model share a
            cache.
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
        # Expose context window so DSPy optimizers know when to stop stuffing the prompt.
        # HuggingFace tokenizers carry model_max_length in their saved config; fall back to
        # 4096 for any tokenizer that omits it (conservative but safe for most models).
        self.context_window: int = getattr(self._mlx_tokenizer, "model_max_length", 4096)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_mlx(self, model_path: str) -> tuple[Any, Any]:
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
        """Run synchronous MLX inference.

        Returns ``(generated_text, flat_prompt)`` so the caller can compute
        token counts without applying the chat template a second time.
        Only the semantically meaningful arguments are accepted; DSPy/LiteLLM
        kwargs are stripped before this method is called.
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
        """Synchronous forward pass — checks DSPy cache, then runs MLX inference."""
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
                "Streaming is not yet supported for AppleLocalLM. "
                "Call forward() for a blocking response."
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

        if cache:
            cached = dspy.cache.get(cache_request)
            if cached is not None:
                return cached

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

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FMResponse:
        """Async forward pass — delegates to ``forward`` via a thread-pool executor
        so MLX inference does not block the event loop.

        Concurrent calls are gated by a semaphore (default: 1) to prevent
        out-of-memory crashes on Apple Silicon when DSPy optimizers issue
        many parallel requests.  Raise ``max_concurrency`` at init time if
        you need higher throughput and have headroom in unified memory.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
        async with self._semaphore:
            return await asyncio.to_thread(
                self.forward, prompt=prompt, messages=messages, **kwargs
            )
