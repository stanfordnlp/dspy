"""MLX inference internals for :class:`~dspy.clients.apple_local.AppleLocalLM`.

Provides the :class:`_MLXMixin` and its supporting dataclass / helpers.
These are private implementation details — import :class:`AppleLocalLM` from
:mod:`dspy.clients.apple_local` for all public usage.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import Any, AsyncGenerator

from dspy.clients.apple_base import _flatten_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming chunk type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
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
    return _flatten_messages(messages)


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------


def _response_format_to_schema(response_format: Any) -> dict[str, Any] | None:
    """Extract a JSON schema dict from a response_format value.

    Returns the Pydantic model's JSON schema if ``response_format`` is a
    ``pydantic.BaseModel`` subclass.  Returns ``None`` for plain-dict formats
    (e.g. ``{"type": "json_object"}``), ``None``, or any non-Pydantic value
    so that callers fall through to prompt-only mode without extra work.

    Args:
        response_format: Value from ``lm_kwargs["response_format"]``.  May be a
            Pydantic ``BaseModel`` subclass, a dict, or ``None``.

    Returns:
        A JSON schema dict, or ``None`` if conversion is not possible.
    """
    try:
        from pydantic import BaseModel as PydanticBaseModel

        if isinstance(response_format, type) and issubclass(response_format, PydanticBaseModel):
            return response_format.model_json_schema()
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# MLX mixin
# ---------------------------------------------------------------------------


class _MLXMixin:
    """MLX inference methods mixed into :class:`~dspy.clients.apple_local.AppleLocalLM`.

    The host class ``__init__`` must set these attributes before calling any
    method defined here:

    Attributes:
        _mlx_model: The loaded MLX model object.
        _mlx_tokenizer: The loaded HuggingFace tokenizer.
        _schema_processor_cache: Per-instance cache of compiled outlines FSMs.
        context_window: Maximum sequence length supported by the loaded model.
    """

    _mlx_model: Any
    _mlx_tokenizer: Any
    _schema_processor_cache: dict[str, Any]
    context_window: int

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
            import mlx_lm
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for AppleLocalLM(backend='mlx'). Install it:\n    pip install mlx-lm"
            ) from exc

        logger.info("AppleLocalLM: loading model %r via mlx-lm…", model_path)
        model, tokenizer = mlx_lm.load(model_path)
        logger.info("AppleLocalLM: model loaded")
        return model, tokenizer

    def _build_schema_processor(self, schema: dict[str, Any]) -> Any | None:
        """Build (or return a cached) outlines logits processor for a JSON schema.

        Uses ``outlines`` (optional dependency) to compile a finite-state machine
        that constrains ``mlx_lm.generate()`` to produce JSON matching ``schema``.
        The compiled processor is cached on the instance so that repeated calls
        with the same schema do not re-pay the FSM compilation cost.

        Args:
            schema: A JSON Schema dict (e.g. from ``PydanticModel.model_json_schema()``).

        Returns:
            An ``outlines`` logits processor, or ``None`` if ``outlines`` is not
            installed or if compilation fails.  When ``None`` is returned,
            callers fall back to prompt-only structured output.
        """
        cache_key = json.dumps(schema, sort_keys=True)
        if cache_key in self._schema_processor_cache:
            return self._schema_processor_cache[cache_key]

        processor: Any = None
        try:
            from outlines import MLXLM
            from outlines.generator import get_json_schema_logits_processor

            outlines_model = MLXLM(self._mlx_model, self._mlx_tokenizer)
            processor = get_json_schema_logits_processor(None, outlines_model, cache_key)
            logger.debug("apple_local: compiled outlines FSM for schema (key=%d chars)", len(cache_key))
        except ImportError:
            logger.warning(
                "apple_local: outlines is not installed — response_format falls back to "
                "prompt-only mode and may not be honoured by small models. "
                "Install with: pip install 'outlines[mlxlm]'"
            )
        except Exception as exc:
            logger.warning("apple_local: failed to build outlines schema processor: %s", exc)

        self._schema_processor_cache[cache_key] = processor
        return processor

    def _generate(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        logits_processors: list[Any] | None = None,
    ) -> tuple[str, str]:
        """Run synchronous (non-streaming) MLX inference.

        Only the semantically meaningful arguments are accepted; DSPy/LiteLLM
        kwargs are stripped before this method is called.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts to format
                via the tokenizer's chat template.
            temperature: Sampling temperature forwarded to ``make_sampler``.
            max_tokens: Maximum number of tokens to generate.
            logits_processors: Optional list of logits processors (e.g. an
                ``outlines`` JSON-schema processor) forwarded to
                ``mlx_lm.generate()``.

        Returns:
            A ``(generated_text, flat_prompt)`` tuple.  ``flat_prompt`` is
            returned so the caller can compute token counts without applying
            the chat template a second time.
        """
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        flat_prompt = _apply_chat_template(self._mlx_tokenizer, messages)

        sampler = make_sampler(temp=float(temperature))
        generate_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "sampler": sampler,
            "verbose": False,
        }
        if logits_processors:
            generate_kwargs["logits_processors"] = logits_processors
        text = mlx_lm.generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=flat_prompt,
            **generate_kwargs,
        )
        return text, flat_prompt

    async def _stream_generate_async(
        self,
        flat_prompt: str,
        temperature: float,
        max_tokens: int,
        logits_processors: list[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Bridge ``mlx_lm.stream_generate()`` (sync generator) to an async generator.

        Runs the synchronous MLX generator in a thread-pool executor and forwards
        each token text to the caller via an ``asyncio.Queue``, keeping the event
        loop unblocked between tokens.

        Args:
            flat_prompt: Pre-formatted prompt string from ``_apply_chat_template``.
            temperature: Sampling temperature forwarded to ``make_sampler``.
            max_tokens: Maximum number of tokens to generate.
            logits_processors: Optional list of logits processors (e.g. an
                ``outlines`` JSON-schema processor) forwarded to
                ``mlx_lm.stream_generate()``.

        Yields:
            Individual token strings as they are produced by the model.
        """
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        sampler = make_sampler(temp=float(temperature))
        stream_kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "sampler": sampler,
        }
        if logits_processors:
            stream_kwargs["logits_processors"] = logits_processors

        def _run() -> None:
            """Generate tokens in a thread and enqueue each one for the async consumer."""
            for response in mlx_lm.stream_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=flat_prompt,
                **stream_kwargs,
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
